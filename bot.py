import chess
import chess.pgn as pgn
import numpy as np
from move_encoder import MoveEncoder
import math
import time
import random
from node import Game
from inference_engine import InferenceEngine
from multiprocessing import Process, Pool
import os

class ChessBot():
    def __init__(self, master=True, num_slaves=40, model=None):
        self.move_encoder = MoveEncoder()
        self.explore = 0.20
        self.init_explore = 1.0
        self.max_depth = 25
        self.same_score_threshold = 0.001
        self.epsilon = 0.00001
        self.chance_limit = 0.4 ** 10
        self.meta_data = {'choose_move_time': 0, 'backprop_time': 0, 'unexplored_moves': 0, 'explored_moves': 0, 'wait_slave': 0, 'comms_time': 0,
            'total_simulations': 0, 'callback_t': 0, 'backprops': 0, 'max_depth': 0}
        self.data_out_path = 'data_out.pgn'

        self.mp_pool = None
        self.master = master
        self.model = model
        self.available_slaves = 0
        self.total_slaves = 0
        self.slave_callbacks = {}
        self.backprop_queue = []
        self.completed_callbacks = []

        if master:
            if self.model == None:
                self.model = InferenceEngine()
                self.model.start()
            if num_slaves > 0:
                self.available_slaves = num_slaves
                self.total_slaves = num_slaves
                # launch slave processes
                self.mp_pool = Pool(processes=num_slaves, initializer=init_slave, initargs=(self.model,))
        self.game = Game(self.model)

    def best_move(self, board, depth=5, time_limit=10, debug=False, eval_freq=15):
        self.meta_data['max_depth'] = 0
        self.game.set_position(board)
        if debug:
            print(self.game.root())

        best_move = self.mcts(depth, time_limit, debug, eval_freq)
        return best_move

    def mcts(self, depth, time_limit, debug, eval_freq):
        cutoff_time = time.time() + time_limit
        eval_time = time.time() + eval_freq

        node = self.game.node()
        board = self.game.board
        init_depth = depth
        bonus_depth = 0
        if board.halfmove_clock > 15:
            if 101 - board.halfmove_clock > depth:
                depth = 101 - board.halfmove_clock
                if depth > self.max_depth:
                    depth = self.max_depth
            # bonus_depth = 51 - depth
        while True:
            # stale game search
            # if node.visits > 500 and depth < self.max_depth:
                # depth = init_depth + node.visits // 1200
                # if self.meta_data['total_simulations'] % 10 == 0:
                #     if board.halfmove_clock > 10:
                #         bonus_depth += 1
                #     elif self.meta_data['total_simulations'] % 40 == 0:
                #         if node.score < 1 and node.score > 0.98:
                #             bonus_depth += 1
                #         elif node.score > 0 and node.score < 0.02:
                #             bonus_depth += 1
                # depth += bonus_depth
              
            # simulate game
            self.simulate_game(depth)
            self.meta_data['total_simulations'] += 1
            t1 = time.time()
            self.clear_callback_queue()
            self.clear_backprop_queue()
            self.meta_data['backprop_time'] += time.time() - t1

            # out of time
            if time.time() > cutoff_time:
                break

            if time.time() > eval_time or len(node.child_links) == 1:
                # end early if confident enough
                moves = []
                for move, link in node.child_links.items():
                    if link.node != None:
                        moves.append((move, link))

                formatted_moves = []
                for i, m in enumerate(moves):
                    move = {'move': m[0], 'weight': m[1].weight, 'score': 1 - m[1].node.score, 'visits': m[1].node.visits}
                    formatted_moves.append(move)
                formatted_moves.sort(key=lambda x: x['score'], reverse=True)
                
                if debug:
                    print('total simulations:', node.visits, 'depth:', depth, 'max depth:', self.meta_data['max_depth'])
                    for m in formatted_moves[:3]:
                        print(m)

                if len(formatted_moves) == 1 or formatted_moves[0]['visits'] * (formatted_moves[0]['score']) > formatted_moves[1]['visits'] * (formatted_moves[1]['score']) * 20:
                    return formatted_moves[0]
                eval_time = time.time() + eval_freq

        self.sync_slaves()
        self.clear_backprop_queue()
        moves = []
        for move, link in node.child_links.items():
            if link.node != None:
                moves.append((move, link))
        lower_bounds = self.calc_confidence_bound(node.visits, moves, lower_bound=True)

        formatted_moves = []
        for i, m in enumerate(moves):
            move = {'move': m[0], 'weight': m[1].weight, 'score': 1 - m[1].node.score, 'visits': m[1].node.visits, 'lower_bound': lower_bounds[i]}
            formatted_moves.append(move)
        formatted_moves.sort(key=lambda x: x['lower_bound'], reverse=True)
        best_move = formatted_moves[0]

        if debug:
            print('total simulations:', node.visits, 'depth:', depth, 'max depth:', self.meta_data['max_depth'])
            for m in formatted_moves[:3]:
                print(m)

        best_moves = []
        for m in formatted_moves:
            if m['lower_bound'] > best_move['lower_bound'] - 0.01:
                best_moves.append(m)
            else:
                break

        # if scores are the same choose move based of instinct
        best_moves.sort(key=lambda x: x['weight'], reverse=True)
        if best_moves[0] != best_move:
            best_move = best_moves[0]
            if debug:
                print('best move')
                print(best_move)

        return best_move

    def simulate_game(self, depth, chance=1):
        node = self.game.node()
        if len(node.child_links) == 0 or (depth <= 0 and chance < self.chance_limit):
            self.backprop_queue.append(self.game.node_stack[:])
            if depth < self.meta_data['max_depth']:
                self.meta_data['max_depth'] = depth
            return node

        t1 = time.time()
        potential_moves = self.choose_next_moves(node, 0, 0)
        main_eval, other_evals = self.assign_slaves(node, potential_moves)
        self.meta_data['choose_move_time'] += time.time() - t1

        # launch slave processes
        for move in other_evals:
            link = node.child_links[move]
            response = self.mp_pool.apply_async(slave_simulate, args=[self.game.board.copy(), link.node, move, depth-1, id(link), time.time(), chance * link.weight], callback=self.slave_callback)
            self.slave_callbacks[id(link)] = {'response': response, 'node': node, 'move': move, 'node_stack': self.game.node_stack[:]}
            self.available_slaves -= 1

        link = node.child_links[main_eval]
        self.game.push(main_eval, depth-1)
        sample_node = self.simulate_game(depth-1, chance=chance * link.weight)
        self.game.pop()
        return node

    def assign_slaves(self, node, potential_moves):
        # returns move to evaluate on main proc and list of moves to evaluate on slave procs
        if self.total_slaves == 0:
            return potential_moves[0], []

        moves_to_eval = []
        # skip moves already being evaluated by slaves
        for move in potential_moves:
            if not id(node.child_links[move]) in self.slave_callbacks:
                moves_to_eval.append(move)

        # if no other moves to explore wait for slave process to complete
        if len(moves_to_eval) == 0:
            main_eval = potential_moves[0]
            t1 = time.time()
            self.slave_callbacks[id(node.child_links[main_eval])]['response'].get()
            self.meta_data['wait_slave'] += time.time() - t1
            return main_eval, []

        main_eval_index = 0
        # prioritise explored moves on main proc
        if self.available_slaves > 0:
            for i, move in enumerate(moves_to_eval):
                if node.child_links[move].node != None:
                    main_eval_index = i
                    break
        main_eval = moves_to_eval.pop(main_eval_index)
        other_eval = []
        for move in moves_to_eval:
            if node.child_links[move].node == None:
                other_eval.append(move)
        other_eval = other_eval[:self.available_slaves]
        return main_eval, other_eval

    def choose_next_moves(self, node, unexplored_lookahead=0, explored_lookahead=0):
        potential_visits = node.visits + unexplored_lookahead
        top_weight = 0
        unexplored_options = []
        explored_options = []
        for move, link in node.child_links.items():
            if link.weight > top_weight:
                top_weight = link.weight
            if link.node == None:
                unexplored_options.append((move, link))
            else:
                explored_options.append((move, link))

        next_moves = []

        if len(unexplored_options) > 0:
            t1 = time.time()
            weights = np.zeros(len(unexplored_options))
            for i in range(len(unexplored_options)):
                link = unexplored_options[i][1]
                weights[i] = link.weight
            
            weights /= top_weight  # scale weights based on certainty
            weights += self.epsilon
            option_scores = (self.init_explore * potential_visits) - (1 / weights)

            good_options = np.argwhere(option_scores > 0).flatten().tolist()
            good_options.sort(key=lambda x: option_scores[x], reverse=True)
            self.meta_data['unexplored_moves'] += time.time() - t1
            for option_index in good_options:
                next_moves.append(unexplored_options[option_index][0])

            if len(explored_options) == 0:
                if len(next_moves) == 0:
                    best_option_index = np.argmax(option_scores)
                    next_moves.append(unexplored_options[best_option_index][0])
                return next_moves

        if len(explored_options) > 0:
            potential_visits = node.visits + explored_lookahead
            t1 = time.time()
            option_scores = self.calc_confidence_bound(potential_visits, explored_options)
            best_option_index = np.argmax(option_scores)
            best_option = explored_options[best_option_index]
            best_score =  1- best_option[1].node.score
            good_options = np.argwhere(option_scores > best_score).flatten().tolist()
            good_options.sort(key=lambda x: option_scores[x], reverse=True)
            for option_index in good_options:
                next_moves.append(explored_options[option_index][0])
            self.meta_data['explored_moves'] += time.time() - t1
        return next_moves

    def calc_confidence_bound(self, simulations_num, options, lower_bound=False):
        scores = np.zeros(len(options))
        visits = np.zeros(len(options))
        for i in range(len(options)):
            link = options[i][1]
            scores[i] = link.node.score
            visits[i] = link.node.visits
        scores = 1 - scores
        visits += 1
        log = math.log(simulations_num+1)
        confidence_range = self.explore * np.sqrt(log / visits)
        if lower_bound:
            confidence_range *= -1
        option_scores = scores + confidence_range
        return option_scores

    def clear_backprop_queue(self):
        while len(self.backprop_queue) > 0:
            self.meta_data['backprops'] += 1
            self.backprop(self.backprop_queue.pop(0))

    def backprop(self, node_stack):
        if len(node_stack) == 0:
            return
        node = node_stack.pop()
        # backprop weighted score between move with highest lower confidence bound and best score

        node.visits += 1
        moves = []
        potential_move = None
        total_scores = 0
        total_visits = 0
        for move, link in node.child_links.items():
            if link.node != None:
                moves.append((move, link))
                total_scores += link.node.score * (link.node.visits + 1)
                total_visits += link.node.visits + 1
                if potential_move == None or link.node.score < potential_move.score:
                    potential_move = link.node

        if len(moves) > 0:
            # lower_bounds = self.calc_confidence_bound(node.visits, moves, lower_bound=True)
            # best_move_index = np.argmax(lower_bounds)
            # best_move = moves[best_move_index][1].node

            # node.score = ((1 - best_move.score) * (best_move.visits+1) + (1 - potential_move.score) * (potential_move.visits+1)) / (best_move.visits + potential_move.visits + 2)
            # node.score = ((1 - best_move.score) + (1 - potential_move.score)) / 2
            # node.score = (1 - potential_move.score)
            # node.score = 1 - best_move.score

            best_score_weight = math.log(len(moves))

            total_scores += potential_move.score * (potential_move.visits + 1) * best_score_weight
            total_visits += (potential_move.visits + 1) * best_score_weight

            # total_scores += best_move.score * (best_move.visits + 1)
            # total_visits += (best_move.visits + 1)

            node.score = 1 - total_scores / total_visits
        self.backprop(node_stack)

    def train_from_board(self, b, result=None):
        board = b.copy()
        # save game
        g = pgn.Game.from_board(board)
        with open(self.data_out_path, "a") as f:
            print(g, file=f, end='\n\n')
        if result == None:
            result = board.result(claim_draw=True)
        winner = 2
        if result == '1-0':
            winner = chess.WHITE
        elif result == '0-1':
            winner = chess.BLACK
            
        inputs_board_state = np.zeros(shape=(len(board.move_stack), 8, 8, 12), dtype=np.int8)
        inputs_castling = np.zeros(shape=(len(board.move_stack), 4), dtype=np.int8)
        outputs_move = np.zeros(shape=(len(board.move_stack), len(self.game.move_encoder.moves)), dtype=np.int8)
        outputs_value = np.zeros(shape=(len(board.move_stack), 2), dtype=np.int8)
        i = 0
        while len(board.move_stack) > 0:
            move = board.pop().uci()
            move_index = self.game.move_encoder.uci_to_index(move, board.turn)
            board_matrix, castling_matrix = self.game.input_matrix(board)
            inputs_board_state[i] = board_matrix
            inputs_castling[i] = castling_matrix
            outputs_move[i][move_index] = 1
            outputs_value[i] = [1, 0] if winner == board.turn else [0, 1]
            if winner == 2:
                outputs_value[i] = [0.5, 0.5]
            i += 1

        self.model.fit([inputs_board_state, inputs_castling], [outputs_move, outputs_value])

    def clear_callback_queue(self):
        t1 = time.time()
        while len(self.completed_callbacks) > 0:
            cb_id = self.completed_callbacks.pop()
            callback_data = self.slave_callbacks[cb_id]
            node = callback_data['node']
            move = callback_data['move']
            new_node = callback_data['new_node']
            node.set_child(move, new_node)
            self.backprop_queue.append(callback_data['node_stack'])
            del self.slave_callbacks[cb_id]
            self.available_slaves += 1
        self.meta_data['callback_t'] += time.time() - t1

    def slave_callback(self, data):
        new_node, cb_id = data
        # join parallel processes
        self.slave_callbacks[cb_id]['new_node'] = new_node
        self.completed_callbacks.append(cb_id)

    def print_stats(self, i=0):
        print(i, self.meta_data, self.game.meta_data)
        if self.total_slaves > 0:
            sum1 = dict()
            sum2 = dict()
            for key, val in self.meta_data.items():
                sum1[key] = val
            for key, val in self.game.meta_data.items():
                sum2[key] = val
            self.sync_slaves()
            for data1, data2 in self.mp_pool.map(slave_stats, list(range(1, self.available_slaves+1))):
                for key, val in data1.items():
                    sum1[key] += val
                for key, val in data2.items():
                    sum2[key] += val
            print('sum', sum1, sum2)
        return self.meta_data, self.game.meta_data

    def sync_slaves(self):
        remaining_tasks = list(self.slave_callbacks.values())
        for callback_data in remaining_tasks:
            callback_data['response'].get()
        self.clear_callback_queue()
        assert(self.available_slaves == self.total_slaves)

class SlaveBot(ChessBot):
    def __init__(self, model):
        ChessBot.__init__(self, master=False, model=model)

def init_slave(model):
    global slave
    slave = SlaveBot(model)
    print("Spawning slave process", os.getpid())


def slave_simulate(board, node, move, depth, id, timestamp, chance):
    slave.meta_data['total_simulations'] += 1
    slave.game.board = board
    slave.meta_data['comms_time'] += time.time() - timestamp
    slave.game.board.push_uci(move)
    if node == None:
        node = slave.game.expand()
    slave.game.node_stack = [node]
    new_node = slave.simulate_game(depth, chance=chance)
    slave.clear_backprop_queue()
    return new_node, id

def slave_stats(i):
    time.sleep(0.2)
    return slave.print_stats(i)

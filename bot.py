import chess
import numpy as np
from model_densenet import get_model, save_model
from tensorflow.python.client import device_lib
from move_encoder import MoveEncoder
import math
import time
import random
from node import Game

class ChessBot():
    def __init__(self):
        self.move_encoder = MoveEncoder()
        gpus = len(device_lib.list_local_devices()) - 1
        self.model, self.model_template = get_model(len(self.move_encoder.moves), gpus)
        self.explore = 0.4
        self.init_explore = 1.1
        self.max_depth = 35
        self.same_score_threshold = 0.001
        self.epsilon = 0.0005
        self.game = Game(self.model)
        self.meta_data = {'choose_move_time': 0, 'backprop_time': 0, 'unexplored_moves': 0, 'explored_moves': 0, 'unexplored_maths': 0}

    def save_model(self):
        save_model(self.model_template)

    def best_move(self, board, depth=7, time_limit=10, debug=False, eval_freq=15):
        self.game.set_position(board)
        if debug:
            print(self.game.root())

        best_move = self.mcts(depth, time_limit, debug, eval_freq)
        return best_move

    def mcts(self, depth, time_limit, debug, eval_freq):
        cutoff_time = time.time() + time_limit
        eval_time = time.time() + eval_freq

        if len(self.game.next_moves()) == 1:
            return self.game.next_moves()[0]

        node = self.game.node()
        board = self.game.board
        while True:
            # stale game search
            if node.visits > 100:
                if depth < self.max_depth and node.visits % 10 == 0:
                    if board.halfmove_clock > 10:
                        depth += 1
                    elif node.score < 1 and node.score > 0.98:
                        depth += 1
                    elif node.score > 0 and node.score < 0.02:
                        depth += 1

            # simulate game
            self.simulate_game(depth)

            # out of time
            if time.time() > cutoff_time:
                break

            if time.time() > eval_time:
                moves = sorted(node.children(), key=lambda x: x['visits'] * (1-x['score']), reverse=True)
                # end early if confident enough
                
                if debug:
                    print('total simulations:', node.visits, 'depth:', depth)
                    for m in moves[:3]:
                        print(m)

                if len(moves) == 1 or moves[0]['visits'] * (1-moves[0]['score']) > moves[1]['visits'] * (1-moves[1]['score']) * 20:
                    return moves[0]
                eval_time = time.time() + eval_freq

        moves = []
        for move, link in node.child_links.items():
            if link.node != None and link.node.visits > 0:
                moves.append((move, link))
        lower_bounds = self.calc_confidence_bound(node.visits, moves, lower_bound=True)

        formatted_moves = []
        for i, m in enumerate(moves):
            move = {'move': m[0], 'weight': m[1].weight, 'score': 1 - m[1].node.score, 'visits': m[1].node.visits, 'lower_bound': lower_bounds[i]}
            formatted_moves.append(move)
        formatted_moves.sort(key=lambda x: x['lower_bound'], reverse=True)
        best_move = formatted_moves[0]

        if debug:
            print('total simulations:', node.visits, 'depth:', depth)
            for m in formatted_moves[:3]:
                print(m)

        best_moves = []
        for m in formatted_moves:
            if m['score'] > best_move['score'] - self.same_score_threshold and m['visits'] >= best_move['visits'] - 2:
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

    def simulate_game(self, depth):
        node = self.game.node()
        node.visits += 1
        if len(node.child_links) == 0 or depth == 0:
            return node.score

        t1 = time.time()
        sample_move = self.choose_next_move(node)

        self.meta_data['choose_move_time'] += time.time() - t1
        self.game.push(sample_move, depth)
        sample_score = 1 - self.simulate_game(depth-1)
        sample_node = self.game.node()
        self.game.pop()

        sample_score = 1 - sample_node.score
        # node.score = ((node.visits - 1) * node.score + sample_score) / node.visits
        # return sample_score
        
        t1 = time.time()

        # backprop score of move with highest lower confidence bound
        moves = []
        potential_move = None
        for move, link in node.child_links.items():
            if link.node != None and link.node.visits > 0:
                moves.append((move, link))
                if potential_move == None or link.node.score < potential_move.score:
                    potential_move = link.node
        
        lower_bounds = self.calc_confidence_bound(node.visits, moves, lower_bound=True)
        best_move_index = np.argmax(lower_bounds)
        best_move = moves[best_move_index][1].node

        node.score = ((1 - best_move.score) * best_move.visits + (1 - potential_move.score) * potential_move.visits) / (best_move.visits + potential_move.visits)

        self.meta_data['backprop_time'] += time.time() - t1
        return sample_score

    def choose_next_move(self, node):
        top_weight = 0
        unexplored_options = []
        explored_options = []
        for move, link in node.child_links.items():
            if link.weight > top_weight:
                top_weight = link.weight
            if link.node == None or link.node.visits == 0:
                unexplored_options.append((move, link))
            else:
                explored_options.append((move, link))

        t1 = time.time()
        best_option = None
        best_option_score = None
        if len(unexplored_options) > 0:
            weights = np.zeros(len(unexplored_options))
            for i in range(len(unexplored_options)):
                link = unexplored_options[i][1]
                weights[i] = link.weight
            
            weights /= top_weight  # scale weights based on certainty
            weights += self.epsilon
            option_scores = (self.init_explore * node.visits) - (1 / weights)
            best_option_index = np.argmax(option_scores)
            best_option_score = option_scores[best_option_index]
            best_option = unexplored_options[best_option_index][0]
        self.meta_data['unexplored_moves'] += time.time() - t1
        
        if len(explored_options) == 0 or (best_option_score != None and best_option_score > 0):
            return best_option

        t1 = time.time()
        option_scores = self.calc_confidence_bound(node.visits, explored_options)
        best_option_index = np.argmax(option_scores)
        best_option = explored_options[best_option_index][0]
        self.meta_data['explored_moves'] += time.time() - t1
        return best_option

    def calc_confidence_bound(self, simulations_num, options, lower_bound=False):
        scores = np.zeros(len(options))
        visits = np.zeros(len(options))
        for i in range(len(options)):
            link = options[i][1]
            scores[i] = link.node.score
            visits[i] = link.node.visits
        scores = 1 - scores
        visits += 1
        log = math.log(simulations_num)
        confidence_range = self.explore * np.sqrt(log / visits)
        if lower_bound:
            confidence_range *= -1
        option_scores = scores + confidence_range
        return option_scores

    def train_from_board(self, b):
        board = b.copy()
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

        self.model.fit([inputs_board_state, inputs_castling], [outputs_move, outputs_value], verbose=1, batch_size=32)
        self.save_model()

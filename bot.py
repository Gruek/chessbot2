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
        self.explore = 0.3
        self.init_explore = 1
        self.max_depth = 35
        self.same_score_threshold = 0.001
        self.epsilon = 0.0005
        self.game = Game(self.model)
        self.meta_data = {'choose_move_time': 0, 'backprop_time': 0, 'maths_time': 0}

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
                moves = sorted(self.game.next_moves(), key=lambda x: x['visits'] * (1-x['score']), reverse=True)
                # end early if confident enough
                
                if debug:
                    print('total simulations:', node.visits, 'depth:', depth)
                    for m in moves[:3]:
                        print(m)

                if moves[0]['visits'] * (1-moves[0]['score']) > moves[1]['visits'] * (1-moves[1]['score']) * 20:
                    return moves[0]
                eval_time = time.time() + eval_freq

        moves = sorted(self.game.next_moves(), key=lambda x: x['visits'] * (1-x['score']), reverse=True)
        best_score = 1 - moves[0]['score']
        visits = moves[0]['visits']
        best_moves = []
        for m in moves:
            m_score = 1 - m['score']
            if m_score > best_score - self.same_score_threshold and m['visits'] >= visits - 2:
                best_moves.append(m)
            else:
                break

        # if scores are the same choose move based of instinct
        best_moves.sort(key=lambda x: x['weight'], reverse=True)
        best_move = best_moves[0]
        if moves[0] != best_move:
            print('best move')
            print(best_move)

        if debug:
            print('total simulations:', node.visits, 'depth:', depth)
            for m in moves[:3]:
                print(m)
        return best_move

    def simulate_game(self, depth):
        node = self.game.node()
        node.visits += 1
        if len(node.child_links) == 0 or depth == 0:
            return

        t1 = time.time()
        sample_move = self.choose_next_move(node)
        self.meta_data['choose_move_time'] += time.time() - t1
        self.game.push(sample_move)

        self.simulate_game(depth-1)
        # sample_node = self.game.node()
        self.game.pop()
        
        t1 = time.time()
        # best_move = None
        # best_confidence_score = None
        # for move in node.children():
        #     conf_score = move['visits'] * (1 - move['score'])
        #     if best_move == None or conf_score > best_confidence_score:
        #         best_move = move
        #         best_confidence_score = conf_score
        # best_score = 1 - best_move['score']
        # node.score = best_score

        best_move = None
        most_visited_move = None
        for move in node.children():
            if best_move == None or move['score'] < best_move['score']:
                best_move = move
            if most_visited_move == None or move['visits'] > most_visited_move['visits']:
                most_visited_move = move
        
        best_move_score = 1 - best_move['score']
        most_visited_move_score = 1 - most_visited_move['score']
        # if checkmate is found, incentivise shortest path to victory
        if best_move_score > 1 or best_move_score < 0:
            best_move_score *= 1.01
        if most_visited_move_score > 1 or most_visited_move_score < 0:
            most_visited_move_score *= 1.01

        node.score = (best_move['visits'] * best_move_score + most_visited_move['visits'] * most_visited_move_score) / (best_move['visits'] + most_visited_move['visits'])

        self.meta_data['backprop_time'] += time.time() - t1

    def choose_next_move(self, node):
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
        weight_multiplier = 1 / top_weight

        # evaluate unexplored options
        best_option = None
        best_option_score = -1
        for move, link in unexplored_options:
            option_score = self.init_explore * node.visits - 1 / (link.weight * weight_multiplier + self.epsilon)
            if option_score > best_option_score:
                best_option = move
                best_option_score = option_score

        if len(explored_options) == 0 or best_option_score > 0:
            return best_option
        
        # evaluate explored options
        best_option = None
        best_option_score = None
        t1 = time.time()
        log = math.log(node.visits)
        self.meta_data['maths_time'] += time.time() - t1
        for move, link in explored_options:
            option_score = 1 - link.node.score + self.explore * math.sqrt(log / (link.node.visits + 1))
            if best_option_score == None or option_score > best_option_score:
                best_option = move
                best_option_score = option_score
        return best_option

    def train_from_board(self, board):
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

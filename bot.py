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
        self.explore = 0.2
        self.init_explore = 1.2
        self.max_depth = 35
        self.same_score_threshold = 0.001
        self.epsilon = 0.0005
        self.game = Game(self.model)

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
            return self.game.next_moves()[0]['move']

        node = self.game.node()
        board = self.game.board
        while True:
            # stale game search
            if node.visits > 100:
                if depth < self.max_depth and node.visits % 10 == 0:
                    if board.halfmove_clock > 15:
                        depth += 1
                    elif node.score < 1 and node.score > 0.98:
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
                    print(moves[:3])

                if moves[0]['visits'] > moves[1]['visits'] * 10:
                    return moves[0]
                eval_time = time.time() + eval_freq

        moves = sorted(self.game.next_moves(), key=lambda x: x['visits'] * (1-x['score']), reverse=True)

        if debug:
            print('total simulations:', node.visits, 'depth:', depth)
            print(moves[:3])

        # best_score = moves[0]['score']
        # # if mcts scores are similar then choose move based of policy model score
        # best_moves = []
        # for move in moves:
        #     if move['score'] > best_score - self.same_score_threshold:
        #         best_moves.append(move)
        # best_moves = sorted(best_moves, key=lambda x: x['weight'], reverse=True)
        # if best_moves[0] != moves[0]:
        #     if debug:
        #         print('best move:')
        #         print(best_moves[0])
        #     return best_moves[0]
        return moves[0]

    def simulate_game(self, depth):
        node = self.game.node()
        node.visits += 1
        if len(node.child_links) == 0 or depth == 0:
            return

        sample_move = self.choose_next_move(node)
        self.game.push(sample_move)
        self.simulate_game(depth-1)
        sample_node = self.game.node()
        sample_score = 1 - sample_node.score
        # if checkmate is found, incentivise shortest path to victory
        if sample_score > 1:
            sample_score *= 1.01
        self.game.pop()
        
        best_move = None
        best_confidence_score = None
        for move in node.children():
            conf_score = move['visits'] * (1 - move['score'])
            if best_move == None or conf_score > best_confidence_score:
                best_move = move
                best_confidence_score = conf_score
        best_score = 1 - best_move['score']
        # best_score_weight = best_move['visits'] / (best_move['visits'] + sample_node.visits)

        # score_delta = best_score * best_score_weight + (1 - best_score_weight) * sample_score

        # node.score = (node.score * (node.visits-1) + score_delta ) / node.visits
        # node.score = sample_score
        node.score = best_score

    def choose_next_move(self, node):
        top_weight = 0
        for link in node.child_links.values():
            if link.weight > top_weight:
                top_weight = link.weight
        weight_multiplier = 1 / top_weight
        next_move = None
        max_ucb = None
        for uci, link in node.child_links.items():
            ucb = self.calc_ucb(link, node.visits, weight_multiplier=weight_multiplier)
            if max_ucb == None or ucb > max_ucb:
                max_ucb = ucb
                next_move = uci
        return next_move

    def calc_ucb(self, node_link, simulation_num, multiplier=1, weight_multiplier=1):
        if node_link.node == None:
            # if node hasn't been explored then use weight to determine when it should be visited
            return self.init_explore * simulation_num - 1 / (node_link.weight * weight_multiplier + self.epsilon)
        # otherwise calculate upper confidence bound
        return 1 - node_link.node.score + self.explore * math.sqrt(math.log(simulation_num) / (node_link.node.visits + 1)) * multiplier

    def train_from_board(self, board):
        result = board.result()
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

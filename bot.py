import chess
import numpy as np
from model_resnet import get_model, save_model
from tensorflow.python.client import device_lib
from move_encoder import MoveEncoder
import math
import time
import random

class ChessBot():
    def __init__(self, next_move_chooser='ucb'):
        self.move_encoder = MoveEncoder()
        self.model = get_model(len(self.move_encoder.moves))
        self.cache = dict()
        self.inferences = 0
        self.cache_retrieval = 0
        self.next_move_chooser = next_move_chooser

    def save_model(self):
        save_model(self.model)

    def board_to_input(self, board):
        p1_color = board.turn
        board_matrix = np.zeros(shape=(8, 8, 12), dtype=np.int8)
        for rank in range(8):
            for file in range(8):
                piece = board.piece_at(rank*8+file)
                if piece:
                    piece_p2 = piece.color != p1_color
                    piece_idx = piece_p2 * 6 + piece.piece_type - 1
                    #rotate the board for black
                    r = rank if p1_color == chess.WHITE else 7 - rank
                    board_matrix[file][r][piece_idx] = 1
        castling_matrix = np.zeros(shape=(4,), dtype=np.int8)
        white_castling = [int(bool(board.castling_rights & chess.BB_A1)), int(bool(board.castling_rights & chess.BB_H1))]
        black_castling = [int(bool(board.castling_rights & chess.BB_A8)), int(bool(board.castling_rights & chess.BB_H8))]
        castling_matrix[0:2] = white_castling if p1_color == chess.WHITE else black_castling
        castling_matrix[2:4] = black_castling if p1_color == chess.WHITE else white_castling
        return board_matrix, castling_matrix

    def best_move_old(self, fen, depth=7, chance_limit=0.01):
        board = chess.Board(fen=fen)
        moves, value = self.run_network(board, board.turn)
        best_move = None
        pov = board.turn
        for move in moves:
            board.push_uci(move['move'])
            move['search_score'] = self.eval_move(board, 1, pov, depth, chance_limit)
            board.pop()
            if not best_move or move['search_score'] > best_move['search_score']:
                best_move = move
        return best_move

    def best_move(self, fen, depth=10, time_limit=10, debug=False):
        board = chess.Board(fen=fen)
        best_move = self.mcts(board, depth, time_limit, debug)
        return best_move

    def mcts(self, board, depth, time_limit, debug):
        self.inferences = 0
        pov = board.turn
        moves, value = self.run_network(board, pov)
        simulation_num = 0
        cutoff_time = time.time() + time_limit

        for move in moves:
            simulation_num += move['visit_count']

        while True:
            # out of time
            if time.time() > cutoff_time:
                if debug:
                    print('total simulations', simulation_num)
                    print(moves)
                
                best_move = None
                best_score = -1
                for move in moves:
                    score = move['mcts_score']
                    if score > best_score:
                        best_score = score
                        best_move = move
                # self.cache[board.fen()] = moves, value
                return best_move

            # choose which move to explore next
            next_move = self.choose_next_move(moves)

            # simulate game
            board.push_uci(next_move['move'])
            sample_score = self.simulate_game(board, pov, depth)
            board.pop()
            next_move['total_score'] += sample_score
            next_move['visit_count'] += 1
            next_move['mcts_score'] = next_move['total_score'] / next_move['visit_count']
            next_move['mcts_score'] = 0.75 * next_move['mcts_score'] + 0.25 * sample_score
            simulation_num += 1


    def calc_ucb(self, move, simulation_num):
        # calculate upper confidence bound for move
        # https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
        if move['visit_count'] == 0:
            return move['score'] + math.sqrt(math.log(simulation_num + 1) / 1) 
        return move['mcts_score'] + 0.5 * math.sqrt(math.log(simulation_num) / move['visit_count'])

    def simulate_game(self, board, pov, depth):
        result = board.result()
        if result != '*':
            # end of game
            if result == '1/2-1/2':
                return 0.5
            else:
                return 1

        moves, value = self.run_network(board, pov)
        if depth == 0:
            return 1 - value
        
        if self.next_move_chooser == 'ucb':
            next_move = self.choose_next_move(moves)
        else:
            next_move = self.choose_next_move_uniformly(moves)
        board.push_uci(next_move['move'])
        sample_score = self.simulate_game(board, pov, depth-1)
        board.pop()
        next_move['total_score'] += sample_score
        next_move['visit_count'] += 1
        next_move['mcts_score'] = next_move['total_score'] / next_move['visit_count']
        next_move['mcts_score'] = 0.75 * next_move['mcts_score'] + 0.25 * sample_score
        best_score = -1
        for move in moves:
            if move['mcts_score'] > best_score:
                best_score = move['mcts_score']
        self.cache[board.fen()] = moves, value
        return 1 - best_score

    def choose_next_move(self, moves):
        total_simulations = 0
        for move in moves:
            total_simulations += move['visit_count']
        next_move = None
        max_ucb = -1
        for move in moves:
            ucb = self.calc_ucb(move, total_simulations)
            if ucb > max_ucb:
                max_ucb = ucb
                next_move = move
        return next_move

    def choose_next_move_uniformly(self, moves):
        rand = random.random()
        running_score = 0
        for move in moves:
            running_score += move['score']
            if running_score >= rand:
                return move

    def validate_moves(self, board, moves_array):
        legal_moves = np.zeros(shape=(len(self.move_encoder.moves),), dtype=np.int8)
        for move in board.legal_moves:
            uci = move.uci()
            move_index = self.move_encoder.uci_to_index(uci, board.turn)
            legal_moves[move_index] = 1
        valid_moves = np.multiply(legal_moves, moves_array)
        valid_moves /= np.sum(valid_moves)
        return valid_moves

    def run_network(self, board, pov):
        fen = board.fen()
        if fen in self.cache:
            self.cache_retrieval += 1
            return self.cache[fen]
        self.inferences += 1

        board_inputs = np.zeros(shape=(1, 8, 8, 12), dtype=np.int8)
        castling_inputs = np.zeros(shape=(1, 4), dtype=np.int8)

        # input
        board_matrix, castling_matrix = self.board_to_input(board)
        board_inputs[0] = board_matrix
        castling_inputs[0] = castling_matrix

        # run model
        policies, values = self.model.predict([board_inputs, castling_inputs])

        self.cache[fen] = self.format_model_output(policies[0], values[0], board, pov)
        return self.cache[fen]

    def format_model_output(self, policy, value, board, pov):
        moves_array = self.validate_moves(board, policy)
        move_scores = []
        for move_index, score in enumerate(moves_array):
            if score > 0:
                move_score = {
                    'move': self.move_encoder.index_to_uci(move_index, board.turn),
                    'score': score,
                    'mcts_score': 0,
                    'total_score': 0,
                    'visit_count': 0
                }
                move_scores.append(move_score)
        state_score = value[0]
        return move_scores, state_score

    def clear_cache(self):
        self.cache = dict()

import chess
import numpy as np
from model_densenet import get_model, save_model
from tensorflow.python.client import device_lib
from move_encoder import MoveEncoder
import math
import time
import random

class ChessBot():
    def __init__(self):
        self.move_encoder = MoveEncoder()
        gpus = len(device_lib.list_local_devices()) - 1
        self.model, self.model_template = get_model(len(self.move_encoder.moves), gpus)
        self.cache = dict()
        self.inferences = 0
        self.cache_retrieval = 0
        self.explore = 0.3
        self.init_explore = 1.5
        self.max_depth = 35
        self.same_score_threshold = 0.001

    def save_model(self):
        save_model(self.model_template)

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

    def best_move(self, board, depth=7, time_limit=10, debug=False, end_early_eval=16):
        best_move = self.mcts(board.copy(), depth, time_limit, debug, end_early_eval)
        return best_move

    def mcts(self, board, depth, time_limit, debug, end_early_eval):
        self.inferences = 0
        moves, value = self.run_network(board)
        simulation_num = 0
        cutoff_time = time.time() + time_limit
        end_early_eval_time = time.time() + end_early_eval

        for move in moves:
            simulation_num += move['visit_count']
        if len(moves) == 1:
            return moves[0]

        while True:
            # choose which move to explore next
            next_move = self.choose_next_move(moves, simulation_num)

            # stale game search
            if simulation_num > 100:
                if depth < self.max_depth and simulation_num % 10 == 0:
                    if board.halfmove_clock > 15:
                        depth += 1
                    elif next_move['mcts_score'] < 1 and next_move['mcts_score'] > 0.98:
                        depth += 1

            # simulate game
            board.push_uci(next_move['move'])
            sample_score = self.simulate_game(board, depth)
            board.pop()
            next_move['total_score'] += sample_score
            next_move['visit_count'] += 1
            next_move['mcts_score'] = next_move['total_score'] / next_move['visit_count']
            # next_move['mcts_score'] = sample_score
            simulation_num += 1

            # out of time
            if time.time() > cutoff_time:
                self.cache[board.fen()] = moves, value
                sorted_moves = sorted(moves, key=lambda x: x['mcts_score'], reverse=True)
                if debug:
                    print('total simulations:', simulation_num, 'depth:', depth)
                    print(sorted_moves[:3])

                best_score = sorted_moves[0]['mcts_score']
                # if mcts scores are similar then choose move based of model score
                best_moves = []
                for move in sorted_moves:
                    if move['mcts_score'] > best_score - self.same_score_threshold:
                        best_moves.append(move)
                best_moves = sorted(best_moves, key=lambda x: x['score'], reverse=True)
                if best_moves[0] != sorted_moves[0]:
                    if debug:
                        print('best move:')
                        print(best_moves[0])
                    return best_moves[0]
                return sorted_moves[0]

            if time.time() > end_early_eval_time:
                sorted_moves = sorted(moves, key=lambda x: x['mcts_score'], reverse=True)
                # end early if confident enough
                best_move_lower_bound = self.calc_ucb(sorted_moves[0], simulation_num, multiplier=-1)
                other_move_upper_bound = None
                for move in sorted_moves[1:]:
                    ucb = self.calc_ucb(move, simulation_num, multiplier=0.7)
                    if other_move_upper_bound == None or ucb > other_move_upper_bound:
                        other_move_upper_bound = ucb
                if debug:
                    print('total simulations:', simulation_num, 'depth:', depth)
                    print(sorted_moves[:3])
                    print('lb:', best_move_lower_bound, 'ub:', other_move_upper_bound)
                if best_move_lower_bound > other_move_upper_bound:
                    self.cache[board.fen()] = moves, value
                    return sorted_moves[0]
                end_early_eval_time = time.time() + end_early_eval


    def calc_ucb(self, move, simulation_num, multiplier=1):
        # calculate upper confidence bound for move
        # https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
        if move['visit_count'] == 0:
            return (self.init_explore * simulation_num - 1 / (move['score'] + 0.0005)) * multiplier
            # return move['score'] + self.init_explore * math.sqrt(math.log(simulation_num + 1) / 1) * multiplier
        return move['mcts_score'] + self.explore * math.sqrt(math.log(simulation_num) / move['visit_count']) * multiplier

    def simulate_game(self, board, depth):
        result = board.result(claim_draw=True)
        if result != '*':
            # end of game
            if result == '1/2-1/2':
                return 0.5
            else:
                return 1.01 + depth / 100

        moves, value = self.run_network(board)
        if depth == 0:
            return 1 - value

        total_simulations = 0
        for move in moves:
            total_simulations += move['visit_count']
        next_move = self.choose_next_move(moves, total_simulations)
        board.push_uci(next_move['move'])
        sample_score = self.simulate_game(board, depth-1)
        board.pop()
        next_move['total_score'] += sample_score
        next_move['visit_count'] += 1
        next_move['mcts_score'] = next_move['total_score'] / next_move['visit_count']
        total_simulations += 1
        # next_move['mcts_score'] = sample_score
        self.cache[board.fen()] = moves, value
        
        best_score = None
        best_move_weight = 1
        for move in moves:
            if best_score == None or move['mcts_score'] > best_score:
                best_score = move['mcts_score']
                best_move_weight = move['visit_count'] / (move['visit_count'] + next_move['visit_count'])
            # if move['mcts_score'] > sample_score:
            #     better_scores += 1
        # best_move_weight = better_scores / (better_scores + 1)
        avg_score = best_score * best_move_weight + sample_score * (1-best_move_weight)
        # avg_score = avg_score * total_simulations / (total_simulations + 1) + value / (total_simulations + 1)
        return 1 - avg_score

    def choose_next_move(self, moves, total_simulations):
        next_move = None
        max_ucb = None
        for move in moves:
            ucb = self.calc_ucb(move, total_simulations)
            if max_ucb == None or ucb > max_ucb:
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

    def run_network(self, board):
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

        self.cache[fen] = self.format_model_output(policies[0], values[0], board)
        return self.cache[fen]

    def format_model_output(self, policy, value, board):
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

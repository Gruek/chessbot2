import chess
import numpy as np
from model_resnet import get_model, save_model
from tensorflow.python.client import device_lib
from move_encoder import MoveEncoder

class ChessBot():
    def __init__(self):
        self.move_encoder = MoveEncoder()
        # gpus = len(device_lib.list_local_devices()) - 1
        self.model = get_model(len(self.move_encoder.moves))

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
                    if p1_color == chess.BLACK:
                        rank = 7-rank
                    board_matrix[file][rank][piece_idx] = 1
        castling_matrix = np.zeros(shape=(4,), dtype=np.int8)
        white_castling = [int(bool(board.castling_rights & chess.BB_A1)), int(bool(board.castling_rights & chess.BB_H1))]
        black_castling = [int(bool(board.castling_rights & chess.BB_A8)), int(bool(board.castling_rights & chess.BB_H8))]
        castling_matrix[0:2] = white_castling if p1_color == chess.WHITE else black_castling
        castling_matrix[2:4] = black_castling if p1_color == chess.WHITE else white_castling
        return board_matrix, castling_matrix

    def best_move(self, fen, depth=7, chance_limit=0.01):
        board = chess.Board(fen=fen)
        moves, value = self.run_network(board)
        best_move = None
        pov = board.turn
        for move in moves:
            board.push_uci(move['move'])
            move['search_score'] = self.eval_move(board, 1, pov, depth, chance_limit)
            board.pop()
            if not best_move or move['search_score'] > best_move['search_score']:
                best_move = move
        return best_move

    def validate_moves(self, board, moves_array):
        legal_moves = np.zeros(shape=(len(self.move_encoder.moves),), dtype=np.int8)
        for move in board.legal_moves:
            uci = move.uci()
            move_index = self.move_encoder.uci_to_index(uci, board.turn)
            legal_moves[move_index] = 1
        valid_moves = np.multiply(legal_moves, moves_array)
        valid_moves /= np.sum(valid_moves)
        return valid_moves

    def eval_move(self, board, chance=1, pov=None, depth=7 ,chance_limit=0.1):
        if pov == None:
            pov = board.turn
        result = board.result()
        if result != '*':
            if result == '1/2-1/2':
                return 0.5 * chance
            if result == '1-0':
                if pov == chess.WHITE:
                    return 1 * chance
                else:
                    return 0
            if result == '0-1':
                if pov == chess.WHITE:
                    return 0
                else:
                    return 1 * chance
        policy, value = self.run_network(board)
        if chance < chance_limit or depth == 0:
            if board.turn != pov:
                value = 1-value
            return value * chance
        score = 0
        for move in policy:
            board.push_uci(move['move'])
            score += self.eval_move(board, move['score'] * chance, pov, depth-1, chance_limit)
            board.pop()
        return score

    def run_network(self, board):
        board_matrix, castling_matrix = self.board_to_input(board)
        board_inputs = np.zeros(shape=(1, 8, 8, 12), dtype=np.int8)
        castling_inputs = np.zeros(shape=(1, 4), dtype=np.int8)
        board_inputs[0] = board_matrix
        castling_inputs[0] = castling_matrix
        moves_array, value = self.model.predict([board_inputs, castling_inputs])
        moves_array = self.validate_moves(board, moves_array[0])
        move_scores = []
        for i, move_score in enumerate(moves_array):
            if move_score > 0:
                move_score = {'move': self.move_encoder.index_to_uci(i, board.turn), 'score': move_score}
                move_scores.append(move_score)
        return move_scores, value[0][0]
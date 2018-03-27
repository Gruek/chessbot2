import chess
from bot import ChessBot
import sqlite3
import numpy as np
from pystockfish import Engine
from IPython.display import clear_output
from IPython.core.display import display

class Trainer():
    def __init__(self):
        self.chbot = ChessBot()

    def play_vs_stockfish(self, fish, think_time, debug):
        self.chbot.clear_cache()
        board = chess.Board()
        fish.newgame()
        stockfish_color = np.random.randint(2)

        while not board.is_game_over():
            if board.turn == stockfish_color:
                fish.setfenposition(board.fen())
                board.push_uci(fish.bestmove()['move'])
            else:
                move = self.chbot.best_move(board, time_limit=think_time, depth=7, debug=debug)
                board.push_uci(move['move'])
            if debug:
                display(board)
        clear_output()
        result = board.result()
        win = 0.5
        if result == '1-0':
            win = 0 if stockfish_color == chess.WHITE else 1
        elif result == '0-1':
            win = 1 if stockfish_color == chess.WHITE else 0
        print(result, len(board.move_stack), win)
        return board, win

    def train_vs_stockfish(self, debug=False, think_time=30):
        fish = Engine(depth=20, param={"Threads": 12, "Hash": 1024})
        wins = 0
        draws = 0
        games = 0
        while True:
            board, win = self.play_vs_stockfish(fish, think_time=think_time, debug=debug)
            if win == 1:
                wins += 1
            elif win == 0.5:
                draws += 1
            games += 1
            self.train_from_board(board)
            print('Wins:', wins, 'Draws:', draws, 'Games:', games)

    def train_from_board(self, board):
        result = board.result()
        winner = 2
        if result == '1-0':
            winner = chess.WHITE
        elif result == '0-1':
            winner = chess.BLACK
        if winner == 2:
            return
            
        inputs_board_state = np.zeros(shape=(len(board.move_stack), 8, 8, 12), dtype=np.int8)
        inputs_castling = np.zeros(shape=(len(board.move_stack), 4), dtype=np.int8)
        outputs_move = np.zeros(shape=(len(board.move_stack), len(self.chbot.move_encoder.moves)), dtype=np.int8)
        outputs_value = np.zeros(shape=(len(board.move_stack), 2), dtype=np.int8)
        i = 0
        while len(board.move_stack) > 0:
            move = board.pop().uci()
            move_index = self.chbot.move_encoder.uci_to_index(move, board.turn)
            board_matrix, castling_matrix = self.chbot.board_to_input(board)
            inputs_board_state[i] = board_matrix
            inputs_castling[i] = castling_matrix
            outputs_move[i][move_index] = 1
            outputs_value[i] = [1, 0] if winner == board.turn else [0, 1]
            i += 1
        # self.chbot.model.train_on_batch([inputs_board_state, inputs_castling], [outputs_move, outputs_value])
        self.chbot.model.fit([inputs_board_state, inputs_castling], [outputs_move, outputs_value], verbose=1, batch_size=32)
        self.chbot.save_model()
        
    
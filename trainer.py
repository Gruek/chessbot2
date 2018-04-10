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

    def play_vs_stockfish(self, fish, think_time, depth, debug):
        board = chess.Board()
        fish.newgame()
        stockfish_color = np.random.randint(2)
        last_score = None

        while not board.is_game_over(claim_draw=True):
            if board.turn == stockfish_color:
                fish.setfenposition(board.fen())
                board.push_uci(fish.bestmove()['move'])
            else:
                ttt = think_time
                if last_score:
                    if last_score > 0.95 or last_score < 0.05:
                        ttt = think_time / 3
                move = self.chbot.best_move(board, time_limit=ttt, depth=depth, debug=debug)
                last_score = move['score']
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

    def train_vs_stockfish(self, debug=False, think_time=20, depth=6):
        fish = Engine(depth=20, param={"Threads": 12, "Hash": 64})
        wins = 0
        draws = 0
        games = 0
        while True:
            board, win = self.play_vs_stockfish(fish, think_time=think_time, depth=depth, debug=debug)
            if win == 1:
                wins += 1
                if think_time > 1:
                    think_time -= 1
            elif win == 0.5:
                draws += 1
            else:
                if think_time < 20:
                    think_time += 1
                pass
            games += 1
            self.chbot.train_from_board(board)
            print('Wins:', wins, 'Draws:', draws, 'Games:', games, 'Think Time:', think_time)
        
    def play_vs_self(self, think_time, debug, depth):
        board = chess.Board()

        while not board.is_game_over():
            move = self.chbot.best_move(board, time_limit=think_time, depth=depth, debug=debug)
            board.push_uci(move['move'])
            if debug:
                display(board)
        clear_output()
        result = board.result()
        win = 0.5
        if result == '1-0':
            win = 1
        elif result == '0-1':
            win = 0
        print(result, len(board.move_stack), win)
        return board, win
    
    def train_vs_self(self, debug=False, think_time=10, depth=8):
        white_wins = 0
        black_wins = 0
        draws = 0
        while True:
            board, win = self.play_vs_self(think_time=think_time, debug=debug, depth=depth)
            if win == 1:
                white_wins += 1
            elif win == 0.5:
                draws += 1
            else:
                black_wins += 1
            self.chbot.train_from_board(board)
            print('White:', white_wins, 'Draws:', draws, 'Black:', black_wins)
import chess
from bot import ChessBot
import sqlite3
import numpy as np
from pystockfish import Engine
from IPython.display import clear_output
from IPython.core.display import display
import sys

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
                    if last_score < 0.01:
                        ttt = think_time / 3
                move = self.chbot.best_move(board, time_limit=ttt, depth=depth, debug=debug)
                last_score = move['score']
                board.push_uci(move['move'])
            if debug:
                display(board)
        clear_output()
        result = board.result(claim_draw=True)
        win = 0.5
        if result == '1-0':
            win = 0 if stockfish_color == chess.WHITE else 1
        elif result == '0-1':
            win = 1 if stockfish_color == chess.WHITE else 0
        print(result, len(board.move_stack), win)
        return board, win

    def train_vs_stockfish(self, debug=False, think_time=20, depth=6, stockfish_depth=6):
        wins = 0
        draws = 0
        games = 0
        while True:
            fish = Engine(depth=stockfish_depth, param={"Threads": 1, "Hash": 64})
            board, win = self.play_vs_stockfish(fish, think_time=think_time, depth=depth, debug=debug)
            if win == 1:
                wins += 1
                if think_time > 1:
                    think_time -= 1
                stockfish_depth += 1
            elif win == 0.5:
                draws += 1
                if think_time < 20:
                    think_time += 1
            else:
                if think_time < 20:
                    think_time += 1
                if stockfish_depth > 3:
                    stockfish_depth -= 1
            games += 1
            self.chbot.train_from_board(board)
            print('Wins:', wins, 'Draws:', draws, 'Games:', games, 'Think Time:', think_time, 'Stockfish depth:', stockfish_depth)
        
    def play_vs_self(self, p1, p2, think_time, debug, depth):
        board = chess.Board()
        last_score = None
        p1_color = np.random.randint(2)

        while not board.is_game_over(claim_draw=True):
            ttt = think_time
            if last_score:
                if last_score > 0.99:
                    ttt = think_time / 3
            move = None
            if board.turn == p1_color:
                move = p1.best_move(board, time_limit=ttt, depth=depth, debug=debug)
            else:
                move = p2.best_move(board, time_limit=ttt, depth=depth, debug=debug)
            last_score = move['score']
            board.push_uci(move['move'])
            if debug:
                display(board)
        if debug:
            clear_output()
        result = board.result(claim_draw=True)
        win = 0.5
        winner = 0.5
        if result == '1-0':
            win = 1
            winner = 1 if p1_color == chess.WHITE else 0
        elif result == '0-1':
            win = 0
            winner = 0 if p1_color == chess.WHITE else 1
        print(result, len(board.move_stack), win, winner)
        return board, win, winner
    
    def train_vs_self(self, debug=False, think_time=10, depth=10):
        white_wins = 0
        black_wins = 0
        draws = 0
        p1_wins = 0
        p2_wins = 0

        p2 = ChessBot(model=self.chbot.model)
        p2.explore = 0.3
        p2.init_explore = 1.05

        while True:
            board, win, winner = self.play_vs_self(p1=self.chbot, p2=p2, think_time=think_time, debug=debug, depth=depth)
            if win == 1:
                white_wins += 1
            elif win == 0.5:
                draws += 1
            else:
                black_wins += 1
            if winner == 1:
                p1_wins += 1
            elif winner == 0:
                p2_wins += 1
            self.chbot.train_from_board(board)
            print('White:', white_wins, 'Draws:', draws, 'Black:', black_wins, 'P1 wins:', p1_wins, 'P2 wins:', p2_wins)
            sys.stdout.flush()

        
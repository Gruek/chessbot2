from bot import ChessBot
import numpy as np
import chess

chbot = ChessBot()
board = chess.Board()

def make_move():
    move = chbot.best_move(board.fen())
    turn = 'WHITE' if board.turn == chess.WHITE else 'BLACK'
    print(turn, move)
    board.push_uci(move['move'])
    print(board, flush=True)

while board.result() == '*':
    make_move()

print(board.result())

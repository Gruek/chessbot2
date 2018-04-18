from bot import ChessBot
import chess
import time
import numpy as np
import cProfile
import threading
import sys

bot = ChessBot()
board = chess.Board()
# board = chess.Board(fen='r1bqk2r/1p1p1ppp/2n1pb2/p1P5/QPP1n3/P3PN2/R4PPP/1NB1KB1R b Kkq - 2 9')
bot.game.set_position(board)
print(board)

def foo():
    bot.best_move(board, time_limit=10, debug=True)

# cProfile.run('foo()')
foo()
bot.print_stats()
bot.mp_pool.close()
bot.mp_pool.join()

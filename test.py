from bot import ChessBot
import chess
import chess.pgn
import time
import numpy as np
import cProfile
import threading
import sys

if __name__ == '__main__':
    # bot = ChessBot()
    # board = chess.Board()
    # board = chess.Board(fen='r1bqk2r/1p1p1ppp/2n1pb2/p1P5/QPP1n3/P3PN2/R4PPP/1NB1KB1R b Kkq - 2 9')
    # bot.game.set_position(board)
    # print(board)

    # def foo():
    #     bot.best_move(board, time_limit=10, debug=True)

    # # cProfile.run('foo()')
    # foo()
    # bot.print_stats()
    # bot.mp_pool.close()
    # bot.mp_pool.join()

    board = chess.Board()
    board.push_uci('e2e4')

    g = chess.pgn.Game.from_board(board)
    with open("test.pgn", "a") as f:
        print(g, file=f, end='\n\n')

    board = chess.Board()
    board.push_uci('d2d4')

    g = chess.pgn.Game.from_board(board)
    with open("test.pgn", "a") as f:
        print(g, file=f, end='\n\n')

    print(board)

    # with open('test.pgn', 'r') as f:
    #     while True:
    #         game = chess.pgn.read_game(f)
    #         if not game:
    #             break
    #         board = game.end().board()
    #         print(board)


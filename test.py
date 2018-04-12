from bot import ChessBot
import chess
import time
import numpy as np
import cProfile

bot = ChessBot()
board = chess.Board(fen='r1bqk2r/1p1p1ppp/2n1pb2/p1P5/QPP1n3/P3PN2/R4PPP/1NB1KB1R b Kkq - 2 9')
bot.game.set_position(board)
print(board)

def foo():
    bot.best_move(board, time_limit=10, debug=True)

cProfile.run('foo()')
# foo()
print(bot.meta_data)
print(bot.game.meta_data)


# board_inputs = np.zeros(shape=(1, 8, 8, 12), dtype=np.int8)
# castling_inputs = np.zeros(shape=(1, 4), dtype=np.int8)


# # input
# t1 = time.time()
# board_matrix, castling_matrix = bot.game.input_matrix(board)
# board_inputs[0] = board_matrix
# castling_inputs[0] = castling_matrix

# # run model
# policies, values = bot.model.predict([board_inputs, castling_inputs])
# print(time.time() - t1)


# t1 = time.time()
# legal_moves = list(board.legal_moves)
# moves_num = len(legal_moves)
# print(moves_num)

# board_inputs = np.zeros(shape=(moves_num, 8, 8, 12), dtype=np.int8)
# castling_inputs = np.zeros(shape=(moves_num, 4), dtype=np.int8)


# # input
# for i, m in enumerate(legal_moves):
#     board.push(m)
#     board_matrix, castling_matrix = bot.game.input_matrix(board)
#     board_inputs[i] = board_matrix
#     castling_inputs[i] = castling_matrix
#     board.pop()

# # run model
# policies, values = bot.model.predict([board_inputs, castling_inputs], batch_size=64)
# print(time.time() - t1)

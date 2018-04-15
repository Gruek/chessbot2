from bot import ChessBot
import chess
import time
import numpy as np
import cProfile
import threading

bot = ChessBot()
board = chess.Board(fen='r1bqk2r/1p1p1ppp/2n1pb2/p1P5/QPP1n3/P3PN2/R4PPP/1NB1KB1R b Kkq - 2 9')
bot.game.set_position(board)
print(board)

def foo():
    bot.best_move(board, time_limit=10, debug=True)

# cProfile.run('foo()')
# foo()
print(bot.meta_data)
print(bot.game.meta_data)



legal_moves = list(board.legal_moves)
moves_num = len(legal_moves)
print(moves_num)

board_inputs = np.zeros(shape=(moves_num, 8, 8, 12), dtype=np.int8)
castling_inputs = np.zeros(shape=(moves_num, 4), dtype=np.int8)


# input
for i in range(moves_num):
    m = legal_moves[i]
    board.push(m)
    board_matrix, castling_matrix = bot.game.input_matrix(board)
    board_inputs[i] = board_matrix
    castling_inputs[i] = castling_matrix
    board.pop()

# run model
def foo2():
    t1 = time.time()
    policies, values = bot.model.predict([board_inputs, castling_inputs], batch_size=64)
    print('bs:', moves_num, time.time() - t1)

for i in range(1):
    foo2()
# cProfile.run('foo2()')
class myThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def inference(self, bs=1):
        board_inputs = np.zeros(shape=(bs, 8, 8, 12), dtype=np.int8)
        castling_inputs = np.zeros(shape=(bs, 4), dtype=np.int8)

        # run model
        t1 = time.time()
        policies, values = bot.model.predict([board_inputs, castling_inputs], batch_size=64)
        print('Thread', self.threadID, 'bs:', bs, time.time() - t1)

    def run(self):
        for i in range(3):
            self.inference()

thread1 = myThread(1)
thread2 = myThread(2)

thread1.start()
# thread2.start()

print('exit main thread')
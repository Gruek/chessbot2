from bot import ChessBot
import numpy as np
import chess
import chess.pgn as pgn
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import sqlite3
from db_trainer import DBTrainer

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

trainer = DBTrainer()
chbot = ChessBot()
f = '/data/kru03a/chbot/data/ficsgamesdb_2016_standard2000_nomovetimes_1514497.pgn'
# game = pgn.read_game(open(f))
# board = game.end().board()
# board.pop()
board = chess.Board()
# print(board)

def make_move():
    move = chbot.best_move(board.fen())
    print(move)
    board.push_uci(move['move'])
    print(board)

# for i in range(1):
#     make_move()

print(board.result())


db = sqlite3.connect(trainer.db_path)
cursor = db.cursor()
cursor.execute('select fen, move, winner from moves_train order by random()')
batch = cursor.fetchmany(5)
for game in batch:
    fen = game[0]
    move = game[1]
    winner = game[2]

    board = chess.Board(fen=fen)
    print(board)
    print(move)
    print(winner, winner == board.turn)

# db.close()
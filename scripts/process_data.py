import chess
import chess.pgn as pgn
import sqlite3
import os

def from_file_to_db(file, db, table):
    f = open(file)
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS ' + table)
    c.execute('CREATE TABLE ' + table + ' (fen text, move text, winner number)')

    games = 0
    moves = 0
    while True:
        game = pgn.read_game(f)
        if not game:
            break
        
        result = game.headers["Result"]
        winner = 2
        if result == '1-0':
            winner = chess.WHITE
        elif result == '0-1':
            winner = chess.BLACK
        if winner == 2:
            continue

        board = game.end().board()
        if len(board.move_stack) < 3:
            continue
        
        games += 1

        while len(board.move_stack) > 0:
            moves += 1
            move = board.pop()
            c.execute('INSERT INTO ' + table + ' (fen, move, winner) VALUES (?, ?, ?)', (board.fen(), move.uci(), winner))

    conn.commit()
    conn.close()
    print(games, moves)

data_path = '../data'
db_path = os.path.join(data_path, 'moves.db')
f1 = os.path.join(data_path, 'ficsgamesdb_2016_standard_nomovetimes_1536291.pgn')
f2 = os.path.join(data_path, 'ficsgamesdb_201701_standard_nomovetimes_1536292.pgn')

from_file_to_db(f1, db_path, 'moves_train')
from_file_to_db(f2, db_path, 'moves_val')

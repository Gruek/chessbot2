import chess
import chess.pgn as pgn
from bot import ChessBot
import time
import os

bot = ChessBot(num_slaves=0)
input_dir = '/data/kru03a/chbot/data/leela_pgn'

def process_pgn(data_file):
    games = []

    with open(data_file, 'r') as f:
        while True:
            g = pgn.read_game(f)
            if g == None:
                break
            games.append(g)
    print(len(games))
    bot.game.set_position(chess.Board())

    for g in games:
        result = g.headers["Result"]
        b = g.end().board()
        print(result, len(b.move_stack), g.headers["Event"], flush=True)
        bot.train_from_board(b, result)


if __name__ == '__main__':
    filelist = list(os.listdir(input_dir))
    print(len(filelist))
    filelist.sort(key=lambda x: int(x.split('games')[1].split('.pgn')[0]))
    start_at = 'games621'
    start_index = 0
    for i, f in enumerate(filelist):
        if start_at in f:
            start_index = i
            break
    filelist = filelist[start_index:]
    print(len(filelist), start_index)

    for filename in filelist:
        filename = os.path.join(input_dir, filename)
        print(filename)
        process_pgn(filename)
    
    bot.game.set_position(chess.Board())
    print('done')

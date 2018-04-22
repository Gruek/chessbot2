import chess
import chess.pgn as pgn
from bot import ChessBot
import time

bot = ChessBot()

data_file = 'data_in.pgn'
games = []

with open(data_file, 'r') as f:
    while True:
        g = pgn.read_game(f)
        if g == None:
            break
        games.append(g.end().board())
print(len(games))

for b in games:
    print(b.result(claim_draw=True), len(b.move_stack))
    bot.train_from_board(b)

# time.sleep(20)
bot.game.set_position(chess.Board())
print('done')

import sys
import chess
import time
from threading import Thread
from bot import ChessBot

class UCIAdapter():
    def __init__(self, chbot):
        self.chbot = chbot
        self.debug = False
        self.board = chess.Board()
        self.name = 'gruekbot'
        self.author = 'AK'
        self.IO = {
            'uci': self.uci,
            'debug': self.set_debug,
            'isready': self.is_ready,
            'setoption': self.setoption,
            'register': self.register,
            'ucinewgame': self.ucinewgame,
            'position': self.position,
            'go': self.go,
            'stop': self.stop,
            'quit': self.quit
        }
        self.log_file = 'uci_coms.log'
        self.search_thread = None
        self.quit_received = False

    def log(self, msg):
        with open(self.log_file, 'a+') as f:
            f.write(msg + '\n')

    def recv(self, inp):
        out = []
        self.log('inp: ' + str(inp))
        inp = inp.split(" ")
        if inp[0] in self.IO:
            out_func = self.IO[inp[0]]
            out = out_func(inp)
        else:
            out = ['unknown command: ' + inp[0]]

        self.log('out: ' + str(out))
        return out

    def uci(self, inp):
        return [
            'id name ' + self.name,
            'id author ' + self.author,
            'uciok'
        ]

    def is_ready(self, inp):
        self.chbot.game.set_position(self.board)
        return ['readyok']

    def set_debug(self, inp):
        if inp[1] == 'on':
            self.debug = True
        else:
            self.debug = False
        return []

    def setoption(self, inp):
        return []

    def register(self, inp):
        return []

    def ucinewgame(self, inp):
        return []

    def position(self, inp):
        if len(inp) < 2:
            self.board = chess.Board()
        else:
            if inp[1] == 'startpos':
                self.board = chess.Board()
            else:
                fen = inp[inp.index('fen') + 1:]
                if 'moves' in inp:
                    fen = fen[:fen.index('moves')]
                fen = " ".join(fen)
                self.board = chess.Board(fen=fen)
            
            if 'moves' in inp:
                inp = inp[inp.index('moves') + 1:]

                for i in range(len(inp)):
                    self.board.push_uci(inp[i])
        
        self.chbot.game.set_position(self.board)
        return []

    def go(self, inp):
        kwargs = {'board': self.board}
        if 'movetime' in inp:
            kwargs['time_limit'] = int(inp[inp.index('movetime') + 1])
        if 'depth' in inp:
            kwargs['depth'] = int(inp[inp.index('depth') + 1])
        if 'infinite' in inp:
            kwargs['time_limit'] = 3600
        self.log('search_options: ' + str(kwargs))

        self.search_thread = Thread(target=self.async_search, args=(kwargs,))
        self.search_thread.start()
        return []

    def async_search(self, search_options):
        move = self.chbot.best_move(**search_options)
        out = 'bestmove ' + move['move'] + '\n'
        self.log('out: ' + out)
        print(out, flush=True)

    def stop(self, inp):
        self.chbot.stop = True
        return []

    def quit(self, inp):
        self.quit_received = True
        return ['bye']

    def start(self):
        while True:
            if self.quit_received == True:
                break
            try:
                inp = input("")
            except EOFError:
                break
            out = self.recv(inp)
            for o in out:
                print(o)

class FakeBot():
    def __init__(self):
        self.stop = False

    def best_move(self, board, time_limit=5):
        self.stop = False
        cutoff_time = time.time() + time_limit
        moves = list(board.legal_moves)

        while time.time() < cutoff_time and self.stop == False:
            time.sleep(1)
        return {'move': moves[0].uci()}

if __name__ == '__main__':
    uci = UCIAdapter(ChessBot())
    uci.start()

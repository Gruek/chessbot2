import sys
import chess

class UCIInterface():
    def __init__(self):
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
            'position': self.position
        }
        self.log_file = 'uci_coms.log'

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
            fen = inp[1]
            if fen == 'startpos':
                self.board = chess.Board()
            else:
                fen = " ".join(inp[inp.index('fen') + 1 : inp.index('moves')])
                self.board = chess.Board(fen=fen)
            inp = inp[inp.index('moves') + 1:]

            i = 0
            while len(inp) - 1 >= i:
                self.board.push_uci(inp[i])
                i += 1
        return []

    def start(self):
        while True:
            try:
                inp = input("")
            except EOFError:
                break
            out = self.recv(inp)
            for o in out:
                print(o)

if __name__ == '__main__':
    uci_interface = UCIInterface()
    uci_interface.start()

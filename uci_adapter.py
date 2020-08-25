import sys
import chess
import time
from threading import Thread
from bot import ChessBot
import multiprocessing

class UCIAdapter():
    def __init__(self):
        self.log_file_path = 'uci_coms.log'
        self.stdout = sys.stdout
        self.log_file = open(self.log_file_path, 'a')
        sys.stderr = self.log_file
        self.chbot = ChessBot(stdout=self.log_file)
        self.debug = False
        self.board = chess.Board()
        self.name = 'gruekbot'
        self.options = {
            'depth': 6,
            'ponder': False
        }
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
        self.search_thread = None
        self.join_called = False
        self.quit_received = False

        # initialize chbot
        self.chbot.game.set_position(self.board)

    def recv(self, inp):
        out = []
        print('inp: ' + str(inp), file=self.log_file, flush=True)
        inp = inp.split(" ")
        if inp[0] in self.IO:
            out_func = self.IO[inp[0]]
            out = out_func(inp)
        else:
            out = ['unknown command: ' + inp[0]]

        print('out: ' + str(out), file=self.log_file, flush=True)
        return out

    def uci(self, inp):
        return [
            'id name ' + self.name,
            'id author ' + self.author,
            'uciok'
        ]

    def sync_search_thread(self):
        # sync threads
        if self.search_thread and self.search_thread.is_alive():
            self.chbot.stop = True
            self.join_called = True
            self.search_thread.join()
            self.join_called = False

    def is_ready(self, inp):
        self.sync_search_thread()
        return ['readyok']

    def set_debug(self, inp):
        if inp[1] == 'on':
            self.options['debug'] = True
        else:
            self.options['debug'] = False
        return []

    def setoption(self, inp):
        try:
            opt_name = inp[2]
            opt_val = inp[4]
            if opt_val.isdigit():
                opt_val = int(opt_val)
            self.options[opt_name] = opt_val
        except IndexError:
            print('setoption index error:', inp, file=self.log_file, flush=True)
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
        kwargs = self.options.copy()
        kwargs['board'] = self.board
        
        if 'movetime' in inp:
            kwargs['time_limit'] = int(inp[inp.index('movetime') + 1])
        if 'depth' in inp:
            kwargs['depth'] = int(inp[inp.index('depth') + 1])
        if 'infinite' in inp:
            kwargs['time_limit'] = 3600
        print('search_options: ' + str(kwargs), file=self.log_file, flush=True)

        self.sync_search_thread()
        self.search_thread = Thread(target=self.async_search, args=(kwargs,))
        self.search_thread.start()
        return []

    def async_search(self, search_options):
        move = self.chbot.best_move(**search_options)
        out = 'bestmove ' + move['move']
        print('out: ' + out, flush=True, file=self.log_file)
        print(out, flush=True)

        if self.options['ponder'] and self.join_called == False:
            # ponder about next move
            search_options['depth'] = search_options['depth'] + 1
            search_options['time_limit'] = 180
            self.board.push_uci(move['move'])
            move = self.chbot.best_move(**search_options)
            print('ponder bestmove ' + move['move'], flush=True, file=self.log_file)

    def stop(self, inp):
        self.chbot.stop = True
        return []

    def quit(self, inp):
        self.quit_received = True
        self.sync_search_thread()
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
    multiprocessing.freeze_support()
    uci = UCIAdapter()
    uci.start()

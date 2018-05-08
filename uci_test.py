import chess.uci
import chess

engine = chess.uci.popen_engine("python ./uci_interface.py")
engine.uci()
engine.debug(True)
print(engine.name)
print(engine.author)
engine.isready()
b = chess.Board(fen='r1bqk2r/1p1p1ppp/2n1pb2/p1P5/QPP1n3/P3PN2/R4PPP/1NB1KB1R b Kkq - 2 9')
moves = list(b.legal_moves)
b.push(moves[0])
# b = chess.Board()
# b.push_uci('e2e4')
engine.position(b)
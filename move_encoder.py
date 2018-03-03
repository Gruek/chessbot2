import chess

class MoveEncoder():
    def __init__(self):
        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.ranks = range(1,9)
        self.moves = []
        self.moves_black = []

        for from_file in self.files:
            for from_rank in self.ranks:
                from_square = from_file + str(from_rank)
                for to_file in self.files:
                    for to_rank in self.ranks:
                        to_square = to_file + str(to_rank)
                        self.moves.append(from_square + to_square)

        self.promotion_options = ['q', 'b', 'r', 'n']
        self.promotion_moves = [        'a7a8', 'a7b8',
                                'b7a8', 'b7b8', 'b7c8',
                                'c7b8', 'c7c8', 'c7d8',
                                'd7c8', 'd7d8', 'd7e8',
                                'e7d8', 'e7e8', 'e7f8',
                                'f7e8', 'f7f8', 'f7g8',
                                'g7f8', 'g7g8', 'g7h8',
                                'h7g8', 'h7h8']

        for promotion in self.promotion_options:
            for moves in self.promotion_moves:
                self.moves.append(moves + promotion)

        for move in self.moves:
            self.moves_black.append(self.translate_move(move))

        self.move_to_index = {}
        for i, move in enumerate(self.moves):
            self.move_to_index[move] = i
        self.move_to_index_black = {}
        for i, move in enumerate(self.moves_black):
            self.move_to_index_black[move] = i

    def index_to_uci(self, index, color):
        if color == chess.WHITE:
            return self.moves[index]
        return self.moves_black[index]

    def uci_to_index(self, uci, color):
        if color == chess.WHITE:
            return self.move_to_index[uci]
        return self.move_to_index_black[uci]

    def translate_move(self, move):
        #rotate the board for black
        from_rank = 9 - int(move[1])
        to_rank = 9 - int(move[3])
        return move[0] + str(from_rank) + move[2] + str(to_rank) + move[4:]


import chess
import numpy as np
from move_encoder import MoveEncoder

class Game():
    def __init__(self, model, board=None):
        self.model = model
        self.move_encoder = MoveEncoder()
        self.board = None
        self.root_node_id = None
        self.node_stack = []
        if board != None:
            self.set_position(board)

        self.meta_data = {'inferences': 0}

    def set_position(self, board):
        self.meta_data['inferences'] = 0
        self.board = board.copy()
        new_node_id = self.node_id(self.board)

        if self.root() != None and new_node_id[:len(self.root_node_id)] == self.root_node_id:
            new_node = self.root().traverse(new_node_id[len(self.root_node_id):])
            if new_node:
                self.node_stack = [new_node]
                self.root_node_id = new_node_id
                return

        self.root_node_id = new_node_id
        self.node_stack = [self.expand()]

    def push(self, move):
        self.board.push_uci(move)
        next_node = self.node_stack[-1].traverse([move])
        if next_node == None:
            next_node = self.expand()
            self.node_stack[-1].child_links[move].node = next_node
        self.node_stack.append(next_node)

    def pop(self):
        self.board.pop()
        self.node_stack = self.node_stack[:-1]

    def root(self):
        if len(self.node_stack) > 0:
            return self.node_stack[0]
        return None

    def node(self):
        return self.node_stack[-1]

    def score(self):
        return self.node().score

    def node_id(self, board):
        return [move.uci() for move in board.move_stack]

    def input_matrix(self):
        p1_color = self.board.turn
        board_matrix = np.zeros(shape=(8, 8, 12), dtype=np.int8)
        for rank in range(8):
            for file in range(8):
                piece = self.board.piece_at(rank*8+file)
                if piece:
                    piece_p2 = piece.color != p1_color
                    piece_idx = piece_p2 * 6 + piece.piece_type - 1
                    #rotate the board for black
                    r = rank if p1_color == chess.WHITE else 7 - rank
                    board_matrix[file][r][piece_idx] = 1
        castling_matrix = np.zeros(shape=(4,), dtype=np.int8)
        white_castling = [int(bool(self.board.castling_rights & chess.BB_A1)), int(bool(self.board.castling_rights & chess.BB_H1))]
        black_castling = [int(bool(self.board.castling_rights & chess.BB_A8)), int(bool(self.board.castling_rights & chess.BB_H8))]
        castling_matrix[0:2] = white_castling if p1_color == chess.WHITE else black_castling
        castling_matrix[2:4] = black_castling if p1_color == chess.WHITE else white_castling
        return board_matrix, castling_matrix

    def validate_moves(self, moves_array):
        legal_moves = np.zeros(shape=(len(self.move_encoder.moves),), dtype=np.int8)
        for move in self.board.legal_moves:
            uci = move.uci()
            move_index = self.move_encoder.uci_to_index(uci, self.board.turn)
            legal_moves[move_index] = 1
        valid_moves = np.multiply(legal_moves, moves_array)
        valid_moves /= np.sum(valid_moves)
        return valid_moves

    def expand(self):
        result = self.board.result(claim_draw=True)
        if result != '*':
            # end of game
            score = -0.001
            if result == '1/2-1/2':
                score = 0.5
            node = Node(score)
            return node
                
        self.meta_data['inferences'] += 1

        board_inputs = np.zeros(shape=(1, 8, 8, 12), dtype=np.int8)
        castling_inputs = np.zeros(shape=(1, 4), dtype=np.int8)

        # input
        board_matrix, castling_matrix = self.input_matrix()
        board_inputs[0] = board_matrix
        castling_inputs[0] = castling_matrix

        # run model
        policies, values = self.model.predict([board_inputs, castling_inputs])
        policy, value = policies[0], values[0][0]
        policy = self.validate_moves(policy)
        node_children = []
        for move_index, weight in enumerate(policy):
            if weight > 0:
                node_children.append({
                    'move': self.move_encoder.index_to_uci(move_index, self.board.turn),
                    'weight': weight
                })
        node = Node(value, node_children)
        return node

    def next_moves(self):
        return self.node().children(include_unvisited=True)

class Node():
    def __init__(self, score, children=[]):
        self.visits = 0
        self.score = score
        self.child_links = {}
        for child in children:
            self.child_links[child['move']] = NodeLink(child['weight'])

    def children(self, include_unvisited=False):
        #flat view of next level children
        child_nodes = []
        for move, link in self.child_links.items():
            child = {'move': move, 'weight': link.weight, 'visits': 0, 'score': -1}
            if link.node:
                child['score'] = link.node.score
                child['visits'] = link.node.visits
                child_nodes.append(child)
            elif include_unvisited:
                child_nodes.append(child)
        return child_nodes

    def traverse(self, path):
        if len(path) == 0:
            return self
        link = self.child_links[path[0]]
        if link and link.node:
            return link.node.traverse(path[1:])
        return None

    def __str__(self):
        return str({'score': self.score, 'visits': self.visits})

class NodeLink():
    def __init__(self, weight):
        self.weight = weight
        self.node = None

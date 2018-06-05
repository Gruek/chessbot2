import chess
import numpy as np
from move_encoder import MoveEncoder
import time

class Game():
    def __init__(self, model, board=None):
        self.model = model
        self.move_encoder = MoveEncoder()
        self.board = None
        self.root_node_id = None
        self.node_stack = []
        if board != None:
            self.set_position(board)

        self.meta_data = {'inferences': 0, 'infer_time': 0, 'check_result_time': 0, 'move_validation_time': 0,
            'push_time': 0, 'pop_time': 0, 'expand_time': 0, 'input_gen_time': 0}

    def set_position(self, board, use_cache=True):
        self.meta_data['inferences'] = 0
        self.board = board.copy()
        new_node_id = self.node_id(self.board)

        if self.root() != None and new_node_id[:len(self.root_node_id)] == self.root_node_id and use_cache:
            new_node = self.root().traverse(new_node_id[len(self.root_node_id):])
            if new_node:
                self.node_stack = [new_node]
                self.root_node_id = new_node_id
                return

        self.root_node_id = new_node_id
        self.node_stack = [self.expand()]
        self.meta_data = {'inferences': 0, 'infer_time': 0, 'check_result_time': 0, 'move_validation_time': 0,
            'push_time': 0, 'pop_time': 0, 'expand_time': 0, 'input_gen_time': 0}

    def push(self, move, depth=0):
        t1 = time.time()
        self.board.push_uci(move)
        next_node = self.node().traverse([move])
        if next_node == None:
            self.meta_data['push_time'] += time.time() - t1
            next_node = self.expand(depth)
            self.node().set_child(move, next_node)
            t1 = time.time()
        self.node_stack.append(next_node)
        self.meta_data['push_time'] += time.time() - t1

    def pop(self):
        t1 = time.time()
        self.board.pop()
        self.node_stack.pop()
        self.meta_data['pop_time'] += time.time() - t1

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

    def input_matrix(self, board):
        p1_color = board.turn
        board_matrix = np.zeros(shape=(8, 8, 12), dtype=np.int8)
        for rank in range(8):
            for file in range(8):
                piece = board.piece_at(rank*8+file)
                if piece:
                    piece_p2 = piece.color != p1_color
                    piece_idx = piece_p2 * 6 + piece.piece_type - 1
                    #rotate the board for black
                    r = rank if p1_color == chess.WHITE else 7 - rank
                    board_matrix[file][r][piece_idx] = 1
        castling_matrix = np.zeros(shape=(4,), dtype=np.int8)
        white_castling = [int(bool(board.castling_rights & chess.BB_A1)), int(bool(board.castling_rights & chess.BB_H1))]
        black_castling = [int(bool(board.castling_rights & chess.BB_A8)), int(bool(board.castling_rights & chess.BB_H8))]
        castling_matrix[0:2] = white_castling if p1_color == chess.WHITE else black_castling
        castling_matrix[2:4] = black_castling if p1_color == chess.WHITE else white_castling
        return board_matrix, castling_matrix

    def validate_moves(self, moves_array):
        legal_moves = {}
        for move in self.board.legal_moves:
            uci = move.uci()
            move_index = self.move_encoder.uci_to_index(uci, self.board.turn)
            weight = moves_array[move_index]
            legal_moves[uci] = NodeLink(weight)
        return legal_moves

    def expand(self, depth=0):
        expand_t1 = time.time()
        t1 = time.time()
        result = self.result()
        if result != -1:
            # end of game
            score = -0.1 / len(self.node_stack)
            if result == 0.5:
                score = 0.5 - 0.001 / len(self.node_stack)
            return Node(score)

        self.meta_data['check_result_time'] += time.time() - t1
        self.meta_data['inferences'] += 1

        # input
        t1 = time.time()
        board_matrix, castling_matrix = self.input_matrix(self.board)
        self.meta_data['input_gen_time'] += time.time() - t1

        # run model
        t1 = time.time()
        policy, value = self.model.predict([board_matrix, castling_matrix])
        self.meta_data['infer_time'] += time.time() - t1
        # policy, value = policies[0], values[0][0]
        value = value[0]
        t1 = time.time()
        node_children = self.validate_moves(policy)
        self.meta_data['move_validation_time'] += time.time() - t1
        node = Node(value, node_children)
        self.meta_data['expand_time'] += time.time() - expand_t1
        return node

    def result(self):
        if self.board.is_checkmate():
            return 0
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0.5
        if len(self.board.move_stack) > 16 and self.board.halfmove_clock >= 5:
            # counting repetitions is expensive so do prelim checks
            if self.board.can_claim_threefold_repetition():
                return 0.5
        return -1

class Node():
    def __init__(self, score, children=None):
        self.visits = 0
        self.score = score
        self.child_links = {}
        if children != None:
            self.child_links = children

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

    def set_child(self, move, node):
        self.child_links[move].node = node

    def size(self):
        count = 0
        for move, link in self.child_links.items():
            if link.node != None:
                count += link.node.size()
        if count == 0:
            return 1
        return count

    def __str__(self):
        return str({'score': self.score, 'visits': self.visits})

class NodeLink():
    def __init__(self, weight, node=None):
        self.weight = weight
        self.node = node

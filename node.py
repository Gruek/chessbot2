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
        self.node_stack = [self.expand(5)]

    def push(self, move, depth=0):
        t1 = time.time()
        self.board.push_uci(move)
        next_node = self.node_stack[-1].traverse([move])
        if next_node == None:
            self.meta_data['push_time'] += time.time() - t1
            next_node = self.expand(depth)
            t1 = time.time()
            self.node_stack[-1].child_links[move].node = next_node
        self.node_stack.append(next_node)
        self.meta_data['push_time'] += time.time() - t1

    def pop(self):
        t1 = time.time()
        self.board.pop()
        self.node_stack = self.node_stack[:-1]
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

    def expand(self, depth):
        if depth > 5:
            return self.expand2()
        # return self.expand2()
        expand_t1 = time.time()
        t1 = time.time()
        result = self.result()
        if result != -1:
            # end of game
            score = -0.001
            if result == 0.5:
                score = 0.5
            return Node(score)

        self.meta_data['check_result_time'] += time.time() - t1
        self.meta_data['inferences'] += 1

        board_inputs = np.zeros(shape=(1, 8, 8, 12), dtype=np.int8)
        castling_inputs = np.zeros(shape=(1, 4), dtype=np.int8)

        # input
        t1 = time.time()
        board_matrix, castling_matrix = self.input_matrix(self.board)
        self.meta_data['input_gen_time'] += time.time() - t1
        board_inputs[0] = board_matrix
        castling_inputs[0] = castling_matrix

        # run model
        t1 = time.time()
        policies, values = self.model.predict([board_inputs, castling_inputs])
        self.meta_data['infer_time'] += time.time() - t1
        policy, value = policies[0], values[0][0]
        t1 = time.time()
        node_children = self.validate_moves(policy)
        self.meta_data['move_validation_time'] += time.time() - t1
        node = Node(value, node_children)
        self.meta_data['expand_time'] += time.time() - expand_t1
        return node

    def expand2(self):
        expand_t1 = time.time()
        result = self.result()
        if result != -1:
            # end of game
            score = -0.001
            if result == 0.5:
                score = 0.5
            return Node(score)

        # forward cache all possible moves
        node_score = -1
        node_moves = {}
        moves_to_eval = []
        legal_moves = list(self.board.legal_moves)
        next_level_legal_moves = {}
        moves_num = len(legal_moves)
        board_inputs = np.zeros(shape=(moves_num+1, 8, 8, 12), dtype=np.int8)
        castling_inputs = np.zeros(shape=(moves_num+1, 4), dtype=np.int8)
        
        for m in legal_moves:
            self.board.push(m)
            move_uci = m.uci()
            result = self.result()
            if result != -1:
                score = -0.001
                if result == 0.5:
                    score = 0.5
                if 1 - score > node_score:
                    node_score = 1 - score
                node_moves[move_uci] = NodeLink(-1, Node(score))
            else:
                board_matrix, castling_matrix = self.input_matrix(self.board)
                index = len(moves_to_eval)
                board_inputs[index] = board_matrix
                castling_inputs[index] = castling_matrix
                moves_to_eval.append(move_uci)
                next_level_legal_moves[move_uci] = list(self.board.legal_moves)
                self.meta_data['inferences'] += 1
            self.board.pop()

        # also evaluate current state
        board_matrix, castling_matrix = self.input_matrix(self.board)
        index = len(moves_to_eval)
        board_inputs[index] = board_matrix
        castling_inputs[index] = castling_matrix

        board_inputs = board_inputs[:len(moves_to_eval) + 1]
        castling_inputs = castling_inputs[:len(moves_to_eval) + 1]

        # run model
        t1 = time.time()
        policies, values = self.model.predict([board_inputs, castling_inputs], batch_size=64)
        self.meta_data['infer_time'] += time.time() - t1
        for i, m in enumerate(moves_to_eval):
            childrens_links = {}
            for next_move in next_level_legal_moves[m]:
                next_move_uci = next_move.uci()
                move_index = self.move_encoder.uci_to_index(next_move_uci, not self.board.turn)
                weight = policies[i][move_index]
                link = NodeLink(weight)
                childrens_links[next_move_uci] = link
            child_score = values[i][0]

            if 1 - child_score > node_score:
                node_score = 1 - child_score
            node_moves[m] = NodeLink(-1, Node(child_score, childrens_links))

        for m in legal_moves:
            move_uci = m.uci()
            move_index = self.move_encoder.uci_to_index(move_uci, self.board.turn)
            node_moves[move_uci].weight = policies[index][move_index]

        self.meta_data['expand_time'] += time.time() - expand_t1
        return Node(node_score, node_moves)

    def result(self):
        if self.board.is_checkmate():
            return 0
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0.5
        if len(self.board.move_stack) > 16 and self.board.halfmove_clock >= 6:
            if self.board.move_stack[-1] == self.board.move_stack[-5]:
                if self.board.move_stack[-2] == self.board.move_stack[-6]:
                    # counting repetitions is expensive so do prelim checks
                    if self.board.can_claim_threefold_repetition():
                        return 0.5
        return -1
        
    def next_moves(self):
        return self.node().children(include_unvisited=True)

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

    def __str__(self):
        return str({'score': self.score, 'visits': self.visits})

class NodeLink():
    def __init__(self, weight, node=None):
        self.weight = weight
        self.node = node

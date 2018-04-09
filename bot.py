import chess
import numpy as np
from model_densenet import get_model, save_model
from tensorflow.python.client import device_lib
from move_encoder import MoveEncoder
import math
import time
import random
from node import Node

class ChessBot():
    def __init__(self):
        self.move_encoder = MoveEncoder()
        gpus = len(device_lib.list_local_devices()) - 1
        self.model, self.model_template = get_model(len(self.move_encoder.moves), gpus)
        self.inferences = 0
        self.explore = 0.3
        self.init_explore = 1.5
        self.max_depth = 35
        self.same_score_threshold = 0.001
        self.root_node = None
        self.root_node_id = None

    def save_model(self):
        save_model(self.model_template)

    def board_to_input(self, board):
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

    def node_id(self, board):
        return [move.uci() for move in board.move_stack]

    def best_move(self, board, depth=7, time_limit=10, debug=False, eval_freq=15):
        b = board.copy()
        if self.root_node == None:
            self.root_node_id = self.node_id(b)
            self.root_node = self.new_node(b)
        else:
            # check if board is within root node subtree
            new_node_id = self.node_id(b)
            if new_node_id[:len(self.root_node_id)] != self.root_node_id:
                self.root_node_id = new_node_id
                self.root_node = self.new_node(b)
                if debug:
                    print('New game detected')
            else:
                new_node = self.root_node.traverse(new_node_id[len(self.root_node_id):])
                if debug:
                    print('Traversing down the tree:')
                    print(new_node)
                if new_node == None:
                    self.root_node_id = new_node_id
                    self.root_node = self.new_node(b)
                else:
                    self.root_node_id = new_node_id
                    self.root_node = new_node

        best_move = self.mcts(b, depth, time_limit, debug, eval_freq)
        return best_move

    def mcts(self, board, depth, time_limit, debug, eval_freq):
        self.inferences = 0
        node = self.root_node
        cutoff_time = time.time() + time_limit
        eval_time = time.time() + eval_freq

        if len(node.child_links) == 1:
            return list(node.child_links.keys())[0]

        while True:
            # stale game search
            if node.visits > 100:
                if depth < self.max_depth and node.visits % 10 == 0:
                    if board.halfmove_clock > 15:
                        depth += 1
                    elif node.score < 1 and node.score > 0.98:
                        depth += 1

            # simulate game
            self.simulate_game(board, node, depth)

            # out of time
            if time.time() > cutoff_time:
                break

            if time.time() > eval_time:
                moves = node.children(include_unvisited=True)
                moves = sorted(moves, key=lambda x: x['visits'], reverse=True)
                # end early if confident enough
                best_move_lower_bound = self.calc_ucb(moves[0], node.visits, multiplier=-1)
                other_move_upper_bound = None
                for move in moves[1:]:
                    ucb = self.calc_ucb(move, node.visits, multiplier=0.9)
                    if other_move_upper_bound == None or ucb > other_move_upper_bound:
                        other_move_upper_bound = ucb
                if debug:
                    print('total simulations:', node.visits, 'depth:', depth)
                    print(moves[:3])
                    print('lb:', best_move_lower_bound, 'ub:', other_move_upper_bound)
                if best_move_lower_bound > other_move_upper_bound:
                    return moves[0]
                eval_time = time.time() + eval_freq

        moves = node.children(include_unvisited=True)
        moves = sorted(moves, key=lambda x: x['visits'], reverse=True)

        if debug:
            print('total simulations:', node.visits, 'depth:', depth)
            print(moves[:3])

        best_score = moves[0]['score']
        # if mcts scores are similar then choose move based of policy model score
        best_moves = []
        for move in moves:
            if move['score'] > best_score - self.same_score_threshold:
                best_moves.append(move)
        best_moves = sorted(best_moves, key=lambda x: x['weight'], reverse=True)
        if best_moves[0] != moves[0]:
            if debug:
                print('best move:')
                print(best_moves[0])
            return best_moves[0]
        return moves[0]

    def simulate_game(self, board, node, depth):
        if not node.child_links:
            return
        if depth == 0:
            return

        next_move_uci = self.choose_next_move(node)
        board.push_uci(next_move_uci)
        next_move_link = node.child_links[next_move_uci]
        if next_move_link.node == None:
            next_move_link.node = self.new_node(board)
        next_state = next_move_link.node
        self.simulate_game(board, next_state, depth-1)
        next_move_score = 1 - next_state.score
        board.pop()
        node.visits += 1
        
        best_state = None
        for state in node.children():
            if best_state == None or state['score'] < best_state['score']:
                best_state = state
        best_score = 1 - best_state['score']
        best_score_weight = best_state['visits'] / (best_state['visits'] + next_state.visits)

        score_delta = best_score * best_score_weight + (1 - best_score_weight) * next_move_score

        node.score = (node.score * (node.visits-1) + score_delta ) / node.visits
        # node.score = sample_score

    def choose_next_move(self, node):
        next_move = None
        max_ucb = None
        for uci, link in node.child_links.items():
            ucb = self.calc_ucb(link, node.visits)
            if max_ucb == None or ucb > max_ucb:
                max_ucb = ucb
                next_move = uci
        return next_move

    def calc_ucb(self, node_link, simulation_num, multiplier=1):
        # calculate confidence bound for move
        if node_link.node == None:
            return self.init_explore * simulation_num - 1 / (node_link.weight + 0.0005)
        return 1 - node_link.node.score + self.explore * math.sqrt(math.log(simulation_num) / (node_link.node.visits + 1)) * multiplier

    def validate_moves(self, board, moves_array):
        legal_moves = np.zeros(shape=(len(self.move_encoder.moves),), dtype=np.int8)
        for move in board.legal_moves:
            uci = move.uci()
            move_index = self.move_encoder.uci_to_index(uci, board.turn)
            legal_moves[move_index] = 1
        valid_moves = np.multiply(legal_moves, moves_array)
        valid_moves /= np.sum(valid_moves)
        return valid_moves

    def new_node(self, board):
        result = board.result(claim_draw=True)
        if result != '*':
            # end of game
            score = -0.001
            if result == '1/2-1/2':
                score = 0.5
            node = Node(score)
            return node
                
        self.inferences += 1

        board_inputs = np.zeros(shape=(1, 8, 8, 12), dtype=np.int8)
        castling_inputs = np.zeros(shape=(1, 4), dtype=np.int8)

        # input
        board_matrix, castling_matrix = self.board_to_input(board)
        board_inputs[0] = board_matrix
        castling_inputs[0] = castling_matrix

        # run model
        policies, values = self.model.predict([board_inputs, castling_inputs])
        policy, value = policies[0], values[0]
        policy = self.validate_moves(board, policy)
        node_children = []
        for move_index, weight in enumerate(policy):
            if weight > 0:
                node_children.append({
                    'move': self.move_encoder.index_to_uci(move_index, board.turn),
                    'weight': weight
                })
        node = Node(value, node_children)
        return node



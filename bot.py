import chess
import numpy as np
from model_densenet import get_model, save_model
from tensorflow.python.client import device_lib
from move_encoder import MoveEncoder
import math
import time
import random
from node import Game

class ChessBot():
    def __init__(self):
        self.move_encoder = MoveEncoder()
        gpus = len(device_lib.list_local_devices()) - 1
        self.model, self.model_template = get_model(len(self.move_encoder.moves), gpus)
        self.explore = 0.2
        self.init_explore = 1.2
        self.max_depth = 35
        self.same_score_threshold = 0.001
        self.epsilon = 0.0005
        self.game = Game(self.model)

    def save_model(self):
        save_model(self.model_template)

    def best_move(self, board, depth=7, time_limit=10, debug=False, eval_freq=15):
        self.game.set_position(board)
        if debug:
            print(self.game.root())

        best_move = self.mcts(depth, time_limit, debug, eval_freq)
        return best_move

    def mcts(self, depth, time_limit, debug, eval_freq):
        cutoff_time = time.time() + time_limit
        eval_time = time.time() + eval_freq

        if len(self.game.next_moves()) == 1:
            return self.game.next_moves()[0]['move']

        node = self.game.node()
        board = self.game.board
        while True:
            # stale game search
            if node.visits > 100:
                if depth < self.max_depth and node.visits % 10 == 0:
                    if board.halfmove_clock > 15:
                        depth += 1
                    elif node.score < 1 and node.score > 0.98:
                        depth += 1

            # simulate game
            self.simulate_game(depth)

            # out of time
            if time.time() > cutoff_time:
                break

            if time.time() > eval_time:
                moves = sorted(self.game.next_moves(), key=lambda x: (x['visits'], x['score']), reverse=True)
                # end early if confident enough
                
                if debug:
                    print('total simulations:', node.visits, 'depth:', depth)
                    print(moves[:3])

                if moves[0]['visits'] > moves[1]['visits'] * 10:
                    return moves[0]
                eval_time = time.time() + eval_freq

        moves = sorted(self.game.next_moves(), key=lambda x: (x['visits'], x['score']), reverse=True)

        if debug:
            print('total simulations:', node.visits, 'depth:', depth)
            print(moves[:3])

        # best_score = moves[0]['score']
        # # if mcts scores are similar then choose move based of policy model score
        # best_moves = []
        # for move in moves:
        #     if move['score'] > best_score - self.same_score_threshold:
        #         best_moves.append(move)
        # best_moves = sorted(best_moves, key=lambda x: x['weight'], reverse=True)
        # if best_moves[0] != moves[0]:
        #     if debug:
        #         print('best move:')
        #         print(best_moves[0])
        #     return best_moves[0]
        return moves[0]

    def simulate_game(self, depth):
        node = self.game.node()
        node.visits += 1
        if len(node.child_links) == 0 or depth == 0:
            return

        sample_move = self.choose_next_move(node)
        self.game.push(sample_move)
        self.simulate_game(depth-1)
        sample_node = self.game.node()
        sample_score = 1 - sample_node.score
        self.game.pop()
        
        best_move = None
        for move in node.children():
            if best_move == None or move['score'] < best_move['score']:
                best_move = move
        best_score = 1 - best_move['score']
        best_score_weight = best_move['visits'] / (best_move['visits'] + sample_node.visits)

        score_delta = best_score * best_score_weight + (1 - best_score_weight) * sample_score

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
            return self.init_explore * simulation_num - 1 / (node_link.weight + self.epsilon)
        return 1 - node_link.node.score + self.explore * math.sqrt(math.log(simulation_num) / (node_link.node.visits + 1)) * multiplier

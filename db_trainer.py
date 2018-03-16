import chess
from bot import ChessBot
import sqlite3
import numpy as np
from tensorflow.python.client import device_lib

class DBTrainer():
    def __init__(self, db_path='/data/kru03a/chbot/data/moves_standard.db', chbot=None):
        self.chbot = chbot or ChessBot()
        self.db_path = db_path
        gpus = len(device_lib.list_local_devices()) - 1
        self.BATCH_SIZE = 128 * gpus

    def format_data(self, batch):
        inputs1 = np.zeros(shape=(len(batch), 8, 8, 12), dtype=np.int8)
        inputs2 = np.zeros(shape=(len(batch), 4), dtype=np.int8)
        outputs1 = np.zeros(shape=(len(batch), len(self.chbot.move_encoder.moves)), dtype=np.int8)
        outputs2 = np.zeros(shape=(len(batch), 2))
        for i, datum in enumerate(batch):
            fen = datum[0]
            move = datum[1]
            winner = datum[2]
            board = chess.Board(fen=fen)
            move_index = self.chbot.move_encoder.uci_to_index(move, board.turn)
            board_matrix, castling_matrix = self.chbot.board_to_input(board)
            inputs1[i] = board_matrix
            inputs2[i] = castling_matrix
            outputs1[i][move_index] = 1
            outputs2[i] = [1, 0] if winner == board.turn else [0, 1]
        return [inputs1, inputs2], [outputs1, outputs2]

    def epoch(self):
        print(self.chbot.model.metrics_names)
        db = sqlite3.connect(self.db_path)
        cursor = db.cursor()
        cursor.execute('select count(*) from moves_train')
        iterations = round(cursor.fetchall()[0][0] / self.BATCH_SIZE)
        iteration = 0
        metrics = np.zeros(len(self.chbot.model.metrics_names), dtype=np.float)
        iterations_metrics = 0
        cursor.execute('select fen, move, winner from moves_train order by random()')
        while True:
            batch = cursor.fetchmany(self.BATCH_SIZE)
            iteration += 1
            iterations_metrics += 1
            if len(batch) == 0:
                print(str(round(iteration / iterations * 100, 2)) + '%', metrics / iterations_metrics, flush=True)
                break
            inputs, outputs = self.format_data(batch)
            metrics += self.chbot.model.train_on_batch(inputs, outputs)
            if iteration % 500 == 0:
                # test_board = chess.Board()
                # test_inputs, test_outputs = self.format_data([(test_board.fen(), 'e2e4', 1)])
                # print(self.chbot.model.predict(test_inputs))
                print(str(round(iteration / iterations * 100, 2)) + '%', metrics / iterations_metrics, flush=True)
                metrics = np.zeros(len(self.chbot.model.metrics_names), dtype=np.float)
                iterations_metrics = 0
            if iteration % 2000 == 0:
                self.chbot.save_model()
        db.close()
        self.chbot.save_model()

    def test(self):
        db = sqlite3.connect(self.db_path)
        cursor = db.cursor()
        cursor.execute('select count(*) from moves_val')
        iterations = round(cursor.fetchall()[0][0] / self.BATCH_SIZE)
        iteration = 0
        metrics = np.zeros(len(self.chbot.model.metrics_names), dtype=np.float)
        iterations_metrics = 0
        cursor.execute('select fen, move, winner from moves_val order by random() limit 100000')
        while True:
            batch = cursor.fetchmany(self.BATCH_SIZE)
            iteration += 1
            iterations_metrics += 1
            if len(batch) == 0:
                print(str(round(iteration / iterations * 100, 2)) + '%', metrics / iterations_metrics, flush=True)
                break
            inputs, outputs = self.format_data(batch)
            metrics += self.chbot.model.test_on_batch(inputs, outputs)
            if iteration % 50 == 0:
                print(str(round(iteration / iterations * 100, 2)) + '%', metrics / iterations_metrics, flush=True)
                metrics = np.zeros(len(self.chbot.model.metrics_names), dtype=np.float)
                iterations_metrics = 0
        db.close()

from multiprocessing import Process, Manager
import os
import numpy as np
from time import sleep, time

class InferenceEngine(Process):
    def __init__(self):
        manager = Manager()
        self.input_dict = manager.dict()
        self.output_dict = manager.dict()
        self.train_queue = manager.Queue()
        Process.__init__(self)
        self.io_wait_time = 0.0001
        self.infer_time = 0

    def run(self):
        from model_densenet import get_model, save_model
        model, template_model = get_model()
        self.model = model
        self.template_model = template_model
        self.p_save_model = save_model
        try:
            while True:
                if len(self.input_dict) > 0:
                    self.p_predict()
                if not self.train_queue.empty():
                    self.p_fit()
                else:
                    sleep(self.io_wait_time)
        except (EOFError, BrokenPipeError, TypeError, FileNotFoundError, ConnectionResetError) as e:
            print('InferenceEngine exit', e)
            print('Infer time', self.infer_time)

    def p_predict(self):
        # receive input
        temp_i = dict()
        for pid, val in self.input_dict.items():
            temp_i[pid] = val
            del self.input_dict[pid]

        # format input
        index_to_pid = dict()
        input_batch1 = np.zeros(shape=(len(temp_i), 8, 8, 12))
        input_batch2 = np.zeros(shape=(len(temp_i), 4))
        pids = temp_i.keys()
        for i, pid in enumerate(pids):
            index_to_pid[i] = pid
            input_batch1[i] = temp_i[pid][0]
            input_batch2[i] = temp_i[pid][1]

        # run model
        t1 = time()
        outputs1, outputs2 = self.model.predict([input_batch1, input_batch2], batch_size=64)
        self.infer_time += time() - t1

        # process output
        for i in range(len(temp_i)):
            pid = index_to_pid[i]
            self.output_dict[pid] = (outputs1[i], outputs2[i])

    def predict(self, inp):
        pid = os.getpid()
        if pid in self.output_dict:
            del self.output_dict[pid]
        self.input_dict[pid] = inp
        while pid not in self.output_dict:
            sleep(self.io_wait_time)
        return self.output_dict[pid]

    def p_fit(self):
        while not self.train_queue.empty():
            inputs, outputs = self.train_queue.get_nowait()
            self.model.fit(inputs, outputs, verbose=1, batch_size=32)
        self.p_save_model(self.model_template)

    def fit(self, inputs, outputs):
        self.train_queue.put((inputs, outputs))

    
if __name__ == '__main__':
    ie = InferenceEngine()
    ie.start()

    inp1 = np.zeros(shape=(8, 8, 12))
    inp2 = np.zeros(shape=4)

    for i in range(3):
        t1 = time()
        out1, out2 = ie.predict((inp1, inp2))
        print('out_delay', time() - t1)
        print(out1.shape)
        print(out2.shape)
    
from multiprocessing import Process, Manager
import os
import numpy as np
from time import sleep, time

class InferenceEngine(Process):
    def __init__(self):
        manager = Manager()
        self.input_dict = manager.dict()
        self.output_dict = manager.dict()
        Process.__init__(self)
        self.io_wait_time = 0.0001

    def run(self):
        from model_densenet import get_model, save_model
        model, template_model = get_model()
        try:
            while True:
                if self.input_dict == None:
                    return
                if len(self.input_dict) > 0:
                    print('running', os.getpid())

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

                    # process input
                    output = model.predict([input_batch1, input_batch2], batch_size=64)

                    # process output
                    for i in range(len(temp_i)):
                        pid = index_to_pid[i]
                        self.output_dict[pid] = output[i]
                    print('finished', os.getpid())
                else:
                    sleep(self.io_wait_time)
        except EOFError as e:
            pass

    def process_input(self, inp):
        pid = os.getpid()
        if pid in self.output_dict:
            del self.output_dict[pid]
        self.input_dict[pid] = inp
        while pid not in self.output_dict:
            sleep(self.io_wait_time)
        return self.output_dict[pid]

    
if __name__ == '__main__':
    ie = InferenceEngine()
    ie.start()
    sleep(5)

    inp = np.zeros(shape=(8, 8, 12))
    t1 = time()
    out = ie.process_input(inp)
    print('out_delay', time() - t1)
    print(out)
    
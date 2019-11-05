import threading
import time
import numpy as np
import globalvar as gl
import queue

service_time = 100  # ms
flow_size = 0
break_flag = False
exit_flag = False
tran_flow = np.zeros((100,100))
#tran_flow = 0
drop_count_list = np.zeros(100)
throughput_list = np.zeros(100)
latency_list = np.zeros(100)
ins_num_list = np.zeros(100, dtype=np.int)


class flow:
    s_time = 0
    f_time = 0


class RX_thread(threading.Thread):
    def __init__(self, rx_queue, drop_count, id, connection):
        super().__init__()
        self.rx_queue = rx_queue
        self.drop_count = drop_count
        self.id = id
        self.connection = connection

    def run(self):
        while True:
            f_size = int(self.get_flow_size(self.id))
            s_time = time.time()
            for i in range(f_size):
                tmp_flow = flow
                tmp_flow.s_time = s_time
                if not self.rx_queue.full():
                    self.rx_queue.put_nowait(tmp_flow)
                else:
                    self.drop_count = self.drop_count + 1
            self.set_drop_count(self.id, self.drop_count)
            time.sleep(0.5)
            if break_flag:
                break

    def get_flow_size(self, id):
        f_s = 0
        if self.id == 1:
            f_s = flow_size
        else:
            for i in range(len(tran_flow[:,id-1])):
                if tran_flow[i][id-1] > 0:
                    f_s = tran_flow[i][id-1]
            #f_s = tran_flow
        return f_s

    def set_drop_count(self, id, value):
        global drop_count_list
        drop_count_list[id] = value


class handel_flow(threading.Thread):
    def __init__(self, rx_queue, tx_queue, name, id, connection):
        super().__init__()
        self.rx_queue = rx_queue
        self.ins_num = 0
        self.tx_queue = tx_queue
        self.throughput = 0
        self.name = name
        self.id = id
        self.packet_loss = 0
        self.delay_list = []
        self.avg_delay = 0
        self.connection = connection

    def run(self):
        global exit_flag
        while True:
            f_batch = []
            timer1 = time.time()
            f_count = 0
            while not self.rx_queue.empty():
                self.ins_num = self.get_ins_num(self.id)
                for i in range(self.ins_num):
                    if not self.rx_queue.empty():
                        f = self.rx_queue.get_nowait()
                        f_batch.append(f)
                time.sleep(service_time / 1000)
                end_time = time.time()
                for j in range(len(f_batch)):
                    f_batch[j].f_time = end_time
                    self.tx_queue.put_nowait(f_batch[j])
                    if len(self.delay_list) == 10:
                        self.avg_delay = self.get_avg_delay(self.delay_list)
                        self.delay_list.clear()
                        #str1 = "%s Latency: %d ms  Packet loss: %d" % (self.name, self.avg_delay, drop_count_list[self.id])
                        #print(str1)
                        self.set_latency(self.id, self.avg_delay)
                    delay = (f_batch[j].f_time - f_batch[j].s_time) * 1000
                    self.delay_list.append(delay)
                f_count = f_count + len(f_batch)
                f_batch.clear()
                timer2 = time.time()
                if timer2 - timer1 >= 1:
                    self.throughput = f_count
                    self.set_tran_flow(self.throughput, self.id, self.connection)
                    #str2 = "%s Throughput: %d f/s" % (self.name, self.throughput)
                    #print(str2)
                    self.set_throughput(self.id, self.throughput)
                    f_count = 0
                    timer1 = time.time()
                if break_flag:
                    exit_flag = True
                    break
            if exit_flag:
                break

    def set_tran_flow(self, f_s, id, connection):
        next_id = 0
        global tran_flow
        for j in range(len(connection[id-1])):
            if connection[id-1][j] == 1:  #maybe need to change
                next_id = j + 1
        tran_flow[id-1][next_id-1] = f_s
        #tran_flow = f_s

    def get_avg_delay(self, delay_list):
        sum_delay = 0
        for i in range(len(delay_list)):
            sum_delay = sum_delay + delay_list[i]
        avg_delay = sum_delay / 10
        return avg_delay

    def set_throughput(self, id, value):
        global throughput_list
        throughput_list[id] = value

    def set_latency(self, id, value):
        global latency_list
        latency_list[id] = value

    def get_ins_num(self, id):
        ins_num = ins_num_list[id]
        return ins_num


class sender(threading.Thread):
    def __init__(self, flow_size_list):
        super().__init__()
        gl.init()
        self.flow_size_list = flow_size_list

    def run(self):
        global break_flag
        index = 0
        while index < len(self.flow_size_list):
            f_size = self.flow_size_list[index]
            self.set_flow_size(f_size)
            index = index + 1
            time.sleep(0.5)
        break_flag = True
        gl.set_value(break_flag)

    def set_flow_size(self, f_s):
        global flow_size
        flow_size = f_s


class helper():
    def get_throughput(self, id):
        throughput_result = throughput_list[id]
        return throughput_result

    def get_latency(self, id):
        latency_result = latency_list[id]
        return latency_result

    def get_packetloss(self, id):
        packetloss_result = drop_count_list[id]
        return packetloss_result

    def set_vm_ins_num(self, id, vm):
        ins_num_list[id] = vm

    def get_vm_ins_num(selfself, id):
        return ins_num_list[id]

import threading
import time
import numpy as np
import globalvar as gl
import queue
import sys

service_time = 100  # ms
flow_size = 0
break_flag = False
exit_flag = False
round_start_flag = False
complete_flag = False
rx_clear_flag = False
handle_clear_flag = False
final_round_flag = False
default_vm_num = 1
delay_span = 50
tran_flow = np.zeros((100,100))
#tran_flow = 0
drop_count_list = np.zeros(100)
throughput_list = np.zeros(100)
latency_list = np.zeros(100)
rx_throughput_list = np.zeros(100)
ins_num_list = np.zeros(100, dtype=np.int)
sfc = [1, 3, 4]


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
        global rx_clear_flag
        global complete_flag
        global drop_count_list
        while True:
            timer1 = time.time()
            f_count = 0
            f_size = int(self.get_flow_size(self.id))
            s_time = time.time()

            for i in range(f_size):
                tmp_flow = flow
                tmp_flow.s_time = s_time
                if not self.rx_queue.full():
                    self.rx_queue.put_nowait(tmp_flow)
                    f_count = f_count + 1
                else:
                    self.drop_count = self.drop_count + 1
            self.set_drop_count(self.id, self.drop_count)
            if complete_flag:
                if not self.rx_queue.empty():
                    self.rx_queue.queue.clear()
                self.drop_count = 0
                drop_count_list = np.zeros(100)
                rx_clear_flag = True
                #print('Node', self.id, 'rx thread para clear')
                if handle_clear_flag and rx_clear_flag and gl.get_continue_flag():
                    complete_flag = False
                    #print("Node", self.id, "In rx thread both thread clear")
            if break_flag:
                break
            timer2 = time.time()
            while timer2 - timer1 < 1:
                time.sleep(0.01)
                timer2 = time.time()
            # if timer2 - timer1 >= 1:
            #     self.set_rx_throughput(self.id, f_count)
            # if timer2 - timer1 >= 1:
            #     str = "Node %d Rx_throughput: %d f/s\n" % (self.id, f_count)
            #     sys.stdout.write(str)
            #time.sleep(0.5)

    def get_flow_size(self, id):
        f_s = 0
        if self.id == sfc[0]:
            f_s = flow_size
            self.set_rx_throughput(id, flow_size)
        else:
            for i in range(len(tran_flow[:,id-1])):
                if tran_flow[i][id-1] > 0:
                    f_s = tran_flow[i][id-1]
            #f_s = tran_flow
        return f_s

    def set_drop_count(self, id, value):
        global drop_count_list
        drop_count_list[id] = value

    def set_rx_throughput(self,id, value):
        global rx_throughput_list
        rx_throughput_list[id] = value




class handle_flow(threading.Thread):
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
        global complete_flag
        global handle_clear_flag
        while True:
            sr = "%d alive\n" % (self.id)
            sys.stdout.write(sr)
            f_batch = []
            #timer1 = time.time()
            f_count = 0
            while not self.rx_queue.empty() or self.id > 0:
                while self.rx_queue.empty():
                    time.sleep(0.1)
                self.ins_num = self.get_ins_num(self.id)
                #timer3 = time.time()
                for i in range(self.ins_num):
                    #timer2 = time.time()
                    # if self.id == sfc[0]:
                    #     if not self.rx_queue.empty() and timer2 - timer1 < 1:
                    #         #f = self.rx_queue.get_nowait()
                    #         f = self.rx_queue.get()
                    #         f_count = f_count + 1
                    #         f_batch.append(f)
                    # else:
                    #     if not self.rx_queue.empty() and timer2 - timer1 < 1 and f_count <= self.get_last_throughput(self.id):
                    #         #f = self.rx_queue.get_nowait()
                    #         f = self.rx_queue.get()
                    #         f_count = f_count + 1
                    #         f_batch.append(f)
                    if not self.rx_queue.empty():
                        # f = self.rx_queue.get_nowait()
                        f = self.rx_queue.get()
                        f_count = f_count + 1
                        f_batch.append(f)
                time.sleep(service_time / 1000)
                #timer4 = time.time()
                end_time = time.time()
                #if timer2 - timer1 >= 1:
                #self.throughput = f_count / (timer4 - timer3)
                #self.throughput = f_count * (1000/ service_time)
                self.throughput = self.ins_num * (1000 / service_time)
                if self.throughput > self.get_rx_throughput(self.id) and self.id == sfc[0]:
                    self.throughput = rx_throughput_list[sfc[0]]
                if self.throughput > self.get_rx_throughput(self.id) and self.id != sfc[0]:
                    self.throughput = self.get_rx_throughput(self.id)
                self.set_tran_flow(self.throughput, self.id, self.connection)
                # str2 = "%s Throughput: %d f/s VMs: %d\n" % (
                # self.name, self.throughput, self.ins_num)
                # sys.stdout.write(str2)
                #self.set_throughput(self.id, self.throughput / (timer2 - timer1))
                self.set_throughput(self.id, self.throughput)
                f_count = 0
                #timer1 = time.time()
                for j in range(len(f_batch)):
                    f_batch[j].f_time = end_time
                    self.tx_queue.put_nowait(f_batch[j])
                    if len(self.delay_list) == delay_span:
                        self.avg_delay = self.get_avg_delay(self.delay_list)
                        self.delay_list.clear()
                        #str1 = "%s Latency: %d ms  Packet loss: %d" % (self.name, self.avg_delay, drop_count_list[self.id])
                        #print(str1)
                        self.set_latency(self.id, self.avg_delay)
                    delay = (f_batch[j].f_time - f_batch[j].s_time) * 1000
                    self.delay_list.append(delay)

                f_batch.clear()

                if complete_flag:
                    self.clear_para(self.rx_queue,self.tx_queue,self.delay_list)
                    handle_clear_flag = True
                    #print('Node', self.id, 'handle thread para clear')
                    if handle_clear_flag and rx_clear_flag and gl.get_continue_flag():
                        complete_flag = False
                        #print("Node", self.id, "In handle thread both thread clear")
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
        self.set_rx_throughput(next_id, f_s)
        #tran_flow = f_s

    def get_avg_delay(self, delay_list):
        sum_delay = 0
        for i in range(len(delay_list)):
            sum_delay = sum_delay + delay_list[i]
        avg_delay = sum_delay / delay_span
        return avg_delay

    def set_throughput(self, id, value):
        global throughput_list
        throughput_list[id] = value

    def set_latency(self, id, value):
        global latency_list
        latency_list[id] = value

    def get_last_throughput(self, id):
        last_id = 0
        for i in range(len(sfc)):
            if sfc[i] == id:
                last_id = sfc[i-1]
        last_th = throughput_list[last_id]
        return last_th

    def get_ins_num(self, id):
        ins_num = ins_num_list[id]
        # ins_num = 1
        # if id == 1:
        #     ins_num = 2
        # if id == 3:
        #     ins_num = 10
        # if id == 4:
        #     ins_num = 4
        return ins_num

    def set_rx_throughput(self,id, value):
        global rx_throughput_list
        rx_throughput_list[id] = value

    def get_rx_throughput(self,id):
        return rx_throughput_list[id]

    def clear_para(self, rx_queue, tx_queue, delay_l):
        global throughput_list
        global latency_list
        global ins_num_list
        global tran_flow
        throughput_list = np.zeros(100)
        latency_list = np.zeros(100)
        tran_flow = np.zeros((100, 100))
        if not rx_queue.empty():
            rx_queue.queue.clear()
        if not tx_queue.empty():
            tx_queue.queue.clear()
        delay_l.clear()
        for i in range(len(ins_num_list)):
            ins_num_list[i] = default_vm_num



class sender(threading.Thread):
    def __init__(self, flow_size_list, MAX_EPISODES):
        super().__init__()
        gl.init()
        self.flow_size_list = flow_size_list
        self.MAX_EPISODES = MAX_EPISODES


    def run(self):
        global break_flag
        #global round_start_flag
        global complete_flag
        global rx_clear_flag
        global handle_clear_flag
        global final_round_flag
        for i in range(self.MAX_EPISODES):
            index = 0
            rx_clear_flag = False
            handle_clear_flag = False
            #round_start_flag = True
            time.sleep(0.5)
            c_f = gl.get_continue_flag()
            if (c_f or i == 0) and not complete_flag:
                if c_f:
                    continue_flag = False
                    gl.set_continue_flag(continue_flag)
                r_s_f = True
                gl.set_round_start_flag(r_s_f)
                while not gl.get_ready_flag():
                    time.sleep(0.1)
                if gl.get_ready_flag():
                    r_s_f_ = False
                    r_f_ = False
                    gl.set_round_start_flag(r_s_f_)
                    gl.set_ready_flag(r_f_)
                    while index < len(self.flow_size_list):
                        f_size = self.flow_size_list[index]
                        self.set_flow_size(f_size)
                        index = index + 1
                        time.sleep(1)
                        if gl.get_continue_flag():
                            break
                str = "Round %d complete\n" % (i+1)
                sys.stdout.write(str)
                if i + 2 >= self.MAX_EPISODES:
                    final_round_flag = True
                while not gl.get_learn_complete_flag():
                    time.sleep(0.1)
                if gl.get_learn_complete_flag():
                    complete_flag = True
                    l_f_ = False
                    gl.set_learn_complete_flag(l_f_)
                #round_start_flag = False
                time.sleep(2)
        if gl.get_continue_flag():
            break_flag = True
            gl.set_break_flag(break_flag)

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

    def get_vm_ins_num(self, id):
        return ins_num_list[id]

    def get_final_round(self):
        return final_round_flag

    def get_complete_flag(self):
        return complete_flag

    def get_round_start_flag(self):
        return round_start_flag

    def get_rx_throughput(self, id):
        return rx_throughput_list[id]
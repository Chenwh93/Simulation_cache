import queue
import threading
from All_thread import RX_thread, handel_flow


class VNF_Node(object):
    def __init__(self,queue_size, id):
        self.rx_queue = queue.Queue(queue_size)
        self.tx_queue = queue.Queue()
        self.throughput = 0
        self.delay = 0
        self.packet_loss = 0
        self.drop_count = 0
        #self.ins_num = 1
        self.r_flow_size = 0
        self.id = id



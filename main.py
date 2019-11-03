import numpy as np
import time
import threading
from VNF_Node import VNF_Node
from All_thread import sender, RX_thread, handel_flow, helper
from topology import Topo

queue_size = 1000 # default
#flow_size = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
#flow_size = [31, 26, 34, 20, 15, 49, 25, 29, 26, 48, 45, 35, 12, 26, 47, 40, 47, 17, 27, 46]
#flow_size = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
flow_size = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
             80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]

sfc_list = [1, 2, 3, 4]
sfc = [1, 3, 4]
vnf_support_list = [20, 24, 28, 32]
up_len = 20

index = 0
node_len = len(vnf_support_list)
node_list = []
rx_thread_list = []
handel_thread_list = []



# connection = np.array([
#     [0,1,0,0,0,0,0,0,0,0],
#     [0,0,1,0,0,0,0,0,0,0],
#     [0,0,0,1,0,0,0,0,0,0],
#     [0,0,0,0,1,0,0,0,0,0],
#     [0,0,0,0,0,1,0,0,0,0],
#     [0,0,0,0,0,0,1,0,0,0],
#     [0,0,0,0,0,0,0,1,0,0],
#     [0,0,0,0,0,0,0,0,1,0],
#     [0,0,0,0,0,0,0,0,0,1],
#     [0,0,0,0,0,0,0,0,0,0],
# ])
connection = np.zeros([node_len+1, node_len+1],dtype=np.int)
# connection = np.array([
#     [0,1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,0,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,0],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,1],
#     [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0]
# ])

def run_proc1():
    for i in range(node_len):
        tmp_node = VNF_Node(queue_size=queue_size, id=i+1)
        node_list.append(tmp_node)
    sender1 = sender(flow_size)
    node_list[0].ins_num = 3
    for i in range(node_len):
        tmp_rx_thread = RX_thread(node_list[i].rx_queue, node_list[i].drop_count, node_list[i].id, connection)
        tmp_handel_thread = handel_flow(node_list[i].rx_queue, node_list[i].ins_num, node_list[i].tx_queue, "node"+str(node_list[i].id), node_list[i].id, connection)
        rx_thread_list.append(tmp_rx_thread)
        handel_thread_list.append(tmp_handel_thread)
    sender1.start()
    for i in range(node_len):
        rx_thread_list[i].start()
        handel_thread_list[i].start()
    sender1.join()
    for i in range(node_len):
        rx_thread_list[i].join()
        handel_thread_list[i].join()

def run_proc2():
    time.sleep(5)
    while True:
        h = helper()
        for i in range(node_len):
            throughput = h.get_throughput(i+1)
            latency = h.get_latency(i+1)
            packetloss = h.get_packetloss(i+1)
            str = "%s%d Throughput %d f/s  Latency: %d ms  Packet loss: %d" % ("node", i+1, throughput, latency, packetloss)
            print(str)
        time.sleep(1)

def set_connection(conn, path):
    server_path = []
    for i in range(len(path)):
        if path[i] in vnf_support_list:
            server_path.append(path[i])
    aix = []
    for j in range(len(vnf_support_list)):
        if vnf_support_list[j] in server_path:
            aix.append(j)
    for k in range(len(aix)):
        if k + 1 < len(aix):
            conn[aix[k]][aix[k+1]] = 1
    conn[aix[-1]][node_len] = 1
    return

if __name__ == "__main__":
    env = Topo()
    p = env.get_path(21, 33, sfc)
    set_connection(connection, p)
    threads = []
    threads.append(threading.Thread(target=run_proc1))
    threads.append(threading.Thread(target=run_proc2))
    for t in threads:
        t.start()
import numpy as np
import time
import threading
import tensorflow as tf
import matplotlib.pyplot as plt
from VNF_Node import VNF_Node
from All_thread import sender, RX_thread, handle_flow, helper
from topology import Topo
from DDPG import DDPG
import globalvar as gl
import sys

queue_size = 1000  # default
# flow_size = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
# flow_size = [31, 26, 34, 20, 15, 49, 25, 29, 26, 48, 45, 35, 12, 26, 47, 40, 47, 17, 27, 46]
#flow_size = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
# flow_size = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80]
flow_size = np.linspace(60,60,50, dtype=np.int)

sfc_list = [1, 2, 3, 4]
sfc = [1, 3, 4]
vnf_support_list = [20, 24, 28, 32]
up_len = 20
default_vm_num = 1
n_state = 4
node_capacity = [10, 10, 10, 10]

throughput_factor = 1
latency_factor = 0.01
packetloss_factor = 0.9
cost_factor = 0.1
cost = 1

MAX_EPISODES = 20
MAX_EP_STEPS = 800

index = 0
node_len = len(vnf_support_list)
node_list = []
rx_thread_list = []
handle_thread_list = []
node_active_stats = [False, False, False, False]
# active_node_list = []

action_space = ['ScaleIn','NA','ScaleOut']
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
MEMORY_CAPACITY = 10000


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
connection = np.zeros([node_len + 1, node_len + 1], dtype=np.int)


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

def get_active_node():
    active = []
    for i in range(len(node_active_stats)):
        if node_active_stats[i]:
            active.append(i + 1)
    return active


def get_state():
    h = helper()
    throughput_list = []
    latency_list = []
    packetloss_list = []
    ins_num_list = []
    for i in sfc:
        throughput_list.append(h.get_throughput(i)/100)
        latency_list.append(h.get_latency(i)/1000)
        packetloss_list.append(h.get_packetloss(i)/100000)
        ins_num_list.append(h.get_vm_ins_num(i)/20)
    return throughput_list, latency_list, packetloss_list, ins_num_list

def dict_to_list(d):
    l = []
    for key in d:
        l.append(d[key])
    return l


def vm_scale_in(vm):
    vm_ = vm - 1
    return vm_


def vm_scale_out(vm):
    vm_ = vm + 1
    return vm_


def step(action):
    reward = []
    th, la, pa, ins = get_state()
    h = helper()
    for a in action:
        if action[a] == 'ScaleIn' or action[a] == 0:
            vm = h.get_vm_ins_num(a)
            vm_ = vm_scale_in(vm)
            if vm_ == 0:
                h.set_vm_ins_num(a, 1)
            else:
                h.set_vm_ins_num(a, vm_)
        if action[a] == 'NA' or action[a] == 1:
            pass
        if action[a] == 'ScaleOut' or action[a] == 2:
            vm = h.get_vm_ins_num(a)
            vm_ = vm_scale_out(vm)
            if vm_ <= node_capacity[a-1]:
                h.set_vm_ins_num(a, vm_)
            else:
                h.set_vm_ins_num(a, node_capacity[a-1])
    th_, la_, pa_, ins_ = get_state()
    active_node_list = get_active_node()
    for i in range(len(active_node_list)):
        #inst_num = h.get_vm_ins_num(active_node_list[i])
        r = throughput_factor * th_[i] - latency_factor * la_[i] - packetloss_factor * pa_[i] - cost_factor * ins_[i] * cost
        #r = throughput_factor * th_[i] - packetloss_factor * pa_[i] - cost_factor * ins_[i] * cost
        reward.append(r)
    return reward, th_, la_, pa_, ins_

def update():
    count = 0
    #while True:
    for j in range(MAX_EP_STEPS):
        h = helper()
        observation_th, observation_la, observation_pa, observation_ins = get_state()
        action_set = dict()
        active_node_list = get_active_node()
        for i in range(len(active_node_list)):
            observation_l = [observation_th[i], observation_la[i], observation_pa[i], observation_ins[i]]
            #observation_l = [observation_th[i], observation_pa[i], observation_ins[i]]
            a = ddpgs[i].choose_action(np.reshape(observation_l, n_state))
            action = {active_node_list[i]: a}
            action_set.update(action)
        reward, observation_th_, observation_la_, observation_pa_, observation_ins_ = step(action_set)

        action_list = dict_to_list(action_set)

        for i in range(len(active_node_list)):
            observation_l = [observation_th[i], observation_la[i], observation_pa[i], observation_ins[i]]
            observation_l_ = [observation_th_[i], observation_la_[i], observation_pa_[i], observation_ins_[i]]
            # observation_l = [observation_th[i], observation_pa[i], observation_ins[i]]
            # observation_l_ = [observation_th_[i], observation_pa_[i], observation_ins_[i]]
            ddpgs[i].store_transition(observation_l, action_list[i], reward[i], observation_l_)

        for i in range(len(active_node_list)):
            if ddpgs[i].pointer > MEMORY_CAPACITY:
                ddpgs[i].learn()

        count = count + 1
        #if h.get_complete_flag():
        if j + 1 >= MAX_EP_STEPS:
            for i in range(len(active_node_list)):
                str1 = "Node %d Reward: %f\n" % (active_node_list[i], reward[i])
                #print('Node', active_node_list[i], ' Reward: ', reward[i])
                sys.stdout.write(str1)
            for i in range(len(active_node_list)):
                throughput = h.get_throughput(active_node_list[i])
                latency = h.get_latency(active_node_list[i])
                packetloss = h.get_packetloss(active_node_list[i])
                str = "%s%d Throughput %d f/s  Latency: %d ms  Packet loss: %d\n" % (
                    "node", active_node_list[i], throughput, latency, packetloss)
                sys.stdout.write(str)
            continue_flag = True
            gl.set_continue_flag(continue_flag)
            #print('Count:', count)
            count = 0


def create_record_list():
    th_l = []
    la_l = []
    pa_l = []
    for i in range(len(sfc_list)):
        th_tmp_l = []
        th_l.append(th_tmp_l)
        la_tmp_l = []
        la_l.append(la_tmp_l)
        pa_tmp_l = []
        pa_l.append(pa_tmp_l)
    return th_l, la_l, pa_l

def run_proc1():
    for i in range(node_len):
        tmp_node = VNF_Node(queue_size=queue_size, id=i + 1)
        node_list.append(tmp_node)
    sender1 = sender(flow_size, MAX_EPISODES)
    helper1 = helper()
    for i in range(node_len):
        helper1.set_vm_ins_num(node_list[i].id, default_vm_num)
        tmp_rx_thread = RX_thread(node_list[i].rx_queue, node_list[i].drop_count, node_list[i].id, connection)
        tmp_handle_thread = handle_flow(node_list[i].rx_queue, node_list[i].tx_queue, "node" + str(node_list[i].id),
                                        node_list[i].id, connection)
        rx_thread_list.append(tmp_rx_thread)
        handle_thread_list.append(tmp_handle_thread)
    sender1.start()
    for i in range(node_len):
        rx_thread_list[i].start()
        handle_thread_list[i].start()
    sender1.join()
    for i in range(node_len):
        rx_thread_list[i].join()
        handle_thread_list[i].join()


def run_proc2():
    time.sleep(5)
    th_l, la_l, pa_l = create_record_list()
    while True:
        h = helper()
        if h.get_final_round():
            for i in sfc:
                throughput = h.get_throughput(i)
                latency = h.get_latency(i)
                packetloss = h.get_packetloss(i)
                # str = "%s%d Throughput %d f/s  Latency: %d ms  Packet loss: %d\n" % (
                # "node", i, throughput, latency, packetloss)
                #sys.stdout.write(str)
                th_l[i-1].append(throughput)
                la_l[i-1].append(latency)
                pa_l[i-1].append(packetloss)
            time.sleep(1)
            f = gl.get_break_flag()
            if f:
                plt.figure()
                for i in range(len(sfc_list)):
                    plt.subplot(3,len(sfc_list), i+1)
                    plt.plot(np.arange(len(th_l[i])), th_l[i])
                for i in range(len(sfc_list)):
                    plt.subplot(3,len(sfc_list), i+5)
                    plt.plot(np.arange(len(la_l[i])), la_l[i])
                for i in range(len(sfc_list)):
                    plt.subplot(3,len(sfc_list), i+9)
                    plt.plot(np.arange(len(pa_l[i])), pa_l[i])
                plt.show()
                break


def run_proc3():
    while True:
        if gl.get_round_start_flag():
            r_f = True
            gl.set_ready_flag(r_f)
            time.sleep(5)
            update()
            l_f = True
            gl.set_learn_complete_flag(l_f)
        f = gl.get_break_flag()
        if f:
            break


def set_connection(conn, path):
    server_path = []
    for i in range(len(path)):
        if path[i] in vnf_support_list:
            server_path.append(path[i])
    aix = []
    for j in range(len(vnf_support_list)):
        if vnf_support_list[j] in server_path:
            aix.append(j)
            node_active_stats[j] = True
    for k in range(len(aix)):
        if k + 1 < len(aix):
            conn[aix[k]][aix[k + 1]] = 1
    conn[aix[-1]][node_len] = 1
    return


if __name__ == "__main__":
    env = Topo()
    sess = []
    p = env.get_path(21, 33, sfc)
    set_connection(connection, p)
    ddpgs = []
    active_node_list = get_active_node()
    action_len = len(action_space)
    for i in range(len(active_node_list)):
        sess.append(tf.Session())
        ddpgs.append(DDPG(sess[i], n_state=n_state, n_action=action_len, id=i, a_learning_rate=LR_A, c_learning_rate= LR_C))
    threads = []
    threads.append(threading.Thread(target=run_proc1))
    threads.append(threading.Thread(target=run_proc2))
    threads.append(threading.Thread(target=run_proc3))
    for t in threads:
        t.start()

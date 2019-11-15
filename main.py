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
import os
from matplotlib.pyplot import plot,savefig

sys.stdout = open('sim_recode.log', mode = 'w',encoding='utf-8')

queue_size = 1000  # default
# flow_size = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
# flow_size = [31, 26, 34, 20, 15, 49, 25, 29, 26, 48, 45, 35, 12, 26, 47, 40, 47, 17, 27, 46]
# flow_size = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
# flow_size = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
#              80, 80, 80, 80, 80, 80, 80, 80, 80]
flow_size = np.linspace(60, 60, 300, dtype=np.int)

sfc_list = [1, 2, 3, 4]
sfc = [1, 3, 4]
vnf_support_list = [20, 24, 28, 32]
up_len = 20
default_vm_num = 1
n_state = 2
node_capacity = [10, 10, 10, 10]

throughput_factor = 0.9
latency_factor = 0.01
packetloss_factor = 1
cost_factor = 0.5
cost = 1

MAX_EPISODES = 60
MAX_EP_STEPS = 500

index = 0
node_len = len(vnf_support_list)
node_list = []
rx_thread_list = []
handle_thread_list = []
node_active_stats = [False, False, False, False]
# active_node_list = []

action_space = ['ScaleIn', 'NA', 'ScaleOut']
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
MEMORY_CAPACITY = 10000
var = [3, 3, 3, 3]
c = [0, 0, 0, 0]
target_util = 1
well_reward_count = [0, 0, 0, 0]
random_init = 5
epsilon_decay = 0.995
last_u = -999

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
rw_l = []


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
        throughput_list.append(h.get_throughput(i) / 100)
        latency_list.append(h.get_latency(i) / 1000)
        packetloss_list.append(h.get_packetloss(i) / 60000)
        ins_num_list.append(h.get_vm_ins_num(i) / 10)
    return throughput_list, latency_list, packetloss_list, ins_num_list


def get_util():
    h = helper()
    util_list = []
    throughput_list = []
    packetloss_list = []
    ins_num_list = []
    for i in sfc:
        # th = h.get_rx_throughput(i)
        th = h.get_rx_throughput(sfc[0])
        vm = h.get_vm_ins_num(i) * 10
        util_list.append((th / vm))
        throughput_list.append(h.get_throughput(i) / 100)
        packetloss_list.append(h.get_packetloss(i) / 10000)
        ins_num_list.append(h.get_vm_ins_num(i) / 10)
    return util_list, throughput_list, packetloss_list, ins_num_list


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


def find_index(a_l, a):
    index = 0
    for i in range(len(a_l)):
        if a_l[i] == a:
            index = i
    return index


def step(action):
    global well_reward_count
    global last_u
    reward = []
    # th, la, pa, ins = get_state()
    # u = get_util()
    h = helper()
    scale_reward = 0
    for a in action:
        if action[a] == 'ScaleIn' or action[a] == 0:
            vm = h.get_vm_ins_num(a)
            vm_ = vm_scale_in(vm)
            if last_u == target_util:
                scale_reward = 10
            if vm_ == 0:
                h.set_vm_ins_num(a, 1)
            else:
                h.set_vm_ins_num(a, vm_)
            # pass
        if action[a] == 'NA' or action[a] == 1:
            pass
        if action[a] == 'ScaleOut' or action[a] == 2:
            vm = h.get_vm_ins_num(a)
            vm_ = vm_scale_out(vm)
            if last_u == target_util:
                scale_reward = 10
            if vm_ <= node_capacity[a - 1]:
                h.set_vm_ins_num(a, vm_)
            else:
                h.set_vm_ins_num(a, node_capacity[a - 1])
            # pass
    # time.sleep(1)
    # th_, la_, pa_, ins_ = get_state()
    u_, th_, pa_, ins_ = get_util()
    last_u = u_
    active_node_list = get_active_node()
    for i in range(len(active_node_list)):
        # inst_num = h.get_vm_ins_num(active_node_list[i])
        # r = throughput_factor * th_[i] - packetloss_factor * pa_[i] - cost_factor * ins_[i] * cost
        # r = throughput_factor * th_[i] - cost_factor * ins_[i] * cost
        # util_error = abs((target_util - u[i]) / target_util)
        util_error_ = abs((target_util - u_[i]) / target_util)
        # r = - util_error_ - packetloss_factor * pa_[i] - cost_factor * ins_[i] * cost
        # r = - util_error_ - cost_factor * ins_[i] * cost
        #r = - util_error_ - scale_reward
        r = th_[i] - cost_factor * ins_[i] - scale_reward
        if r >= -0.2:
            well_reward_count[active_node_list[i] - 1] += 1
        reward.append(r)
    # return reward, th_, la_, pa_, ins_
    return reward, u_, th_, pa_, ins_


def update():
    global rw_l
    global var
    global c
    global random_init
    # while True:
    act_l = create_action_list()
    for j in range(MAX_EP_STEPS):
        h = helper()
        # observation_th, observation_la, observation_pa, observation_ins = get_state()
        observation_ul, observation_th, observation_pa, observation_ins = get_util()
        action_set = dict()
        a_l = []

        active_node_list = get_active_node()
        for i in range(len(active_node_list)):
            # observation_l = [observation_th[i], observation_pa[i], observation_ins[i]]
            observation_l = [observation_th[i], observation_ins[i]]
            #observation_l = [observation_ul[i]]
            pr = ddpgs[i].choose_action(np.reshape(observation_l, n_state))
            pr_ = pr.ravel().copy()
            k1 = int(np.argmax(pr_))
            k2 = find_index(pr_, np.random.choice(pr_))
            #noise = np.clip(np.random.normal(0, var[active_node_list[i] - 1]), 0, 1)
            noise = round(np.random.rand(1)[0], 3)
            #k1 = find_index(pr_, np.random.choice(pr_))
            if random_init > 0:
                # k2 = find_index(pr_, np.random.choice(pr_))
                # pr_[k1] += noise
                # pr_[k2] += np.random.rand(1)[0]
                if noise > pr_[k1]:
                    tmp = round(pr_[k1] * 0.9, 3)
                    pr_[k1] = round(pr_[k1] - tmp, 6)
                    pr_[k2] = round(pr_[k2] + tmp, 6)
                else:
                    pr_[k1] = round(pr_[k1] - noise, 6)
                    pr_[k2] = round(pr_[k2] + noise, 6)
            else:
                pass
                #pr_[k1] += noise
            # print(noise)
            a_l.append(pr_)
            # print(pr.ravel(), pr_)
            # a = np.random.choice(np.arange(pr.shape[1]), p=pr_)
            a = np.argmax(pr_)
            # a_ = np.clip(np.random.normal(a, var[active_node_list[i]-1]), 0, 2)
            action = {active_node_list[i]: a}
            act_l[active_node_list[i] - 1].append(a)
            # print(a, action)
            action_set.update(action)
        # reward, observation_th_, observation_la_, observation_pa_, observation_ins_ = step(action_set)
        reward, observation_ul_, observation_th_, observation_pa_, observation_ins_ = step(action_set)

        # action_list = dict_to_list(action_set)

        for i in range(len(active_node_list)):
            observation_l = [observation_th[i], observation_ins[i]]
            observation_l_ = [observation_th_[i], observation_ins_[i]]
            #observation_l = [observation_ul[i]]
            # print(active_node_list[i], "l", observation_l, action_set[active_node_list[i]])
            #observation_l_ = [observation_ul_[i]]
            # print(active_node_list[i], "l_", observation_l_)
            rw_l[active_node_list[i] - 1].append(reward[i])
            ddpgs[i].store_transition(observation_l, a_l[i], reward[i], observation_l_)
        a_l.clear()
        for i in range(len(active_node_list)):
            if ddpgs[i].pointer > MEMORY_CAPACITY:
                c[active_node_list[i] - 1] += 1
                var[active_node_list[i] - 1] *= epsilon_decay
                ddpgs[i].learn()

        # for i in range(len(active_node_list)):
        #     rx_th = h.get_rx_throughput(active_node_list[i])
        #     throughput = h.get_throughput(active_node_list[i])
        #     vm = h.get_vm_ins_num(active_node_list[i])
        #     str = "%s%d RX_Throughput %d f/s Throughput %d f/s VM: %d\n" % (
        #         "node", active_node_list[i], rx_th, throughput, vm)
        #     sys.stdout.write(str)

        # if h.get_complete_flag():
        if j + 1 >= MAX_EP_STEPS:
            for i in range(len(active_node_list)):
                str1 = "Node %d Reward: %f Explore: %f Count: %d Well Reward_Count: %d\n" % (
                    active_node_list[i], reward[i], var[active_node_list[i] - 1], c[active_node_list[i] - 1],
                    well_reward_count[active_node_list[i] - 1])
                # print('Node', active_node_list[i], ' Reward: ', reward[i])
                sys.stdout.write(str1)
            for i in range(len(active_node_list)):
                throughput = h.get_throughput(active_node_list[i])
                latency = h.get_latency(active_node_list[i])
                packetloss = h.get_packetloss(active_node_list[i])
                ins_num = h.get_vm_ins_num(active_node_list[i])
                str2 = "%s%d Throughput %d f/s  Latency: %d ms  Packet loss: %d VM: %d\n" % (
                    "node", active_node_list[i], throughput, latency, packetloss, ins_num)
                sys.stdout.write(str2)
            for i in range(len(active_node_list)):
                str3 = "%s%d Action: %s\n" % ("Node", active_node_list[i], str(act_l[active_node_list[i] - 1]))
                sys.stdout.write(str3)
            continue_flag = True
            random_init -= 1
            gl.set_continue_flag(continue_flag)
            # print('Count:', count)
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


def create_action_list():
    act_l = []
    for i in range(len(sfc_list)):
        act_tmp_l = []
        act_l.append(act_tmp_l)
    return act_l


def create_reward_list():
    rw = []
    for i in range(len(sfc_list)):
        rw_tmp_l = []
        rw.append(rw_tmp_l)
    return rw


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
    time.sleep(10)
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
                # sys.stdout.write(str)
                th_l[i - 1].append(throughput)
                la_l[i - 1].append(latency)
                pa_l[i - 1].append(packetloss)

            time.sleep(0.1)
            f = gl.get_break_flag()
            if f:
                print('Running time: ', time.time() - t1)
                plt.figure()
                for i in range(len(sfc_list)):
                    plt.subplot(3, len(sfc_list), i + 1)
                    plt.plot(np.arange(len(th_l[i])), th_l[i])
                for i in range(len(sfc_list)):
                    plt.subplot(3, len(sfc_list), i + 5)
                    plt.plot(np.arange(len(la_l[i])), la_l[i])
                for i in range(len(sfc_list)):
                    plt.subplot(3, len(sfc_list), i + 9)
                    plt.plot(np.arange(len(pa_l[i])), pa_l[i])
                #plt.show()
                savefig("sim_1.png")
                plt.figure()
                for i in range(len(sfc_list)):
                    plt.subplot(len(sfc_list), 1, i + 1)
                    plt.plot(np.arange(len(rw_l[i])), rw_l[i])
                #plt.show()
                savefig("sim_2.png")
                sys.stdout.close()
                os._exit(0)
                break


def run_proc3():
    global rw_l
    rw_l = create_reward_list()
    while True:
        if gl.get_round_start_flag():
            r_f = True
            gl.set_ready_flag(r_f)
            time.sleep(10)
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
    t1 = time.time()
    env = Topo()
    sess = []
    p = env.get_path(21, 33, sfc)
    set_connection(connection, p)
    ddpgs = []
    active_node_list = get_active_node()
    action_len = len(action_space)
    for i in range(len(active_node_list)):
        sess.append(tf.Session())
        ddpgs.append(DDPG(sess[i], n_state=n_state, n_action=action_len, id=active_node_list[i], a_learning_rate=LR_A,
                          c_learning_rate=LR_C))
    threads = []
    threads.append(threading.Thread(target=run_proc1))
    threads.append(threading.Thread(target=run_proc2))
    threads.append(threading.Thread(target=run_proc3))
    for t in threads:
        t.start()

import numpy as np
import random
from topology import Topo
import queue

sfc_list = [1, 2, 3, 4]

sfc = [1, 3, 4]
vnf_support_list = [20, 24, 28, 32]
up_len = 20
node_len = len(vnf_support_list)
connection = np.zeros([node_len+1, node_len+1],dtype=np.int)

def set_connection(conn, path):
    server_path = []
    for i in range(len(path)):
        if path[i] in vnf_support_list:
            server_path.append(path[i])
    print(server_path)
    aix = []
    for j in range(len(vnf_support_list)):
        if vnf_support_list[j] in server_path:
            aix.append(j)
    print(aix)
    for k in range(len(aix)):
        if k + 1 < len(aix):
            conn[aix[k]][aix[k+1]] = 1
    conn[aix[-1]][node_len] = 1
    return


#env = Topo()
# p = env.get_path(21, 33, sfc)
# print(p)
# set_connection(connection,p)
# print(connection)
#env.print_topo()
#p = env.get_shortest_path(20,29)
#print(c)
def find_index(a_l,a):
    index = 0
    for i in range(len(a_l)):
        if a_l[i] == a:
            index = i
    return index
var = 1
for i in range(1500):
    a = [0.343, 0.345, 0.312]
    a_ = a.copy()
    #s = np.clip(np.random.normal(0, var), 0, 1)
    s = np.random.rand(1)[0]
    s = round(s, 3)

    print(s)
    k1 = int(np.argmax(a_))
    k2 = find_index(a_, np.random.choice(a_))
    if s > a[k1]:
        tmp = round(a_[k1] * 0.9, 3)
        print(tmp)
        a_[k1] = round(a_[k1] - tmp, 6)
        a_[k2] = round(a_[k2] + tmp, 6)
    else:
        a_[k1] = round(a_[k1] - s, 6)
        a_[k2] = round(a_[k2] + s, 6)

    print(a)
    print(a_)
    var *= .995
    #print(var)
# a = np.random.rand(1)[0]
#
# print(a)

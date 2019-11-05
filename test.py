import numpy as np
from topology import Topo

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


# env = Topo()
# p = env.get_path(21, 33, sfc)
# print(p)
# set_connection(connection,p)
# print(connection)
#env.print_topo()
#p = env.get_shortest_path(20,29)
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
th_l[0].append(1)
print(th_l)
print(la_l)
print(pa_l)
#print(c)
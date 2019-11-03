import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

core_switch_list = np.arange(4)
aggr_switch_list = np.arange(8)
edge_switch_list = np.arange(8)
server_list = np.arange(16)

up_aggr = len(core_switch_list)
up_edge = up_aggr + len(aggr_switch_list)
up_server = up_edge + len(edge_switch_list)

vnf_support_list = [20, 24, 28, 32]


class Topo(object):
    def __init__(self):
        self._build_topo()
        self.vnf_support_list = vnf_support_list

    def _build_topo(self):
        self.G = nx.Graph()
        # create node
        for i in range(len(core_switch_list)):
            self.G.add_node(core_switch_list[i], type="core_switch", id=core_switch_list[i])
        for i in range(len(aggr_switch_list)):
            self.G.add_node(up_aggr + aggr_switch_list[i], type="core_switch", id=aggr_switch_list[i])
        for i in range(len(edge_switch_list)):
            self.G.add_node(up_edge + edge_switch_list[i], type="core_switch", id=edge_switch_list[i])
        for i in range(len(server_list)):
            self.G.add_node(up_server + server_list[i], type="core_switch", id=server_list[i])
        # create edge
        for i in range(len(edge_switch_list)):
            self.G.add_edge(up_edge + edge_switch_list[i], up_server + server_list[2 * i])
            self.G.add_edge(up_edge + edge_switch_list[i], up_server + server_list[2 * i + 1])
        for i in range(len(aggr_switch_list)):
            if i % 2 == 0:
                self.G.add_edge(up_aggr + aggr_switch_list[i], up_edge + edge_switch_list[i])
                self.G.add_edge(up_aggr + aggr_switch_list[i], up_edge + edge_switch_list[i + 1])
            else:
                self.G.add_edge(up_aggr + aggr_switch_list[i], up_edge + edge_switch_list[i - 1])
                self.G.add_edge(up_aggr + aggr_switch_list[i], up_edge + edge_switch_list[i])
        for i in range(len(core_switch_list)):
            if i < len(core_switch_list) / 2:
                for j in range(len(aggr_switch_list)):
                    if j % 2 == 0:
                        self.G.add_edge(core_switch_list[i], up_aggr + aggr_switch_list[j])
            else:
                for j in range(len(aggr_switch_list)):
                    if j % 2 == 1:
                        self.G.add_edge(core_switch_list[i], up_aggr + aggr_switch_list[j])

    def print_topo(self):
        nx.draw(self.G, with_labels=True)
        plt.show()

    def get_shortest_path(self, s, t):
        path = nx.shortest_path(self.G, source=s, target=t)
        return path

    def add_element_to_list(self, s_list, t_list):
        for i in range(len(s_list)):
            t_list.append(s_list[i])

    def get_path(self, s, t, sfc):
        path = []
        tmp_p = self.get_shortest_path(s, vnf_support_list[sfc[0] - 1])
        self.add_element_to_list(tmp_p[0:-1], path)
        for i in range(len(sfc)):
            if i + 1 < len(sfc):
                tmp_p = self.get_shortest_path(vnf_support_list[sfc[i] - 1], vnf_support_list[sfc[i + 1] - 1])
                self.add_element_to_list(tmp_p[0:-1], path)
        self.add_element_to_list(self.get_shortest_path(vnf_support_list[sfc[-1] - 1], t), path)
        return path

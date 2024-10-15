# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import copy
seed = 1
np.random.seed(seed)
random.seed(seed)


class VNF:
    def __init__(self, vnf_type):
        self.type = vnf_type
        self.cpu = np.random.uniform(40, 50)
        self.storage = np.random.uniform(40, 50)
        self.forward = np.random.uniform(40, 50)
        self.deployed_node = None


class SFC:
    def __init__(self, network):
        self.network = network
        self.vnf_types = ['Firewall', 'Load Balancer', 'Cache', 'IDS', 'router', 'switch']
        self.chain = []
        self.edges = []
        self.source = random.choice(list(network.G.nodes))
        self.destination = random.choice(list(network.G.nodes))
        while self.source == self.destination:
            self.destination = random.choice(list(network.G.nodes))
        self.priority = random.randint(1, 3)

        for vnf_type in self.vnf_types:
            vnf = VNF(vnf_type)
            self.chain.append(vnf)

    def print_chain_resources(self):
        for idx, vnf in enumerate(self.chain):
            print(f"VNF {idx + 1} ({vnf.type}): CPU={vnf.cpu:.2f}, Storage={vnf.storage:.2f},forward={vnf.forward:.2f}")


class PhysicalNetwork:
    def __init__(self, topology_file):
        self.G = nx.read_gml(topology_file)
        for node in self.G.nodes:
            cpu = np.random.uniform(100, 180)
            storage = np.random.uniform(100, 180)
            forward = np.random.uniform(100, 180)
            self.G.nodes[node].clear()
            self.G.nodes[node].update({'cpu':  cpu,  # 节点的资源信息
                                       'storage': storage,
                                       'forward': forward,
                                       'cpu_used': 0,  # 已使用资源量
                                       'storage_used': 0,
                                       'forward_used': 0,
                                       'cpu_remain': cpu,  # 剩余资源量
                                       'storage_remain': storage,
                                       'forward_remain': forward})
        for edge in self.G.edges:
            bandwidth = np.random.uniform(500, 600)
            delay = np.random.uniform(2, 5)
            self.G.edges[edge].clear()
            self.G.edges[edge].update({'bandwidth': bandwidth,  # 边的资源信息
                                       'bandwidth_used': 0,  # 已使用资源量
                                       'bandwidth_remain': bandwidth,  # 剩余资源量
                                       'delay': delay})

        self.font_size, self.node_size, self.width = 3, 100, .5  # 显示网络的字体大小，节点大小, 线宽
        self.pos = nx.spring_layout(self.G)  # 弹簧布局算法
        # self.pos = nx.circular_layout(self.graph, scale=6, center=None, dim=2)  # 环形布局算法

    def display_net(self):
        '''
            显示原始网络拓扑图
        '''
        nx.draw(self.G, self.pos, with_labels=True, node_color='skyblue', node_size=self.node_size, font_size=self.font_size)
        nx.draw_networkx_edges(self.G, self.pos, width=self.width, alpha=0.5, edge_color='gray')
        edge_labels = {edge: f"{delay:.1f}" for edge, delay in nx.get_edge_attributes(self.G, 'delay').items()}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=self.font_size)
        plt.title("Physical Network Topology with SFC Chains")
        plt.show()

    def save_sfc_topology(self, sfcs, sfc_paths):
        '''
        保存sfc部署路径的图形
            sfcs: 多条包含vnf的sfc
            sfc_paths: 多条包含vnf连通边的sfc路径
        '''
        if not os.path.exists("sfc_image"):
            os.makedirs("sfc_image")  # 创建保存 SFC 图像的目录

        # 单独绘制每条 SFC 并保存到文件
        for i, (sfc, sfc_path) in enumerate(zip(sfcs, sfc_paths)):
            plt.figure()
            nx.draw(self.G, self.pos, with_labels=True, node_color='skyblue', node_size=self.node_size, font_size=self.font_size)
            nx.draw_networkx_edges(self.G, self.pos, width=self.width, alpha=0.5, edge_color='gray')
            edge_labels = {edge: f"{delay:.1f}" for edge, delay in nx.get_edge_attributes(self.G, 'delay').items()}
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=self.font_size)

            colors = ['red',  'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'yellow', 'magenta']
            color = colors[i % len(colors)]
            # 高亮显示 VNF 节点
            vnf_nodes = [vnf.deployed_node for vnf in sfc.chain]
            nx.draw(self.G, self.pos, nodelist=[vnf_nodes[0]], node_color='yellow', node_size=self.node_size, font_size=self.font_size)
            nx.draw(self.G, self.pos, nodelist=vnf_nodes[1:-1], node_color=color, node_size=self.node_size, font_size=self.font_size)
            nx.draw(self.G, self.pos, nodelist=[vnf_nodes[-1]], node_color='magenta', node_size=self.node_size, font_size=self.font_size)
            # 显示当前 SFC 的路径
            nx.draw_networkx_edges(self.G, self.pos, edgelist=[(sfc_path[i], sfc_path[i+1])
                                   for i in range(len(sfc_path)-1)], width=self.width, edge_color=color)
            # 显示sfc的路径文本
            sfc_path_string = ''.join([f"{node} -> " for node in sfc_path])[:-4]
            plt.title(f"SFC {i+1}: {sfc_path_string}", transform=plt.gca().transAxes, fontsize=3*self.font_size)
            plt.savefig(f"sfc_image/sfc{i+1}.svg", format="svg")
            # plt.show()
            plt.close()

    def deploy_sfc(self, sfc, search_method='min_delay'):
        '''
        部署一条SFC，确保满足资源和时延约束
            sfc: SFC对象
            search_method: 部署算法，
                'min_delay'表示最短路径，
                'min_edge'表示最少边数, 
        '''
        # 备份原始网络资源信息, 部署失败时可以恢复原来网络状态
        G_nodes_backup = copy.deepcopy(self.G.nodes)
        G_edges_backup = copy.deepcopy(self.G.edges)

        deployed_nodes = []
        deployed_path = []
        for vnf in sfc.chain:
            # 找到可放置当前vnf的节点
            available_nodes = [node for node in self.G.nodes if self.check_node_resources(node, vnf) and node not in deployed_nodes]
            if not available_nodes:
                print(f"无法找到合适的节点来部署 VNF {vnf.type}")
                self.G.nodes = G_nodes_backup
                self.G.edges = G_edges_backup
                return False, None

            if len(deployed_nodes) == 0:
                # 如果这是第一个VNF，随机选择一个可用节点
                selected_node = random.choice(available_nodes)
                vnf.deployed_node = selected_node
                deployed_nodes.append(selected_node)
                deployed_path.append(selected_node)
                self.update_node_resources(selected_node, vnf)  # 更新节点资源
            else:
                # 找到前一个部署的VNF节点，查找从前一个节点到当前可用节点的最短路径
                prev_node = deployed_nodes[-1]
                min_delay = float('inf')
                min_edge = float('inf')
                best_node = None
                best_path = None

                for node in available_nodes:
                    try:
                        if search_method == 'min_delay':
                            path_min_delay = nx.shortest_path(self.G, source=prev_node, target=node, weight='delay')  # 找到时间最短路径
                            delay = sum([self.G.edges[(path_min_delay[i], path_min_delay[i+1])]['delay'] for i in range(len(path_min_delay) - 1)])
                            if delay < min_delay and self.cheak_edge_resources(path_min_delay, vnf):
                                min_delay = delay
                                best_node = node
                                best_path = path_min_delay
                        elif search_method == 'min_edge':
                            path_min_edge = nx.shortest_path(self.G, source=prev_node, target=node)  # 找到最短距离（最少边数）的路径
                            edge_num = len(path_min_edge)-1
                            if edge_num < min_edge and self.cheak_edge_resources(path_min_edge, vnf):
                                min_edge = edge_num
                                best_node = node
                                best_path = path_min_edge
                    except nx.NetworkXNoPath:
                        continue

                if best_node is None or best_path is None:
                    # 没有找到最优节点恢复原来网络状态
                    print(f"无法找到从节点 {prev_node} 到其他节点的路径")
                    self.G.nodes = G_nodes_backup
                    self.G.edges = G_edges_backup
                    return False, None

                # 部署VNF并更新资源
                vnf.deployed_node = best_node
                deployed_nodes.append(best_node)
                deployed_path += best_path[1:]
                # print(deployed_nodes, deployed_path)
                self.update_node_resources(best_node, vnf)
                self.update_edge_resources(best_path, vnf)

        return True, deployed_path

    def check_node_resources(self, node, vnf):
        '''
        检查节点资源是否足够部署VNF
        '''
        if self.G.nodes[node]['cpu_remain'] < vnf.cpu or self.G.nodes[node]['storage_remain'] < vnf.storage or self.G.nodes[node]['forward_remain'] < vnf.forward:
            return False
        return True

    def cheak_edge_resources(self, path, vnf):
        '''
        检查路径上的剩余带宽资源是否足够部署VNF
        '''
        for i in range(len(path) - 1):
            if self.G.edges[(path[i], path[i+1])]['bandwidth_remain'] < vnf.forward:
                return False
        return True

    def update_node_resources(self, node, vnf):
        '''
        更新节点资源，分配VNF使用的资源
        '''
        self.G.nodes[node]['cpu_used'] += vnf.cpu
        self.G.nodes[node]['cpu_remain'] -= vnf.cpu
        self.G.nodes[node]['storage_used'] += vnf.storage
        self.G.nodes[node]['storage_remain'] -= vnf.storage
        self.G.nodes[node]['forward_used'] += vnf.forward
        self.G.nodes[node]['forward_remain'] -= vnf.forward

    def update_edge_resources(self, path, vnf):
        '''
        更新路径上的带宽和延迟资源
        '''
        for i in range(len(path) - 1):
            self.G.edges[(path[i], path[i+1])]['bandwidth_used'] += vnf.forward
            self.G.edges[(path[i], path[i+1])]['bandwidth_remain'] -= vnf.forward

    def check_net_is_overloaded(self):
        '''
        检查网络是否过载
        '''
        for node in self.G.nodes:
            if self.G.nodes[node]['cpu_remain'] < 0 or self.G.nodes[node]['storage_remain'] < 0 or self.G.nodes[node]['forward_remain'] < 0:
                print('node_res overload')
                return True
        for edge in self.G.edges:
            if self.G.edges[edge]['bandwidth_remain'] < 0:
                print('edge overload', edge)
                return True
        return False


topology_file = "./Chinanet.gml"
network = PhysicalNetwork(topology_file)

sfc_num = 20
sfcs = []
sfc_paths = []

for i in range(sfc_num):
    sfc = SFC(network)
    success, deployed_path = network.deploy_sfc(sfc,)
    if success:
        sfcs.append(sfc)
        sfc_paths.append(deployed_path)
    print(i, network.check_net_is_overloaded())
network.save_sfc_topology(sfcs, sfc_paths)
# %%

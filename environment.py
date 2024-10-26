# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import copy


class VNF:
    def __init__(self, vnf_type):
        self.type = vnf_type
        self.cpu = np.random.uniform(40, 50)
        self.storage = np.random.uniform(40, 50)
        self.forward = np.random.uniform(40, 50)
        self.deployed_node = None


class SFC:
    def __init__(self, G):
        self.vnf_types = ['Firewall', 'Load Balancer', 'Cache', 'IDS', 'router', 'switch']
        self.chain = []
        self.source = np.random.choice(list(G.nodes))
        self.destination = np.random.choice(list(G.nodes))
        while self.source == self.destination:
            self.destination = np.random.choice(list(G.nodes))
        self.priority = np.random.randint(1, 3)
        for vnf_type in self.vnf_types:
            vnf = VNF(vnf_type)
            self.chain.append(vnf)


class PhysicalNetwork:
    def __init__(self, topology_file, sfc_num=6, overload_threshold=0.85, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.G = nx.read_gml(topology_file)
        mapping = {node: i for i, node in enumerate(self.G.nodes)}
        self.G = nx.relabel_nodes(self.G, mapping)  # 节点名称修改为0-n形式

        self.sfc_num = sfc_num
        self.overload_threshold = overload_threshold
        self.sfcs = []
        self.sfc_paths = []

        self._nodes_edges_resource_init()
        self._sfc_init()
        self.G_nodes_backup = copy.deepcopy(self.G.nodes)
        self.G_edges_backup = copy.deepcopy(self.G.edges)
        self.sfcs_backup = copy.deepcopy(self.sfcs)

        self.font_size, self.node_size, self.width = 3, 100, .5  # 显示网络的字体大小，节点大小, 线宽
        self.pos = nx.spring_layout(self.G)  # 弹簧布局算法
        # self.pos = nx.circular_layout(self.graph, scale=6, center=None, dim=2)  # 环形布局算法

        # 状态和动作空间维度
        self.state_dim = self._get_oberservation().shape[0]
        self.action_dim = 3

    def _nodes_edges_resource_init(self):
        '''
            初始化RL训练的节点属性和边资源
        '''
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

    def _sfc_init(self):
        '''
            初始化sfc链的每一个vnf的资源
        '''
        self.sfcs = [SFC(self.G) for _ in range(self.sfc_num)]

    def sfc_path2sequence(self, sfc_path):
        '''
            将sfc路径元祖转化为路径序列
           (a,b), (b,c) -> a,b,c
        '''
        sequence = []
        for i, path in enumerate(sfc_path):
            if i == 0:
                sequence += path
            else:
                sequence += path[1:]
        return sequence

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
        node_overload, edge_overload = [], []
        for node in self.G.nodes:
            overload = False
            if (self.G.nodes[node]['cpu_used']/self.G.nodes[node]['cpu'] > self.overload_threshold or
                self.G.nodes[node]['storage_used']/self.G.nodes[node]['storage'] > self.overload_threshold or
                    self.G.nodes[node]['forward_used'] / self.G.nodes[node]['forward'] > self.overload_threshold):
                overload = True
            node_overload.append(overload)

        for edge in self.G.edges:
            overload = False
            if self.G.edges[edge]['bandwidth']/self.G.edges[edge]['bandwidth'] > self.overload_threshold:
                overload = True
            edge_overload.append(overload)

        return node_overload, edge_overload

    def _get_oberservation(self):
        '''
            获取当前状态：节点和边的使用资源、剩余资源、部署的sfc链路的vnf资源占用情况
        '''
        obs = []
        for node in self.G.nodes:
            obs.append(self.G.nodes[node]['cpu_used'])
            obs.append(self.G.nodes[node]['cpu_remain'])
            obs.append(self.G.nodes[node]['storage_used'])
            obs.append(self.G.nodes[node]['storage_remain'])
            obs.append(self.G.nodes[node]['forward_used'])
            obs.append(self.G.nodes[node]['forward_remain'])
        for edge in self.G.edges:
            obs.append(self.G.edges[edge]['bandwidth_used'])
            obs.append(self.G.edges[edge]['bandwidth_remain'])
            obs.append(self.G.edges[edge]['delay'])
        for sfc in self.sfcs:
            for vnf in sfc.chain:
                obs.append(vnf.cpu)
                obs.append(vnf.storage)
                obs.append(vnf.forward)
        return np.array(obs)

    def _calculate_baln(self, values, capacities, n):
        """ 计算负载方差 """
        ratios = values / capacities
        mean_ratio = np.sum(ratios) / n
        baln = np.sum((ratios - mean_ratio) ** 2) / n
        return baln

    def _reward(self):
        '''
            计算当前状态的奖励
        '''
        # 获取所有节点的资源
        cpu_remain = np.array([self.G.nodes[node]['cpu_remain'] for node in self.G.nodes])
        cpu_used = np.array([self.G.nodes[node]['cpu_used'] for node in self.G.nodes])
        cpu_total = np.array([self.G.nodes[node]['cpu'] for node in self.G.nodes])
        memory_remain = np.array([self.G.nodes[node]['storage_remain'] for node in self.G.nodes])
        memory_used = np.array([self.G.nodes[node]['storage_used'] for node in self.G.nodes])
        memory_total = np.array([self.G.nodes[node]['storage'] for node in self.G.nodes])
        forward_remain = np.array([self.G.nodes[node]['forward_remain'] for node in self.G.nodes])
        forward_used = np.array([self.G.nodes[node]['forward_used'] for node in self.G.nodes])
        forward_total = np.array([self.G.nodes[node]['forward'] for node in self.G.nodes])
        bandwidth_remain = np.array([self.G.edges[edge]['bandwidth_remain'] for edge in self.G.edges])
        bandwidth_used = np.array([self.G.edges[edge]['bandwidth_used'] for edge in self.G.edges])
        bandwidth_total = np.array([self.G.edges[edge]['bandwidth'] for edge in self.G.edges])

        # 计算方差（负载均衡指标）
        baln_cpu = self._calculate_baln(cpu_remain, cpu_total, len(self.G.nodes))
        baln_memory = self._calculate_baln(memory_remain, memory_total, len(self.G.nodes))
        baln_forward = self._calculate_baln(forward_remain, forward_total, len(self.G.nodes))
        baln_bandwidth = self._calculate_baln(bandwidth_remain, bandwidth_total, len(self.G.edges))

        # 计算总时延（处理时延 + 传输时延）
        T_total = 0
        sfc_paths_sequence = [self.sfc_path2sequence(sfc_path) for sfc_path in self.sfc_paths]
        for _, (sfc, sfc_path_sequence) in enumerate(zip(self.sfcs, sfc_paths_sequence)):
            T_transmission = [self.G.edges[(sfc_path_sequence[i], sfc_path_sequence[i+1])]['delay'] for i in range(len(sfc_path_sequence)-1)]
            T_process = []
            for nvf in sfc.chain:
                # todo 计算处理时延
                pass
            T_total += sum(T_process + T_transmission)

        # 计算总奖励
        reward = .5*(-baln_cpu - baln_memory-baln_forward - baln_bandwidth) + .5*T_total
        return reward

    def step(self, action):
        '''
            执行动作，部署sfc链路，更新网络资源，返回状态、奖励、是否结束
            action: 选择第几条sfc 第几个vnf 部署到哪个节点
        '''
        '''
        [[[25, 8], [8, 4], [4, 8, 9], [9, 27], [27, 16]]
        '''
        sfc = self.sfcs[action[0]]
        sfc_path = self.sfc_paths[action[0]]

        # 部署到同一个节点上，不允许部署
        if action[2] in [vnf.deployed_node for vnf in sfc.chain]:
            r = self._reward()-100
            return self._get_oberservation(), r,  True,  {'is_overload': True}

        # 迁出节点 计算迁出后的节点资源
        vnf_target_ = copy.deepcopy(sfc.chain[action[1]])
        vnf_target_.cpu *= -1
        vnf_target_.storage *= -1
        vnf_target_.forward *= -1

        # 对应节点资源恢复
        for vnf in sfc.chain:
            if vnf_target_.deployed_node == vnf.deployed_node:
                self.update_node_resources(vnf_target_.deployed_node, vnf_target_)
        # 对应边资源恢复
        for edge in sfc_path:
            if vnf_target_.deployed_node in [edge[0], edge[1]]:
                self.update_edge_resources(edge, vnf_target_)

        # 迁入节点 计算迁入后的节点资源

        # 查找sfc中与vnf_source_相邻的vnf
        pre_node, next_node = None, None
        for edge in sfc_path:
            if edge[-1] == vnf_target_.deployed_node:
                pre_node = edge[0]
            elif edge[0] == vnf_target_.deployed_node:
                next_node = edge[-1]

        # 迁入节点 计算迁入后的网络资源
        vnf_source_ = sfc.chain[action[1]]
        vnf_source_.deployed_node = action[2]
        self.update_node_resources(action[2], vnf_source_)
        if pre_node == None:
            # 该节点为第一个节点
            path = nx.shortest_path(self.G, source=action[2], target=next_node)
            sfc_path[0] = path
            self.update_edge_resources(path, vnf_source_)
        elif next_node == None:
            # 该节点为最后一个节点
            path = nx.shortest_path(self.G, source=pre_node, target=action[2])
            sfc_path[-1] = path
            self.update_edge_resources(path, vnf_source_)
        else:
            path_left = nx.shortest_path(self.G, source=pre_node, target=action[2])
            path_right = nx.shortest_path(self.G, source=action[2], target=next_node)

            for i, edge in enumerate(sfc_path):
                if edge[0] == pre_node and edge[-1] == vnf_target_.deployed_node:
                    sfc_path[i] = path_left
                elif edge[0] == vnf_target_.deployed_node and edge[-1] == next_node:
                    sfc_path[i] = path_right

            self.update_edge_resources(path_left, vnf_source_)
            self.update_edge_resources(path_right, vnf_source_)

        r = self._reward()
        is_overload = any(self.check_net_is_overloaded())
        if is_overload == False:
            r += 1000
        return self._get_oberservation(), r, False, {'is_overload': is_overload}

    def deploy_sfc(self, sfc):
        deployed_nodes = []
        deployed_path = []
        for vnf in sfc.chain:
            available_nodes = [node for node in self.G.nodes if node not in deployed_nodes]
            if len(deployed_nodes) == 0:
                # 如果这是第一个VNF，随机选择一个节点
                selected_node = np.random.choice(available_nodes)
                vnf.deployed_node = selected_node
                deployed_nodes.append(selected_node)
                # deployed_path.append([selected_node])
                self.update_node_resources(selected_node, vnf)  # 更新节点资源
            else:
                # 找到前一个部署的VNF节点，查找从前一个节点到当前可用节点的最短路径
                prev_node = deployed_nodes[-1]
                min_edge = float('inf')
                best_node = None
                best_path = None

                for node in available_nodes:
                    try:
                        path_min_edge = nx.shortest_path(self.G, source=prev_node, target=node)  # 找到最短距离（最少边数）的路径
                        edge_num = len(path_min_edge)-1
                        if edge_num < min_edge and self.cheak_edge_resources(path_min_edge, vnf):
                            min_edge = edge_num
                            best_node = node
                            best_path = path_min_edge
                    except nx.NetworkXNoPath:
                        continue

                if best_node is None or best_path is None:
                    return False, None

                # 部署VNF并更新资源
                vnf.deployed_node = best_node
                deployed_nodes.append(best_node)
                deployed_path.append(best_path)
                self.update_node_resources(best_node, vnf)
                self.update_edge_resources(best_path, vnf)
        return True, deployed_path

    def reset(self):
        '''
            重置网络资源，同时部署sfc，初始化状态为节点或者边过载
        '''
        reset_success = False
        while not reset_success:
            self.G.nodes, self.G.edges = copy.deepcopy(self.G_nodes_backup), copy.deepcopy(self.G_edges_backup)
            self.sfcs, self.sfc_paths = [], []

            for _ in range(self.sfc_num):
                sfc = SFC(self.G)
                succes, deployed_path = self.deploy_sfc(sfc)
                if not succes:
                    break
                self.sfcs.append(sfc)
                self.sfc_paths.append(deployed_path)

            if len(self.sfcs) < self.sfc_num:
                continue

            node_overload, edge_overload = self.check_net_is_overloaded()
            reset_success = True if any(node_overload + edge_overload) else False
        return self._get_oberservation()

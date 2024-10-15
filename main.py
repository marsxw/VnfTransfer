# %%
from collections import Counter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
# import demo
import copy
import random
import itertools
import csv


'''
在软件定义网络(Software Defined Network,SDN)
网络功能虚拟化(Network Function Virtualization, NFV)

服务功能链(Service Function Chain, SFC),
虚拟网络功能(Virtual Network Function,VNF)
L 链路
N VNF里面节点
C CPU需求
B 贷款需求
D 时延限制
'''


class PhysicalNetwork:
    def __init__(self, topology_file):
        self.graph = nx.read_gml(topology_file)
        self.node_resources = {}
        self.edge_resources = {}

    def display_topology(self):
        nx.draw(self.graph, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)
        plt.title("Physical Network Topology")
        plt.show()

    def generate_resources(self):
        for node in self.graph.nodes:
            cpu = np.random.uniform(100, 180)
            storage = np.random.uniform(100, 180)
            forward = np.random.uniform(100, 180)
            self.node_resources[node] = {'cpu': cpu, 'storage': storage, 'forward': forward}
        for edge in self.graph.edges:
            bandwidth = np.random.uniform(500, 600)
            delay = np.random.uniform(2, 5)
            self.graph.edges[edge]['bandwidth'] = bandwidth
            self.graph.edges[edge]['delay'] = delay
            self.edge_resources[edge] = {'bandwidth': bandwidth, 'delay': delay}
            self.edge_resources[edge] = {'delay': delay}
            # print(f"Edge: {edge}, Bandwidth: {bandwidth}, Delay: {delay}")

    def collect_node_vnf_info(self, deployed_nodes_data):
        node_vnf_info = {}
        for node, deployed_vnfs in deployed_nodes_data.items():
            for deployed_vnf in deployed_vnfs:
                sfc_index = deployed_vnf['SFC Index']
                vnf_index = deployed_vnf['VNF Index']
                vnf_type = deployed_vnf['VNF Type']
                cpu = deployed_vnf['CPU']
                storage = deployed_vnf['Storage']
                # forward=deployed_vnf['forward']
                if node not in node_vnf_info:
                    node_vnf_info[node] = []
                node_vnf_info[node].append(
                    # {'SFC Index': sfc_index, 'VNF Index': vnf_index, 'VNF Type': vnf_type, 'CPU': cpu,
                    #  'Storage': storage,'forward': forward})
                    {'SFC Index': sfc_index, 'VNF Index': vnf_index, 'VNF Type': vnf_type, 'CPU': cpu,
                     'Storage': storage})

        return node_vnf_info


class VNF:
    def __init__(self, vnf_type):
        self.type = vnf_type
        self.cpu = np.random.uniform(40, 50)
        self.storage = np.random.uniform(40, 50)
        self.forward = np.random.uniform(40, 50)
        self.deployed_node = None

    def get_cpu(self):
        return self.cpu

    def get_storage(self):
        return self.storage

    def get_forward(self):
        return self.forward


class SFC:
    def __init__(self, network):
        self.network = network
        self.vnf_types = ['Firewall', 'Load Balancer', 'Cache', 'IDS', 'router', 'switch']
        self.chain = []
        self.edges = []
        self.source = random.choice(list(network.graph.nodes))
        self.destination = random.choice(list(network.graph.nodes))
        while self.source == self.destination:
            self.destination = random.choice(list(network.graph.nodes))
        self.priority = random.randint(1, 3)

        # 生成 SFC 链
        self.generate_chain()

    # 生成 SFC 链
    def generate_chain(self):
        prev_vnf = None
        for vnf_type in self.vnf_types:
            vnf = VNF(vnf_type)
            self.chain.append(vnf)
    # 打印 SFC 链资源信息

    def print_chain_resources(self):
        for idx, vnf in enumerate(self.chain):
            print(f"VNF {idx + 1} ({vnf.type}): CPU={vnf.cpu:.2f}, Storage={vnf.storage:.2f},forward={vnf.forward:.2f}")


def find_unused_nodes(network, deployed_nodes):
    # print(list(itertools.chain.from_iterable(deployed_nodes.values())))
    unused_nodes = [node for node in network.node_resources if node not in list(itertools.chain.from_iterable(deployed_nodes.values()))]
    # print(len(unused_nodes), '=============================')
    return unused_nodes


def deploy_vnf(vnf, network, deployed_nodes):
    available_nodes = find_unused_nodes(network, deployed_nodes)
    if not available_nodes:
        return False, None

    best_node = None
    max_available_resources = 0  # 储存已部署的 VNF 占用的最大资源量
    for node in available_nodes:
        cpu_remaining = network.node_resources[node]['cpu']
        storage_remaining = network.node_resources[node]['storage']
        # forward_remaining = network.node_resources[node]['forward']
        if (cpu_remaining >= vnf.cpu and storage_remaining >= vnf.storage):
            current_available_resources = cpu_remaining + storage_remaining
            if current_available_resources > max_available_resources:
                best_node = node
                max_available_resources = current_available_resources

    if best_node is not None:
        vnf.deployed_node = best_node
        network.node_resources[best_node]['cpu'] = network.node_resources[best_node]['cpu']-vnf.cpu
        network.node_resources[best_node]['storage'] = network.node_resources[best_node]['storage'] - vnf.storage
        # network.node_resources[best_node]['forward'] -= vnf.forward
        print(f"VNF {vnf.type} deployed on node: {best_node}")
        deployed_nodes.setdefault(best_node, []).append(vnf)  # 如果该节点不存在对应的键，则插入空列表
        return True, best_node
    else:
        print(f"Failed to deploy VNF of type {vnf.type}")
        return False, None


def find_shortest_path_edges(graph, src_node, dst_node):
    shortest_paths = list(nx.all_shortest_paths(graph, source=src_node, target=dst_node, weight='delay'))
    all_shortest_path_edges = []
    total_shortest_path_delays = []

    for path in shortest_paths:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        total_delay = sum(graph[path[i]][path[i + 1]]['delay'] for i in range(len(path) - 1))  # 直接访问边的 delay 属性
        all_shortest_path_edges.append(path_edges)
        total_shortest_path_delays.append(total_delay)

    return all_shortest_path_edges, total_shortest_path_delays


def greedy_placement(sfc_list, network, zong, remaining_cpu, remaining_storage, node_used):
    deployed_nodes = {}  # Track deployed nodes for each VNF type
    total_delay = 0  # Total delay
    remaining_cpu = copy.deepcopy(remaining_cpu)  # 存储每个节点的剩余 CPU
    remaining_storage = copy.deepcopy(remaining_storage)  # 存储每个节点的剩余存储空间
    # remaining_forward = copy.deepcopy(remaining_forward)  # 存储每个节点的��余带宽
    used_ziyuan = copy.deepcopy(node_used)
    zong1 = copy.deepcopy(zong)

    prev_deployed_nodes = None  # 前一个部署的节点

    # 遍历部署每一个 SFC
    for sfc in sfc_list:
        deployed_nodes = {}  # 重置部署节点记录
        total_delay = 0  # 重置延迟时间

        # 部署 VNFs
        kkk = []
        for idx, vnf in enumerate(sfc.chain):
            success, deployed_node = deploy_vnf(vnf, network, deployed_nodes)  # 用到上面的函数进行部署
            if not success:  # 如果不成功会怎么样
                print(f"Failed to deploy VNF {vnf.type}. No available nodes with sufficient resources.")
                return False, used_ziyuan

            # 使用上一条SFC的部署节点记录
            if prev_deployed_nodes:
                deployed_nodes = prev_deployed_nodes

            kkk.append(deployed_node)
            if idx > 0:  # 避免第一个节点部署但是第一个节点之前是没有节点部署的从而找不到边的情况
                src_node = sfc.chain[idx - 1].deployed_node  # 这是取sfc第一个vnf所部署的节点为源节点
                dst_node = vnf.deployed_node  # 这是取sfc链上的第二个vnf所部署的节点为目的节点

                all_shortest_path_edges, all_total_delays = find_shortest_path_edges(network.graph, src_node, dst_node)  # 使用最短路径的函数
                if not all_shortest_path_edges:  # 检查最短路径是否为空
                    print(f"No path found between nodes {src_node} and {dst_node}")
                    return False

                shortest_path_index = all_total_delays.index(min(all_total_delays))
                shortest_path_edges = all_shortest_path_edges[shortest_path_index]
                shortest_path_delay = all_total_delays[shortest_path_index]
                total_delay += shortest_path_delay

                prev_vnf = sfc.chain[idx - 1]
                prev_vnf_bandwidth = prev_vnf.get_forward()

                print(f"Shortest path edges from {src_node} to {dst_node}:")
                for edge in shortest_path_edges:
                    edge_delay = network.graph[edge[0]][edge[1]]['delay']
                    edge_bandwidth = network.graph[edge[0]][edge[1]].get('bandwidth', 0)
                    vnf_bandwidth = vnf.get_forward()  # Get the required bandwidth for the VNF

                    # Update the edge bandwidth
                    network.graph[edge[0]][edge[1]]['bandwidth'] = max(0, edge_bandwidth - vnf_bandwidth)

                    print(
                        f"Edge {edge}: Delay={edge_delay}, Updated Bandwidth={network.graph[edge[0]][edge[1]]['bandwidth']}")

                print(f"Total delay for the path from {src_node} to {dst_node}: {shortest_path_delay}")
                print(f"Total delay for the SFC: {total_delay}")

        #############
        # 创建一个空字典来存储计数
        count_dict = {}

        # 使用 for 循环统计每个元素的次数
        for item in kkk:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1

        # 打印结果
        for k, v in count_dict.items():
            if v > 1:
                print('---------------------------------------------------------', count_dict)
                import sys
                sys.exit()

        # 打印节点资源使用情况

        print("Final resource usage:")
        for node in zong1:
            total_cpu = zong1[node]["cpu"]  # 总cpu资源
            total_storage = zong1[node]["storage"]  # 总存储资源
            # total_forward = zong1[node]["forward"]# 总带宽
            # used_cpu = sum(v.cpu for vnf_list in deployed_nodes.values() for v in vnf_list if v.deployed_node == node)
            # used_storage = sum(
            #     v.storage for vnf_list in deployed_nodes.values() for v in vnf_list if v.deployed_node == node)
            # 计算所部署节点上的所有 VNF 的 CPU 和存储资源之和
            used_cpu = sum(
                v.cpu for vnf_list in deployed_nodes.values() for v in vnf_list if vnf_list[-1].deployed_node == node)
            # used_forward = sum(v.forward for vnf_list in deployed_nodes.values() for v in vnf_list if
            #                    vnf_list[-1].deployed_node == node)#
            used_storage = sum(v.storage for vnf_list in deployed_nodes.values() for v in vnf_list if
                               vnf_list[-1].deployed_node == node)
            if node in deployed_nodes:
                used_ziyuan[node]['cpu'] = used_ziyuan[node]['cpu'] + used_cpu
                used_ziyuan[node]['storage'] = used_ziyuan[node]['storage'] + used_storage
                # used_ziyuan[node]['forward'] = used_ziyuan[node]['forward'] + used_forward
                remaining_cpu[node] = total_cpu - used_ziyuan[node]['cpu']  # 剩余cpu资源
                remaining_storage[node] = total_storage - used_ziyuan[node]['storage']  # 剩余存储资源
                # remaining_forward[node] = total_forward - used_ziyuan[node]['forward']

            print(f"Node {node} - Total CPU: {total_cpu}, Total Storage: {total_storage}, Used CPU: {used_ziyuan[node]['cpu']}, Used Storage: {
                used_ziyuan[node]['storage']},Remaining CPU: {total_cpu-used_ziyuan[node]['cpu']}, Remaining Storage: {total_storage-used_ziyuan[node]['storage']}")

        # 记录当前SFC的部署节点记录
        prev_deployed_nodes = deployed_nodes

    # 将打印的内容保存至csv文件
    output_file = r'./node_resources.csv'

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入表头
        writer.writerow(
            ['Node', 'Total CPU', 'Total Storage', 'Used CPU', 'Used Storage', 'Remaining CPU', 'Remaining Storage'])

        # 写入每个节点的资源使用情况
        for node in zong1:
            total_cpu = zong1[node]["cpu"]
            total_storage = zong1[node]["storage"]
            # total_forward = zong1[node]["forward"]
            used_cpu = used_ziyuan[node]['cpu']
            used_storage = used_ziyuan[node]['storage']
            # used_forward = used_ziyuan[node]['forward']
            remaining_cpu = total_cpu - used_ziyuan[node]['cpu']
            remaining_storage = total_storage - used_ziyuan[node]['storage']
            # remaining_forward = total_forward - used_ziyuan[node]['forward']

            writer.writerow([node, total_cpu, total_storage, used_cpu, used_storage, remaining_cpu, remaining_storage])

    print(f"The resource usage information has been saved to {output_file}")

    # Write the link resource information to a CSV file
    output_link_file = r'./link-resource.csv'

    with open(output_link_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Link', 'Delay', 'Total Bandwidth', 'Updated Bandwidth'])

        for edge in network.graph.edges():
            src, dst = edge
            edge_delay = network.graph[src][dst]['delay']
            total_bandwidth = network.graph[src][dst].get('bandwidth', 0)
            updated_bandwidth = total_bandwidth - sum(
                vnf.get_forward() for vnf in sfc.chain if vnf.deployed_node in [src, dst])

            writer.writerow([f"({src}, {dst})", edge_delay, total_bandwidth, updated_bandwidth])

    print(f"The link resource information has been saved to {output_link_file}")

    return True, used_ziyuan


def collect_deployed_node_info(sfc_list, deployed_nodes):
    all_deployed_nodes_info = {}  # Dictionary to store deployed nodes information

    for sfc_idx, sfc in enumerate(sfc_list):
        for vnf in sfc.chain:
            deployed_node = vnf.deployed_node  # Get the deployed node for the VNF
            vnf_info = {
                "sfc_id": sfc_idx + 1,
                "vnf_type": vnf.type,
                "cpu": vnf.cpu,
                "storage": vnf.storage,
                "forward": vnf.forward
            }
            if deployed_node not in all_deployed_nodes_info:
                all_deployed_nodes_info[deployed_node] = [vnf_info]
            else:
                all_deployed_nodes_info[deployed_node].append(vnf_info)

    return all_deployed_nodes_info


topology_file = "./Chinanet.gml"
network = PhysicalNetwork(topology_file)
network.generate_resources()

num_sfc = 200
deployed_nodes = {}
zong = copy.deepcopy(network.node_resources)
node_resources_used = {node: {'cpu': 0, 'storage': 0} for node in network.node_resources}
remaining_cpu = {}
remaining_storage = {}
# remaining_forward={}

for node in zong:
    remaining_cpu[node] = zong[node]['cpu']
    remaining_storage[node] = zong[node]['storage']
    # remaining_forward[node] = zong[node]['forward']

with open('deployment_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['SFC ID', 'Priority', 'Node', 'VNF Type', 'CPU', 'Storage', 'forward']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(num_sfc):
        sfc = SFC(network)
        sfc_id = i + 1
        print(f"\nSFC {i + 1}:")
        sfc.print_chain_resources()

        success, ziyuan = greedy_placement([sfc], network, zong, remaining_cpu, remaining_storage,
                                           node_resources_used)

        if success:
            print("SFC deployed successfully")
            node_resources_used = copy.deepcopy(ziyuan)
            deployed_nodes_info = collect_deployed_node_info([sfc], deployed_nodes)

            print("\nDeployed Nodes Information:")
            for node, sfc_vnfs in deployed_nodes_info.items():
                for vnf_info in sfc_vnfs:
                    vnf_info['Priority'] = sfc.priority
                    row_data = {
                        'SFC ID': sfc_id,
                        'Priority': vnf_info['Priority'],
                        'Node': node,
                        'VNF Type': vnf_info['vnf_type'],
                        'CPU': vnf_info['cpu'],
                        'Storage': vnf_info['storage'],
                        'forward': vnf_info['forward']  # 新添加的 bandwidth 列
                    }

                    filtered_row_data = {key: row_data[key] for key in fieldnames}
                    writer.writerow(filtered_row_data)
                    print(f"  SFC ID: {sfc_id}, Node: {node}, VNF Type: {vnf_info['vnf_type']}, CPU: {
                        vnf_info['cpu']}, Storage: {vnf_info['storage']},forward:{vnf_info['forward']}")
        else:
            print("Failed to deploy the SFC")

 
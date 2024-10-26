# %%
# 导入必要的库
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个空的无向图
G = nx.Graph()

# 添加节点
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)

# 添加边（连接节点）
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)
G.add_edge(3, 4)

# 为节点和边添加属性
G.nodes[1]['name'] = "Node 1"
G.nodes[2]['name'] = "Node 2"
G.edges[1, 2]['weight'] = 5
G.edges[3, 4]['weight'] = 2

# 绘制网络图
pos = nx.spring_layout(G)  # 使用 spring 布局算法
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_weight='bold')

# 显示节点标签
labels = nx.get_edge_attributes(G, 'weight')  # 获取边的属性“weight”
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)  # 在图上显示权重

plt.title("Basic Network Graph using NetworkX")
plt.show()


# %%

# 创建一个有向图，并添加边和延迟
G = nx.DiGraph()
G.add_edge('A', 'B', delay=2)
G.add_edge('A', 'C', delay=5)
G.add_edge('B', 'C', delay=1)
G.add_edge('B', 'D', delay=4)
G.add_edge('C', 'D', delay=1)

# 设置布局
pos = nx.spring_layout(G)

# 绘制图节点和边
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold')

# 显示边的权重(延迟)
edge_labels = nx.get_edge_attributes(G, 'delay')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Network Graph with Delays")
plt.show()


def find_shortest_path_edges(graph, src_node, dst_node):
    shortest_paths = list(nx.all_shortest_paths(graph, source=src_node, target=dst_node, weight='delay'))
    print(shortest_paths)
    all_shortest_path_edges = []
    total_shortest_path_delays = []

    for path in shortest_paths:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        total_delay = sum(graph[path[i]][path[i + 1]]['delay'] for i in range(len(path) - 1))  # 直接访问边的 delay 属性
        all_shortest_path_edges.append(path_edges)
        total_shortest_path_delays.append(total_delay)

    return all_shortest_path_edges, total_shortest_path_delays


# 调用 find_shortest_path_edges 函数，查找从 A 到 D 的最短路径
all_edges, all_delays = find_shortest_path_edges(G, 'A', 'D')

# 打印最短路径的边和延迟
print("Shortest path edges:", all_edges)
print("Total delays:", all_delays)

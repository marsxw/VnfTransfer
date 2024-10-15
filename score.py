import pandas as pd

# 读取 CSV 文件
node_resources_df = pd.read_csv('node_resources.csv')
deployment_results_df = pd.read_csv('deployment_results.csv')

# 定义超标阈值
cpu_threshold_percentage = 0.85
storage_threshold_percentage = 0.85

# 查找超标节点
exceed_nodes = []
for index, row in node_resources_df.iterrows():
    total_cpu = row['Total CPU']
    total_storage = row['Total Storage']
    used_cpu = row['Used CPU']
    used_storage = row['Used Storage']

    # 计算85%的总值
    cpu_threshold = cpu_threshold_percentage * total_cpu
    storage_threshold = storage_threshold_percentage * total_storage

    # 检查是否超过85%
    cpu_exceeds = used_cpu > cpu_threshold
    storage_exceeds = used_storage > storage_threshold

    if cpu_exceeds or storage_exceeds:
        exceed_nodes.append(row['Node'])

# 计算每个超标节点的sfc的平均优先级、CPU使用百分比和内存使用百分比的方差
result_data = []
vnf_data = []
for node in exceed_nodes:
    # 过滤出当前节点的部署信息
    node_deployment_df = deployment_results_df[deployment_results_df['Node'] == node]

    if not node_deployment_df.empty:
        sfc_priorities = node_deployment_df['Priority'].tolist()
        used_cpu = node_resources_df[node_resources_df['Node'] == node]['Used CPU'].values[0]
        total_cpu = node_resources_df[node_resources_df['Node'] == node]['Total CPU'].values[0]
        used_storage = node_resources_df[node_resources_df['Node'] == node]['Used Storage'].values[0]
        total_storage = node_resources_df[node_resources_df['Node'] == node]['Total Storage'].values[0]
        cpu_percentage = used_cpu / total_cpu
        storage_percentage = used_storage / total_storage

        average_priority = sum(sfc_priorities) / len(sfc_priorities)
        cpu_percentage_variance = cpu_percentage
        storage_percentage_variance = storage_percentage

        # 根据模糊满意度计算节点评分
        score = (average_priority * 0.4) + (1 - cpu_percentage_variance) * 0.3 + (1 - storage_percentage_variance) * 0.3

        result_data.append({'Node': node, 'Average Priority': average_priority, 'CPU Percentage Variance': cpu_percentage_variance, 'Storage Percentage Variance': storage_percentage_variance, 'Score': score})
        # 获取每个VNF的信息
        for index, row in node_deployment_df.iterrows():
            vnf_data.append({'Node': node, 'SFC ID': row['SFC ID'], 'VNF Type': row['VNF Type'] ,'VNF Priority': row['Priority'], 'VNF CPU Percentage': row['CPU'] / total_cpu, 'VNF Storage Percentage': row['Storage'] / total_storage})


# 将结果转换为 DataFrame 并按评分从高到低进行排序
result_df = pd.DataFrame(result_data)
result_df = result_df.sort_values(by='Score', ascending=False)

# 将VNF信息转换为 DataFrame
vnf_df = pd.DataFrame(vnf_data)

# 模糊满意度评价函数
def calculate_satisfaction_score(priority, cpu_percentage, storage_percentage):

    score = (priority * 0.4) + (cpu_percentage * 0.3) + (storage_percentage * 0.3)
    return score

# 对vnf_data中的每一行进行模糊满意度评价并计算得分
for index, row in vnf_df.iterrows():
    priority = row['VNF Priority']
    cpu_percentage = row['VNF CPU Percentage']
    storage_percentage = row['VNF Storage Percentage']
    score = calculate_satisfaction_score(priority, cpu_percentage, storage_percentage)
    vnf_df.at[index, 'Satisfaction Score'] = score

# 将结果输出到 CSV 文件
result_df.to_csv('result_output.csv', index=False)
vnf_df.to_csv('vnf_output.csv', index=False)

# 读取 result_output.csv 中的节点顺序
sorted_nodes = result_df['Node'].tolist()

# 根据节点顺序重新排序 vnf_df
vnf_df['Node'] = pd.Categorical(vnf_df['Node'], categories=sorted_nodes, ordered=True)
vnf_df = vnf_df.sort_values(by=['Node', 'Satisfaction Score'], ascending=[True, False])

# 将结果输出到 CSV 文件
vnf_df.to_csv('migrate_vnf.csv', index=False)
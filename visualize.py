import numpy as np
import matplotlib.pyplot as plt

# 假设有一系列故障检测的结果，每个结果包括FAR, FDR, DD和故障类型标签
results = [
    {'fault_type': 'Type 1', 'FAR': 0.05, 'FDR': 0.95, 'DD': 0.1, 'n_components': 5, 'threshold': 0.95},
    {'fault_type': 'Type 2', 'FAR': 0.10, 'FDR': 0.90, 'DD': 0.15, 'n_components': 10, 'threshold': 0.95},
    # 添加更多故障类型和设置
]

# 数据处理以适应图表
fault_types = [res['fault_type'] for res in results]
FARs = [res['FAR'] for res in results]
FDRs = [res['FDR'] for res in results]
DDs = [res['DD'] for res in results]
n_components_set = [res['n_components'] for res in results]


def plot_performance_metrics(results):
    fault_types = [f"{res['fault_type']} ({res['n_components']})" for res in results]
    FARs = [res['FAR'] for res in results]
    FDRs = [res['FDR'] for res in results]
    DDs = [res['DD'] for res in results]

    fig, ax = plt.subplots(3, 1, figsize=(14, 18))

    ax[0].bar(fault_types, FARs, color='blue', label='FAR')
    ax[0].set_title('False Alarm Rate (FAR) by Fault Type and n_components')
    ax[0].set_ylabel('Rate')
    ax[0].set_xticklabels(fault_types, rotation=45, ha='right')
    ax[0].legend()

    ax[1].bar(fault_types, FDRs, color='green', label='FDR')
    ax[1].set_title('Fault Detection Rate (FDR) by Fault Type and n_components')
    ax[1].set_ylabel('Rate')
    ax[1].set_xticklabels(fault_types, rotation=45, ha='right')
    ax[1].legend()

    ax[2].bar(fault_types, DDs, color='red', label='DD')
    ax[2].set_title('Detection Delay (DD) by Fault Type and n_components')
    ax[2].set_ylabel('Delay')
    ax[2].set_xticklabels(fault_types, rotation=45, ha='right')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

plot_performance_metrics(results)


#
# # 箱形图展示不同n_components设置下的误差分布
# reconstruction_errors = [np.random.normal(loc=0, scale=1, size=100) for _ in results]  # 示例数据
# fig, ax = plt.subplots()
# ax.boxplot(reconstruction_errors, labels=n_components_set)
# ax.set_title('Reconstruction Error Distribution by n_components')
# ax.set_xlabel('n_components')
# ax.set_ylabel('Reconstruction Error')
# plt.show()
#
# # 折线图展示不同n_components下的FDR变化
# fig, ax = plt.subplots()
# ax.plot(n_components_set, FDRs, marker='o', linestyle='-', color='b')
# ax.set_title('FDR by n_components')
# ax.set_xlabel('n_components')
# ax.set_ylabel('FDR')
# plt.show()

import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from visualize import *


cca_results = []

# 加载数据
data = loadmat('TE_data.mat')

# 数据预处理
scaler = StandardScaler()

# 正常操作数据 (训练)
X_train_normal = data['d00'].reshape(-1, 52)
X_train_normal = X_train_normal[:480, :]  # 截断

# 初始化存储结果
results = []

# 循环处理每种故障类型
for i in range(1, 22):
    fault_type = f'd{i:02}'

    X_train_fault = data[fault_type].reshape(-1, 52)
    X = np.vstack([X_train_normal, X_train_fault])  # 训练集中的 正常+异常 合在一起
    X_scaled = scaler.fit_transform(X)  # 合起来以后再标准化  此方法用于在训练集上标准化
    y_train = np.concatenate([np.zeros(480), np.ones(480)])


    # 将原始数组重塑为 960 x 1 的二维列向量
    column_vector = y_train.reshape(-1, 1)
    # 使用广播将列向量扩展为 960 x 52 的二维数组
    # np.tile 函数将列向量复制 52 次，形成 960 x 52 的数组
    y_train = np.tile(column_vector, (1, 52))


    # 训练CCA模型用于故障数据
    cca = CCA(n_components=1)
    cca.fit(X_scaled, y_train)

    # 故障测试数据
    X_test_fault = data[f'{fault_type}_te'].reshape(-1, 52)
    y_test_fault = np.concatenate([np.zeros(160), np.ones(800)])  # 前160正常，后800故障


    # 将原始数组重塑为 960 x 1 的二维列向量
    column_vector = y_test_fault.reshape(-1, 1)
    # 使用广播将列向量扩展为 960 x 52 的二维数组
    # np.tile 函数将列向量复制 52 次，形成 960 x 52 的数组
    y_test_fault = np.tile(column_vector, (1, 52))


    # 标准化测试数据
    X_test_fault_scaled = scaler.transform(X_test_fault)  # 此方法用于在测试集上标准化

    # 使用CCA模型进行故障检测
    X_c, y_pred_fault = cca.transform(X_test_fault_scaled, y_test_fault)
    # 使用CCA得分判断故障
    threshold = np.percentile(X_c, 20)  # 设定阈值
    # threshold = 0
    print("thre = ", threshold)
    # y_pred_fault_label = y_pred_fault[:, 0] > 0.1
    y_pred_fault_label = X_c > threshold

    # t2, threshold = cca(X_scaled, y_train, X_test_fault_scaled, y_test_fault)
    # y_pred_fault_label = (t2 > threshold).astype(int)
    # y_pred_fault = t2

    # 计算性能指标
    TP = np.sum((y_pred_fault_label == 1) & (y_test_fault == 1))
    FP = np.sum((y_pred_fault_label == 1) & (y_test_fault == 0))
    TN = np.sum((y_pred_fault_label == 0) & (y_test_fault == 0))
    FN = np.sum((y_pred_fault_label == 0) & (y_test_fault == 1))

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FDR = TP / (TP + FN) if (TP + FN) > 0 else 0
    DD = calculate_detection_delay(y_pred_fault_label)

    cca_results.append({'fault_type': fault_type, 'FAR': FAR, 'FDR': FDR, 'DD': DD})

    print(f"{fault_type}: FAR: {FAR}, FDR: {FDR}, DD: {DD}\n")

    if (not i % 4) and i < 20:
        fig_num = int(220 + i / 4)
        # plt.figure(fig_num, figsize=(15, 6))
        plt.subplot(fig_num)
        plt.plot(X_c, label='CCA Predicted')
        plt.axhline(y=threshold, color='red', label='Threshold', linestyle='--')
        plt.title(f'CCA Prediction Scores for Fault Type {fault_type}', fontsize=12)
        # plt.xlabel('Sample Index')
        # plt.ylabel('Predicted Score')
        plt.legend()
        # plt.plot(y_pred_fault, label='X_c')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)



cca_svm_store_res('cca', cca_results, 0.4)

plt.show()

"""

# 绘制结果
fault_types = [res['fault_type'] for res in results]
FARs = [res['FAR'] for res in results]
FDRs = [res['FDR'] for res in results]
DDs = [res['DD'] for res in results]

plt.figure(figsize=(14, 6))
plt.subplot(131)
plt.bar(fault_types, FARs, color='red')
plt.title('False Alarm Rate (FAR)')
plt.xticks(rotation=45)

plt.subplot(132)
plt.bar(fault_types, FDRs, color='green')
plt.title('Fault Detection Rate (FDR)')
plt.xticks(rotation=45)

plt.subplot(133)
plt.bar(fault_types, DDs, color='blue')
plt.title('Detection Delay (DD)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
"""



"""
import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# 加载数据
data = loadmat('TE_data.mat')

# 数据预处理
scaler = StandardScaler()

# 正常操作数据 (训练)
X_train_normal = data['d00'].reshape(-1, 52)
X_train_normal_scaled = scaler.fit_transform(X_train_normal)

# 训练CCA模型
cca = CCA(n_components=1)

results = []

# 循环处理每种故障类型
for i in range(1, 22):
    fault_type = f'd{i:02}'

    # 故障训练数据
    X_train_fault = data[fault_type].reshape(-1, 52)
    X_train_fault_scaled = scaler.transform(X_train_fault)

    # 创建Y矩阵（二进制标签）
    y_train = np.zeros((X_train_normal.shape[0] + X_train_fault.shape[0], 1))
    y_train[X_train_normal.shape[0]:, 0] = 1  # 标记故障数据

    # 合并训练数据
    X_train = np.vstack([X_train_normal_scaled, X_train_fault_scaled])

    # 训练CCA
    cca.fit(X_train, y_train)

    # 故障测试数据
    X_test_fault = data[f'{fault_type}_te'].reshape(-1, 52)
    X_test_fault_scaled = scaler.transform(X_test_fault)

    # 应用CCA模型
    X_c, Y_c = cca.transform(X_test_fault_scaled, np.ones((X_test_fault.shape[0], 1)))

    # 使用CCA得分判断故障
    fault_detection = Y_c[:, 0] > 0.1  # 设定阈值

    # 性能评估（示例）
    TP = np.sum(fault_detection[160:])
    FP = np.sum(fault_detection[:160])
    TN = 160 - FP
    FN = 800 - TP

    FAR = FP / (FP + TN)
    FDR = TP / (TP + FN)

    results.append({'fault_type': fault_type, 'FAR': FAR, 'FDR': FDR})

    print(f"{fault_type}: FAR: {FAR}, FDR: {FDR}")

# 可根据需要添加可视化部分或进一步的分析

"""
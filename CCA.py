import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据
data = loadmat('TE_data.mat')

# 数据预处理
scaler = StandardScaler()

# 正常操作数据 (训练)
X_train_normal = data['d00'].reshape(-1, 52)

# 初始化存储结果
results = []

# 循环处理每种故障类型
for i in range(1, 22):
    fault_type = f'd{i:02}'

    X_train_fault = data[fault_type].reshape(-1, 52)
    X = np.vstack([X_train_normal, X_train_fault])  # 训练集中的 正常+异常 合在一起
    y_train = np.concatenate([np.zeros(500), np.ones(480)])
    X_scaled = scaler.fit_transform(X)  # 合起来以后再标准化  此方法用于在训练集上标准化

    # 训练CCA模型用于故障数据
    cca = CCA(n_components=1)
    cca.fit(X_scaled, y_train)

    # 故障测试数据
    X_test_fault = data[f'{fault_type}_te'].reshape(-1, 52)
    y_test_fault = np.concatenate([np.zeros(160), np.ones(800)])  # 前160正常，后800故障

    # 标准化测试数据
    X_test_fault_scaled = scaler.transform(X_test_fault)  # 此方法用于在测试集上标准化

    # 使用PLS模型进行故障检测
    X_c, y_pred_fault = cca.transform(X_test_fault_scaled, y_test_fault)

    # 使用CCA得分判断故障
    y_pred_fault_label = y_pred_fault[:, 0] > 0.1  # 设定阈值

    # 计算性能指标
    TP = np.sum((y_pred_fault_label == 1) & (y_test_fault == 1))
    FP = np.sum((y_pred_fault_label == 1) & (y_test_fault == 0))
    TN = np.sum((y_pred_fault_label == 0) & (y_test_fault == 0))
    FN = np.sum((y_pred_fault_label == 0) & (y_test_fault == 1))

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FDR = TP / (TP + FN) if (TP + FN) > 0 else 0
    DD = (TP + FN) / len(y_test_fault)  # 检测延迟简化为故障检测占比

    results.append({'fault_type': fault_type, 'FAR': FAR, 'FDR': FDR, 'DD': DD})

    print(f"{fault_type}: FAR: {FAR}, FDR: {FDR}, DD: {DD}\n")

    if not not(FDR>0.7 and FAR<0.5):
        plt.figure(i, figsize=(15, 6))
        plt.subplot(121)
        plt.plot(y_pred_fault, label='CCA Predicted')
        plt.axhline(y=0.1, color='red', label='Threshold', linestyle='--')
        plt.title(f'PLS Prediction Scores for Fault Type {fault_type}')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Score')
        plt.legend()
        plt.subplot(122)
        plt.plot(X_c, label='X_c')

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
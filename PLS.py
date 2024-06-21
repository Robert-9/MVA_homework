import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from visualize import *


pls_results = []

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

    # 训练PLS模型用于故障数据
    pls = PLSRegression(n_components=10)
    pls.fit(X_scaled, y_train)

    # 故障测试数据
    X_test_fault = data[f'{fault_type}_te'].reshape(-1, 52)
    y_test_fault = np.concatenate([np.zeros(160), np.ones(800)])  # 前160正常，后800故障

    # 标准化测试数据
    X_test_fault_scaled = scaler.transform(X_test_fault)  # 此方法用于在测试集上标准化

    # 使用PLS模型进行故障检测
    y_pred_fault = pls.predict(X_test_fault_scaled)
    y_pred_fault_label = (y_pred_fault[:, 0] > 0.4).astype(int)  # 假设阈值为0.4

    # 计算性能指标
    TP = np.sum((y_pred_fault_label == 1) & (y_test_fault == 1))
    FP = np.sum((y_pred_fault_label == 1) & (y_test_fault == 0))
    TN = np.sum((y_pred_fault_label == 0) & (y_test_fault == 0))
    FN = np.sum((y_pred_fault_label == 0) & (y_test_fault == 1))

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FDR = TP / (TP + FN) if (TP + FN) > 0 else 0
    # DD = (TP + FN) / len(y_test_fault)  # 检测延迟简化为故障检测占比
    DD = calculate_detection_delay(y_pred_fault_label)

    pls_results.append({
        'fault_type': fault_type,
        'threshold': 0.4,
        'FAR': FAR,
        'FDR': FDR,
        'DD': DD
    })

    print(f"{fault_type}: FAR: {FAR}, FDR: {FDR}, DD: {DD}\n")

    if (not i % 4) and i < 20:
        # plt.figure(i, figsize=(15, 6))
        fig_num = int(220 + i / 4)
        plt.subplot(fig_num)
        plt.plot(y_pred_fault, label='PLS Predicted Scores')
        plt.axhline(y=0.4, color='red', label='Threshold', linestyle='--')
        plt.title(f'PLS Prediction Scores for Fault Type {fault_type}', fontsize=12)
        # plt.xlabel('Sample Index')
        # plt.ylabel('Predicted Score')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()


pca_pls_store_res('pls', pls_results, 10, 0.4)

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
"""
使用PCA进行故障检测
    包括故障检测、结果的绘图、保存结果数据至json文件

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import json


def pca_store_res(results, n_components, threshold_percent):
    """
        构建最终的结果字典
    """
    final_results = {
        'n_components': n_components,
        'threshold_percent': threshold_percent,
        'faults': {}
    }
    for res in results:
        final_results['faults'][res['fault_type']] = {
            'threshold': res['threshold'],
            'FAR': res['FAR'],
            'FDR': res['FDR'],
            'DD': res['DD']
        }
    # 存储结果到JSON文件
    with open(f'pcaResults_nCom{n_components}_threPer{threshold_percent}.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    return final_results


pca_results = []
nComp = 0.7
thre_per = 17

# 加载数据
data = loadmat('D:\大学的东西\硕\课程\过程监测与故障诊断\Final Task\TE_Data.mat')

X_train_normal = data['d00'].reshape(-1, 52)  # 正常训练数据，每个样本包含52个变量
X_test_normal = data['d00_te'].reshape(-1, 52)  # 正常测试数据，每个样本包含52个变量

# 数据预处理
scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
# X_test_normal_scaled = scaler.transform(X_test_normal)


# 初始化PCA并用正常数据训练
def pca_init(n_com, inputs):
    pca = PCA(n_components=n_com)
    pca.fit(inputs)
    return pca


# X_test_normal_pca = pca.transform(X_test_normal_scaled)
# X_test_normal_projected = pca.inverse_transform(X_test_normal_pca)


# 整合并标准化测试数据
def preprocess_test_data(scaler, pca, X_test_fault):
    X_test_fault_scaled = scaler.transform(X_test_fault)
    X_test_fault_pca = pca.transform(X_test_fault_scaled)
    X_test_fault_projected = pca.inverse_transform(X_test_fault_pca)
    return X_test_fault_scaled, X_test_fault_projected

# 计算 n_components
# variance_ratios = pca.explained_variance_ratio_  # 获取方差比
# cumulative_variance_ratios = np.cumsum(variance_ratios)  # 计算累计方差比
# components_needed = np.sum(cumulative_variance_ratios <= 0.95) + 1  # 找到累计方差比达到或超过95%的主成分数量
# print(f"需要的主成分数量: {components_needed}")


pca = pca_init(nComp, X_train_normal_scaled)
# 针对每种故障单独进行故障检测和性能评估
for i in range(1, 22):
    fault_type = f'd{i:02}'
    # print(f"\nFault Type: {fault_label}")

    X_train_fault = data[fault_type].reshape(-1, 52)  # 故障训练数据
    X_test_fault = data[f'{fault_type}_te'].reshape(-1, 52)  # 故障测试数据

    # 整合测试数据并标准化
    X_test_fault_scaled = scaler.transform(X_test_fault)

    # PCA
    X_test_fault_pca = pca.transform(X_test_fault_scaled)
    X_test_fault_projected = pca.inverse_transform(X_test_fault_pca)

    # 计算重构误差
    reconstruction_error = np.mean((X_test_fault_scaled - X_test_fault_projected) ** 2, axis=1)
    # 计算阈值
    threshold = np.percentile(reconstruction_error, thre_per)

    # print("threshold = ", threshold)

    # 故障检测
    faults = reconstruction_error > threshold

    # 生成测试数据标签（前160正常，后800故障）
    y_test_fault = np.concatenate([np.zeros(160), np.ones(800)])

    # 计算FAR, FDR, DD
    TP = np.sum(faults & y_test_fault.astype(bool))
    FP = np.sum(faults & ~y_test_fault.astype(bool))
    FN = np.sum(~faults & y_test_fault.astype(bool))
    TN = np.sum(~faults & ~y_test_fault.astype(bool))

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FDR = TP / (TP + FN) if (TP + FN) > 0 else 0
    DD = np.mean(reconstruction_error[y_test_fault == 1]) - np.mean(reconstruction_error[y_test_fault == 0])

    # print(f"FAR: {FAR}, FDR: {FDR}, DD: {DD}\n")

    pca_results.append({
        'fault_type': fault_type,
        'threshold': threshold,
        'FAR': FAR,
        'FDR': FDR,
        'DD': DD
    })

    if not i % 4:
        # 绘图
        plt.figure(i, figsize=(15, 6))
        # plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(reconstruction_error, label='Reconstruction Error')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f'Reconstruction Error and Threshold for {fault_type}')
        plt.legend()

        plt.subplot(122)
        plt.plot(y_test_fault, label='Actual Faults', linestyle='--')
        plt.plot(faults.astype(int), label='Detected Faults')
        plt.title(f'Fault Detection for {fault_type}')
        plt.legend()
        plt.tight_layout()


pca_store_res(pca_results, nComp, thre_per)
print("\n\nresults:\n", pca_results)
plt.show()

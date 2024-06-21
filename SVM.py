import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from visualize import *


svm_results = []

# 加载数据
data = scipy.io.loadmat('TE_data.mat')

# 初始化标准化工具
scaler = StandardScaler()

# 初始化SVM模型 使用软标签
svm_model = SVC(kernel='rbf', probability=True)

# 用于存储每种故障的指标
metrics = {}

# 处理每种故障
for i in range(1, 22):
    fault_type = f'd{i:02}'
    # 准备训练数据
    X_train = np.vstack([data['d00'], data[fault_type]])
    y_train = np.array([0] * 500 + [1] * 480)

    # 准备测试数据
    # X_test = np.vstack([data['d00_te'], data[f'd{i:02d}_te']])
    # y_test = np.array([0] * 960 + [1] * 960)  # 假设故障样本测试集全部为故障状态
    X_test = data[f'{fault_type}_te']
    y_test = np.array([0] * 160 + [1] * 800)

    # 数据归一化
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    svm_model.fit(X_train_scaled, y_train)

    # 进行预测
    # y_prob = svm_model.predict(X_test_scaled)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率

    # 设置阈值
    threshold = 0.4  # 你可以调整这个阈值以优化性能

    # 根据阈值进行分类
    y_pred = (y_prob >= threshold).astype(int)


    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # 计算FAR, FDR
    FAR = fp / (fp + tn)
    FDR = tp / (tp + fn)
    DD = calculate_detection_delay(y_pred)

    # 存储指标
    # metrics[f'Fault {i}'] = {'FAR': FAR, 'FDR': FDR, 'DD': DD}
    svm_results.append({'fault_type': fault_type, 'FAR': FAR, 'FDR': FDR, 'DD': DD})

    if (not i % 4) and i < 20:
        fig_num = int(220 + i / 4)
        plt.subplot(fig_num)
        # plt.figure(i, figsize=(15, 6))
        plt.plot(y_prob, label='y_prob')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f'SVM Prediction Score for {fault_type}', fontsize=12)
        plt.legend()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


# 打印所有故障的指标
# for fault, vals in metrics.items():
#     print(f'{fault}: FAR={vals["FAR"]:.4f}, FDR={vals["FDR"]:.4f}, DD={vals["DD"]}')


cca_svm_store_res('svm', svm_results, 0.4)

plt.show()
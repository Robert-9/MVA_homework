import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据
data = loadmat('TE_data.mat')

# 数据预处理
scaler = StandardScaler()

# 初始化PLS模型
pls = PLSRegression(n_components=2)

# 正常操作训练数据
X_train_normal = data['d00'].reshape(-1, 52)
y_train_normal = np.zeros(X_train_normal.shape[0])
X_train_normal_scaled = scaler.fit_transform(X_train_normal)

# 训练PLS模型
pls.fit(X_train_normal_scaled, y_train_normal)

# 循环处理每种故障类型
for i in range(1, 22):
    fault_type = f'd{i:02}'

    # 故障训练数据
    X_train_fault = data[fault_type].reshape(-1, 52)
    y_train_fault = np.ones(X_train_fault.shape[0])

    # 故障测试数据
    X_test_fault = data[f'{fault_type}_te'].reshape(-1, 52)
    y_test_fault = np.ones(X_test_fault.shape[0])

    # 标准化训练和测试数据
    X_train_fault_scaled = scaler.fit_transform(X_train_fault)
    X_test_fault_scaled = scaler.transform(X_test_fault)

    # 训练PLS模型
    pls.fit(X_train_fault_scaled, y_train_fault)

    # 测试模型
    y_pred_fault = pls.predict(X_test_fault_scaled)
    y_pred_fault = (y_pred_fault > 0.5).astype(int).flatten()  # 假设阈值为0.5

    # 性能评估
    print(f"Results for fault type {fault_type}:")
    print(classification_report(y_test_fault, y_pred_fault))
    print(confusion_matrix(y_test_fault, y_pred_fault))

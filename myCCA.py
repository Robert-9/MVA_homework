import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from scipy.linalg import svd, diagsvd
from scipy.stats import chi2


def my_cca(Y_tr, U_tr, epsilon=1e-10):
    """
    执行CCA分析，包括矩阵正则化和稳健的矩阵运算。

    参数:
        Y_tr (numpy.ndarray): 输入矩阵Y。
        U_tr (numpy.ndarray): 输入矩阵U。
        epsilon (float): 对角加载的小正数，用于保证协方差矩阵的正定性。

    返回:
        U (numpy.ndarray): Y的左奇异向量矩阵。
        S (numpy.ndarray): 奇异值矩阵，对角线上的值为典型相关系数。
        V (numpy.ndarray): U的右奇异向量矩阵。
        P (numpy.ndarray): Y的变换矩阵，包含主要的典型向量。
        P_res (numpy.ndarray): Y的残余变换矩阵，包含次要的典型向量。
        L (numpy.ndarray): U的变换矩阵，包含主要的典型向量。
        L_res (numpy.ndarray): U的残余变换矩阵，包含次要的典型向量。
    """
    Y_cov = Y_tr @ Y_tr.T + epsilon * np.eye(Y_tr.shape[0])
    U_cov = U_tr @ U_tr.T + epsilon * np.eye(U_tr.shape[0])
    YU_cov = Y_tr @ U_tr.T

    # 使用特征分解来计算矩阵的稳健平方根逆
    def matrix_sqrt_inv(mat):
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        # 只保留正特征值
        positive_indices = eigenvalues > epsilon
        inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues[positive_indices])
        return eigenvectors[:, positive_indices] @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors[:,
                                                                                   positive_indices].T

    Y_norm = matrix_sqrt_inv(Y_cov)
    U_norm = matrix_sqrt_inv(U_cov)

    # SVD分解
    matrix_to_decompose = Y_norm @ YU_cov @ U_norm
    U, S_diag, Vt = np.linalg.svd(matrix_to_decompose)
    V = Vt.T

    rank = np.linalg.matrix_rank(S_diag)

    # 提取典型相关系数和变换矩阵
    P = Y_norm @ U[:, :rank]
    P_res = Y_norm @ U[:, rank:]
    L = U_norm @ V[:, :rank]
    L_res = U_norm @ V[:, rank:]

    return U, np.diag(S_diag[:rank]), V, P, P_res, L, L_res


def cca_fd(Y_tr, U_tr, Y_test, U_test):
    """
    基于CCA的故障检测函数。

    参数:
        Y_tr (numpy.ndarray): 训练集输入Y。
        U_tr (numpy.ndarray): 训练集输入U。
        Y_test (numpy.ndarray): 测试集输入Y，可能包含故障。
        U_test (numpy.ndarray): 测试集输入U，可能包含故障。

    返回:
        T2_stats (numpy.ndarray): 测试数据的T2统计量数组。
        T2_threshold (float): 基于显著性水平的T2阈值。
    """

    # 调用CCA函数获取变换矩阵和相关系数
    U, S, V, P, P_res, L, L_res = my_cca(Y_tr, U_tr)

    # 计算残差
    rank_S = np.linalg.matrix_rank(S)
    residuals = [P.T @ Y_test[:, i] - S[:rank_S, :rank_S] @ L.T @ U_test[:, i] for i in range(Y_test.shape[1])]
    residuals = np.array(residuals).T

    # 计算残差协方差和T2统计量
    # cov_residuals = np.cov(residuals)

    if residuals.ndim == 1:
        residuals = residuals.reshape(1, -1)  # 确保残差是二维的

    # 确保返回一个二维数组，即使只有一个变量
    cov_residuals = np.cov(residuals, rowvar=False)
    if cov_residuals.size == 1:
        cov_residuals = cov_residuals.reshape(1, 1)  # 确保协方差矩阵至少是1x1

    T2_stats = np.array([x.T @ np.linalg.inv(cov_residuals) @ x for x in residuals.T])

    alpha = 0.05
    T2_threshold = chi2.ppf(1 - alpha, rank_S)

    return T2_stats, T2_threshold


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
    # cca = CCA(n_components=1)
    # cca.fit(X_scaled, y_train)

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

    # # 使用PLS模型进行故障检测
    # X_c, y_pred_fault = cca.transform(X_test_fault_scaled, y_test_fault)
    # # 使用CCA得分判断故障
    # # threshold = np.percentile(X_c, 20)  # 设定阈值
    # threshold = 0
    # print("thre = ", threshold)
    # # y_pred_fault_label = y_pred_fault[:, 0] > 0.1
    # y_pred_fault_label = X_c < threshold

    t2, threshold = cca_fd(X_scaled, y_train, X_test_fault_scaled, y_test_fault)
    y_pred_fault_label = (t2 > threshold).astype(int)
    y_pred_fault = t2

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

    if not(FDR>0.7 and FAR<0.5):
        plt.figure(i, figsize=(15, 6))
        plt.subplot(121)
        plt.plot(y_pred_fault, label='CCA Predicted')
        plt.title(f'CCA Prediction Scores for Fault Type {fault_type}')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Score')
        plt.legend()
        plt.subplot(122)
        plt.plot(y_pred_fault, label='X_c')
        plt.axhline(y=threshold, color='red', label='Threshold', linestyle='--')
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
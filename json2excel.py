import json
import pandas as pd


file_path = 'svmResults_threVal0.4.json'

# 加载JSON数据
with open(file_path, 'r', encoding='utf-8') as file:
    # 加载JSON数据
    data = json.load(file)

# 访问'faults'字典
faults = data["faults"]

# 创建一个列表来存储结果
results = []

# 遍历'faults'字典
for fault_id, fault_data in faults.items():
    # 只关心'FAR', 'FDR', 'DD'字段
    far = fault_data.get("FAR", "N/A")  # 使用get方法，如果字段不存在则返回"N/A"
    fdr = fault_data.get("FDR", "N/A")
    dd = fault_data.get("DD", "N/A")
    if far > 0:
        far = round(far, 3)
    else:
        far = 0.0
    fdr = round(fdr, 3)

    # 将结果添加到列表中
    results.append({
        "dxx": fault_id,
        "FAR": far,
        "FDR": fdr,
        "DD": dd
    })

# 打印结果
for result in results:
    print(result)

# 将结果转换为DataFrame
df = pd.DataFrame(results)

# 保存到Excel文件
df.to_excel('svm_faults_data.xlsx', index=False)
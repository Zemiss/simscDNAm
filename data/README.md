# 数据与预处理

入口为 [prepare.ipynb](prepare.ipynb)。原始 H5AD 只读；本地数据存入 `raw/`，服务器从允许的数据根目录读取。修改 `INPUT_PATHS`、`DATASET`、`OUTPUT_DIR` 后，在 simscDNAm 内核中运行。

按源数据集 × Subclass 分层，以种子 42 划分 80% 训练和 20% 留出；无标签及全缺失细胞剔除。零值和 NaN 保留，真实有限值须位于 0–1。自然区域至少 1000 个，区域 ID 唯一。复制来源无放回抽样，训练和留出追加同一批 1000 列，数值及缺失位置与来源完全相同。

输出放在 `outputs/preprocessing/<数据集>/`：`TrainingReal.h5ad`、`HeldoutReal.h5ad`、`region_pairs.csv`、细胞清单和 `run_summary.json`。所有模型使用同一份输出。逐区域填补统计由各模型使用训练数据拟合，预处理不填补外层留出。

`preprocessing.py` 实现划分与导出；`training.py` 实现分块输入和训练 Dataset；`evaluation.py` 实现评估的区域轴校验、抽样和训练参照统计。原始数据不提交 Git。

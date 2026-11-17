# 数据与预处理

```bash
python -m data.prepare --dataset liu2023/full
python -m data.prepare --inputs data/raw/数据集/原始.h5ad --output outputs/preprocessing/自定义数据集
python -m data.prepare --help
```

使用 simscDNAm 环境。默认输入是 Liu2023 五个原始 H5AD；本地原始数据位于 `data/raw/`，服务器位于项目允许的数据根目录。`--seed` 默认为 42，`--test-fraction` 默认为 0.2，非空输出目录拒绝覆盖。

按源数据集 × Subclass 分层划分训练和留出；无标签及全缺失细胞剔除。真实零和 NaN 保留，有限值须位于 0–1。自然区域至少 1000 个，区域 ID 唯一。复制来源无放回抽样，训练和留出追加相同的 1000 列，值和缺失位置与来源一致。

输出为 `TrainingReal.h5ad`、`HeldoutReal.h5ad`、`region_pairs.csv`、细胞清单和 `run_summary.json`。所有模型读取同一划分，填补统计只使用各模型的训练数据。

`preprocessing.py` 实现划分与导出，`training.py` 实现分块输入和训练 Dataset，`evaluation.py` 实现评估预处理，`baseline.py` 实现基线共用读入和逐元素核对，`paths.py` 管理默认路径与服务器输出约束。原始数据不提交 Git。

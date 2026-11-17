# simscDNAm

单细胞 DNA 甲基化数据模拟。正式流程为：条件 VAE 与潜空间扩散 → 同类型训练缺失掩码 → 联合秩映射及经验分位数对齐。训练、推理和评估 notebook 均通过 Jupyter 运行，环境使用 `simscDNAm`。

## 目录

| 目录 | 职责与入口 |
|---|---|
| `data/` | [prepare.ipynb](data/prepare.ipynb) 处理原始数据；模块负责输入、质控、划分和复制区域 |
| `model/src/` | 条件 VAE、潜空间去噪器、扩散日程、损失、DDIM 与配置 |
| `model/script/` | [train.ipynb](model/script/train.ipynb) 训练；[inference.ipynb](model/script/inference.ipynb) 复用模型推理 |
| `baseline/FittedMarginal/` | [inference.ipynb](baseline/FittedMarginal/inference.ipynb)，零、一和 Beta 混合边缘基线 |
| `baseline/simATAC/` | [inference.ipynb](baseline/simATAC/inference.ipynb)，官方 simATAC 与既有小类型估计回退 |
| `baseline/simCAS/` | [inference.ipynb](baseline/simCAS/inference.ipynb)，官方 simCAS 与固定类型配额抽样 |
| `baseline/STADiffuser/` | [inference.ipynb](baseline/STADiffuser/inference.ipynb)，禁用空间条件的官方模型 |
| `evaluate/A1/` | [evaluate.ipynb](evaluate/A1/evaluate.ipynb)，边缘分布指标及两类分布图 |
| `evaluate/A2/` | [evaluate.ipynb](evaluate/A2/evaluate.ipynb)，复制区域相关性及小提琴／柱状图 |
| `evaluate/A3/` | [evaluate.ipynb](evaluate/A3/evaluate.ipynb)，ANOVA、PCA、30近邻 miLISI 及联合 UMAP |
| `outputs/` | [输出规则](outputs/README.md)，按数据集、模型和实验分开；旧结果归档 |

## 运行顺序

```bash
conda activate simscDNAm
python -m pip install -r requirements.txt
python -m ipykernel install --user --name simscdnam --display-name "Python (simscDNAm)"
jupyter lab
```

1. 修改并运行 `data/prepare.ipynb`，生成同一数据集的训练／留出 H5AD 和 `region_pairs.csv`。原始矩阵为细胞 × 区域，须有真实 `obs['Subclass']`；NaN 为缺失，0 为真实零值。
2. 修改并运行 `model/script/train.ipynb`。已有完整训练目录时，在 `inference.ipynb` 加载 checkpoint、内部 fit 掩码和映射参数生成新数据。
3. 运行各基线目录内的 `inference.ipynb`，使用相同训练输入与目标配额。R 和官方模型依赖见 [baseline/README.md](baseline/README.md)。
4. 修改 [evaluate/config.yaml](evaluate/config.yaml)，只列出本轮已完成的方法，分别运行 A1/A2/A3 的 notebook，各入口只执行本实验。服务器改用 [config.server.yaml](evaluate/config.server.yaml)。

本地原始数据放在 `data/raw/`，运行结果写入 `outputs/`。服务器原始数据只读取 `/home/sccasimp/data/xb/simscDNAm/`，代码位于 `/home/sccasimp/program/xb/simscDNAm/`，产物只写 `/home/sccasimp/data/xb/outputs/simscDNAm/`。大矩阵、权重、缓存和运行中的 notebook 输出不提交 Git。

## 评估口径与解释

| 实验 | 定义 |
|---|---|
| A1 | 自然区域按真实类型比例加权的 1-WD、类型细胞均值 MSE、区域均值 Pearson |
| A2 | 实际生成的 1000 对原始／复制区域 Pearson；另存自然区域 Spearman 辅助统计 |
| A3 | 真实训练参照 ANOVA 选最多 50000 个自然区域、PCA 最多 50 维、30近邻 miLISI；按类型中位数等权平均 |

A1/A3 排除末尾 1000 个复制区域，合成矩阵不截断；A2 不在生成后重建复制列。图中文字使用英文，其余说明使用中文。

联合秩映射只使用内部 fit 真实值，但会改变原生成排序与几何，高复制相关不能单独证明原始 VAE 学到了依赖结构。simATAC、simCAS 保留原生计数或分数，可能超出 0–1；当前比较存在数值域差异。已多轮查看的留出结果属于探索性验证。

既有结果：[2026年11月17日全量及子集比较](outputs/reports/knn30_20261117/README.md)、[STADiffuser 五方法比较](outputs/reports/stadiffuser_20261111/README.md)。本轮仅整理代码与路径，未重新训练这些正式结果。

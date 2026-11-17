# simscDNAm

单细胞 DNA 甲基化模拟。正式流程为条件 VAE 与潜空间扩散 → 同类型训练缺失掩码 → 联合秩映射及经验分位数对齐。正式代码全部使用 Python，通过命令行运行。

## 目录

| 目录 | 职责与入口 |
|---|---|
| `data/` | [prepare.py](data/prepare.py)：共用质控、训练／留出划分和训练前复制区域；`training.py`、`evaluation.py`、`baseline.py` 实现数据读取与预处理 |
| `model/src/` | 模型架构、训练、生成、缺失掩码、联合秩映射和完整流程 |
| `model/script/` | [train.py](model/script/train.py)、[inference.py](model/script/inference.py)：训练和独立推理命令行入口 |
| `baseline/` | FittedMarginal、simATAC、simCAS、STADiffuser 各目录的 `inference.py` |
| `evaluate/A1/` | [evaluate.py](evaluate/A1/evaluate.py)：边缘分布指标及两类分布图 |
| `evaluate/A2/` | [evaluate.py](evaluate/A2/evaluate.py)：复制区域相关性及小提琴／柱状图 |
| `evaluate/A3/` | [evaluate.py](evaluate/A3/evaluate.py)：ANOVA、PCA、30近邻 miLISI 及联合 UMAP |
| `outputs/` | [输出规则](outputs/README.md)：按数据集、模型和实验分别保存 |

## 安装与运行

在仓库根目录执行，使用 conda 环境 `simscDNAm`：

```bash
conda activate simscDNAm
python -m pip install -r requirements.txt

python -m data.prepare --dataset liu2023/full
python -m model.script.train --dataset liu2023/full

python -m baseline.FittedMarginal.inference --dataset liu2023/full
python -m baseline.simATAC.inference --dataset liu2023/full
python -m baseline.simCAS.inference --dataset liu2023/full
python -m baseline.STADiffuser.inference --dataset liu2023/full --device cuda:0

python -m evaluate.A1.evaluate --config evaluate/config.yaml
python -m evaluate.A2.evaluate --config evaluate/config.yaml
python -m evaluate.A3.evaluate --config evaluate/config.yaml
```

1. `data.prepare` 默认读取 Liu2023 的五个原始 H5AD；其他数据使用 `--inputs 文件1.h5ad 文件2.h5ad`，输出通过 `--output` 指定。
2. 训练和各基线默认读取 `outputs/preprocessing/<数据集>/` 的共用输入。可用 `--training`、`--target`、`--pairs` 指定实际路径，各命令的 `--help` 列出支持项。
3. 已有完整训练目录时，运行 `python -m model.script.inference --trained-run 训练目录 --training 原训练.h5ad --target 目标.h5ad --output 新输出目录`。使用 `--device cpu` 或 `--device cuda:0` 选择设备。
4. 在 [evaluate/config.yaml](evaluate/config.yaml) 中只填写本轮已生成的方法。服务器使用 [config.server.yaml](evaluate/config.server.yaml)。三个评估命令独立运行并保存各自的指标和图。

也可直接运行文件，例如 `python model/script/train.py --help`；从其他工作目录直接运行时使用文件的绝对路径。导入模块不会开始训练、下载官方模型或读写实验数据。正式运行不需要 Jupyter 或 IPython。

## 数据与输出

原始矩阵为细胞 × 区域，须有真实 `obs['Subclass']`；NaN 表示缺失，0 为真实零值。本地原始数据放在 `data/raw/`，运行结果写入 `outputs/`。

服务器代码位于 `/home/sccasimp/program/xb/simscDNAm/`；原始数据从 `/home/sccasimp/data/xb/simscDNAm/` 读取，产物只写入 `/home/sccasimp/data/xb/outputs/simscDNAm/`。大矩阵、权重和缓存不提交 Git。基线额外依赖见 [baseline/README.md](baseline/README.md)。

## 评估口径

| 实验 | 定义 |
|---|---|
| A1 | 自然区域按真实类型比例加权的 1-WD、类型细胞均值 MSE、区域均值 Pearson |
| A2 | 实际生成的 1000 对原始／复制区域 Pearson；另存自然区域 Spearman 辅助统计 |
| A3 | 真实训练参照 ANOVA 选最多 50000 个自然区域、PCA 最多 50 维、30近邻 miLISI；按类型中位数等权平均 |

A1/A3 排除末尾 1000 个复制区域，合成矩阵不截断，A2 不在生成后重建复制列。图中文字使用英文。

联合秩映射只使用内部 fit 真实值，但会改变原生成排序与几何，高复制相关不能单独证明原始 VAE 学到了依赖结构。simATAC、simCAS 保留原生计数或分数，可能超出 0–1；当前比较存在数值域差异。多轮查看的留出结果属于探索性验证。

既有结果：[全量及子集比较](outputs/reports/knn30_20261117/README.md)、[STADiffuser 五方法比较](outputs/reports/stadiffuser_20261111/README.md)。本轮迁移运行方式，未重新训练这些正式结果。

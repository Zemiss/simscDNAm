# simscDNAm

simscDNAm 用于生成单细胞DNA甲基化数据，完整流程为 **CVAE及条件潜空间扩散 → 同类型训练缺失掩码 → 联合秩映射与经验分位数对齐**。训练、生成和映射在Jupyter中执行。

## 目录

| 入口 | 用途 |
|---|---|
| [simscDNAm/train.ipynb](simscDNAm/train.ipynb) | 主模型训练或复用checkpoint，依次执行mask和对齐，导出正式数据 |
| [simscDNAm/README.md](simscDNAm/README.md) | 模型结构、配置、复用条件、输出和方法边界 |
| [baseline/FittedMarginal.ipynb](baseline/FittedMarginal.ipynb) | 零、一和Beta混合边缘基线 |
| [baseline/simATAC.ipynb](baseline/simATAC.ipynb) | simATAC及既有小类别估计回退 |
| [baseline/simCAS.ipynb](baseline/simCAS.ipynb) | 官方simCAS及固定类别配额抽样 |
| [evaluate/A1A2A3.ipynb](evaluate/A1A2A3.ipynb) | 四方法统一A1/A2/A3和四类PNG/PDF图 |
| [results/liu2023/README.md](results/liu2023/README.md) | Liu2023全量数据比较、实际配置、分数与图表 |

## 环境与运行

使用conda环境`simscDNAm`，从仓库根目录启动Jupyter，并选择`simscdnam`内核：

```bash
conda activate simscDNAm
python -m pip install -r requirements.txt
python -m ipykernel install --user --name simscdnam --display-name "Python (simscDNAm)"
jupyter lab
```

1. 准备真实训练H5AD与独立留出H5AD，均含真实`obs['Subclass']`；训练输入的完整区域轴须在训练前追加1000个复制区域并保留配对CSV。NaN表示缺失，0是真实数值。
2. 修改主模型notebook的`DATASETS`、`RUN_DATASETS`和`RUN_ROOT`。`PRETRAINED_RUNS={}`从头训练；填写既有CVAE＋mask目录可复用。默认提供Liu2023全量数据的服务器路径示例。
3. 修改并运行三个baseline notebook。默认`RUN_MODE="run"`实际生成；需要R和官方simCAS运行时，详见[baseline说明](baseline/README.md)。
4. 在[evaluate/config.yaml](evaluate/config.yaml)填写同一真实留出、真实训练参照、四种方法输出和配对CSV，运行统一评估notebook。

输出应使用新目录。输入、模型权重、H5AD、拟合分位数和大型缓存不随代码仓库上传；本仓库保存代码、小型结果表和图表。

## 方法和评估口径

CVAE保持原网络、损失和DDIM生成。mask整行抽自同Subclass的内部fit真实细胞。对齐只用fit真实值：先在正态秩空间学习最多32个联合PCA因子，投影生成分数并调整因子协方差，再逐区域匹配真实经验分位数；固定投影强度1。小样本类型保持原生成值，缺失位置保持。

对齐作用于同类型的一整批生成细胞，会改变原CVAE的排序和细胞几何。高复制相关来自后处理约束，不能单独证明原CVAE自主学好依赖结构。

| 评估 | 内容 |
|---|---|
| A1 | 全部自然区域的同类型1-WD、类型细胞均值MSE、区域均值Pearson，按真实类型比例加权 |
| A2 | 1000对原始／复制区域实际输出的Pearson，报告有效配对数；另存自然区域Spearman统计 |
| A3 | 在真实训练参照上ANOVA选50000自然区域、PCA50维，再算30近邻miLISI；逐方法联合UMAP用于展示 |

当前全量及九个子集的四方法比较、全量缺失mask／对齐消融，统一采用30近邻，见[完整得分与图表](results/knn30_20261117/README.md)。miLISI仍按真实Subclass的中位数等权平均，UMAP也使用30近邻。本地源码与服务器默认目录均使用30近邻；本轮五项指标已实际重算，权重与合成数据保持不变。

A1/A3排除复制尾部，合成值不截断。simATAC和simCAS使用当前协议的比例输入、保留原生计数／分数，数值域可能超过0–1；数值域差异见输入诊断，不将该比较解释为原方法在其原任务上的能力排序。

Liu2023本轮使用4740训练细胞（4257 fit、483内部验证）和1190留出细胞，映射在全部273544个区域上应用同一规则。该留出划分已多轮查看，属于探索性验证。对齐明显改善1-WD和复制Pearson，也降低原CVAE的A1 Pearson、miLISI；完整升降见结果页。

## 验证

```bash
python -m pytest simscDNAm validation baseline evaluate -q
```

覆盖网络与训练隔离、mask来源、映射分位数、NaN、区域轴、训练缓存核验、中断恢复，以及通过真实Jupyter内核运行的两种参照和两种H5AD读取模式。完整数据复现另核对正式映射矩阵、NaN、checkpoint与五项指标。

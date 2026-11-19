# 基线训练与生成

各基线目录的 `inference.py` 独立运行，使用同一预处理训练集、目标 Subclass 配额和完整区域轴。

```bash
python -m baseline.FittedMarginal.inference --dataset liu2023/full
python -m baseline.simATAC.inference --dataset liu2023/full
python -m baseline.simCAS.inference --dataset liu2023/full
python -m baseline.PeakVI.inference --dataset liu2023/full --device cuda:0
python -m baseline.DiTSim.inference --dataset liu2023/full --device cuda:0
```

各入口支持 `--training`、`--target`、`--output` 和 `--seed`，使用 `--help` 查看全部参数。默认结果写入 `outputs/baseline/<数据集>/<基线>/`。模块导入和帮助命令不会开始训练或安装依赖。

| 基线 | 依赖与方法 |
|---|---|
| FittedMarginal | Python；零、一和 Beta 混合边缘分布，保留观测率与 NaN |
| simATAC | R、Rscript 和官方 simATAC；训练中位数填补，保留原小类型估计回退；独立 R 库路径为 `baseline/simATAC/.runtime/r_library` |
| simCAS | 官方源码放入 `baseline/simCAS/vendor/`，另需 rpy2 和 R4；使用环境变量 `R_HOME` 指定 R 运行时，未设置时沿用服务器默认路径 |
| PeakVI | 官方 scvi-tools 1.4.2；原生随机后验可及性概率输出，按同类型训练供体行匹配目标配额；[说明](PeakVI/README.md) |
| DiTSim | 官方固定源码与 quick_start 的完整 AE/DiT 参数，按真实类型配额调用 sampling；[说明](DiTSim/README.md) |

PeakVI 和 DiTSim 的训练与生成函数在 `notebooks/基线_*.ipynb` 内，Python 入口只传递路径、种子和设备并执行当前 simscDNAm 内核。使用 `python -m ipykernel install --user --name simscdnam --display-name simscDNAm` 注册内核。执行记录保存到项目 `notebooks/执行记录/`。

复制区域在训练前就已存在，各模型直接生成全部列。PeakVI 和 DiTSim 共用仅从真实训练集拟合的逐区域中位数填补缓存，生成后只裁剪到 0–1，不加掩码或对齐。simATAC、simCAS 保留原生计数／分数，已有方法结果不改。

三个传统基线支持 `--reference 已有结果.h5ad` 进行逐元素核对。仅在需要重放历史中间结果时使用 `--replay-root 历史根目录` 并同时提供 `--reference`。

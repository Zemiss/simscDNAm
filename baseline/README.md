# 基线推理

每个基线有独立文件夹和 `inference.ipynb`，算法及生成函数直接写在 notebook 内。均使用 `data/prepare.ipynb` 导出的完整训练集、独立留出集的 Subclass 配额及相同区域轴。修改各入口的 `DATASET` 和路径后，通过 simscDNAm 环境的 Jupyter 运行。

| 基线 | 依赖及方法 |
|---|---|
| FittedMarginal | Python；按类型拟合零、一、Beta 混合分布，保留观测率与 NaN |
| simATAC | R、Rscript 和官方 simATAC；训练中位数填补，保留现有小类型估计回退 |
| simCAS | 官方 simCAS、rpy2 和 R4；将官方仓库放在 `simCAS/vendor/`，`R_HOME` 按机器调整 |
| STADiffuser | 官方提交 `11be35ffce082d5d9cd4bfdcc8f8352fb4a58971`；首次运行下载并核对 SHA256，依赖放在 `STADiffuser/.runtime/`；需要 CUDA |

输出放在 `outputs/baseline/<数据集>/<基线>/`。三个传统基线生成 `data.h5ad`、复制配对表和 `verification.json`。STADiffuser 正式训练为 AE 500 轮、扩散 1000 轮；`pilot` 仅为试运行，独立存入 `pilot/`。非空间配置使用自环图与 `spatial_coord=None`，不做 RNA 总量归一化。

复制区域在训练前就已存在，各模型直接生成全部列。simATAC、simCAS 保留原生计数／分数，STADiffuser 保留原生连续输出，不为改善评估而截断、加掩码或对齐。

三个传统基线保留原 `replay` 校验模式，仅用于核对历史中间结果；常规运行保持 `RUN_MODE="run"`、`REFERENCE_PATH=None`。

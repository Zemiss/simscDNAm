# 基线训练与生成

各基线目录的 `inference.py` 独立运行，使用同一预处理训练集、目标 Subclass 配额和完整区域轴。

```bash
python -m baseline.FittedMarginal.inference --dataset liu2023/full
python -m baseline.simATAC.inference --dataset liu2023/full
python -m baseline.simCAS.inference --dataset liu2023/full
python -m baseline.STADiffuser.inference --dataset liu2023/full --device cuda:0
```

各入口支持 `--training`、`--target`、`--output` 和 `--seed`，使用 `--help` 查看全部参数。默认结果写入 `outputs/baseline/<数据集>/<基线>/`。模块导入和帮助命令不会开始训练或安装依赖。

| 基线 | 依赖与方法 |
|---|---|
| FittedMarginal | Python；零、一和 Beta 混合边缘分布，保留观测率与 NaN |
| simATAC | R、Rscript 和官方 simATAC；训练中位数填补，保留原小类型估计回退；独立 R 库路径为 `baseline/simATAC/.runtime/r_library` |
| simCAS | 官方源码放入 `baseline/simCAS/vendor/`，另需 rpy2 和 R4；使用环境变量 `R_HOME` 指定 R 运行时，未设置时沿用服务器默认路径 |
| STADiffuser | 固定官方提交 `11be35ffce082d5d9cd4bfdcc8f8352fb4a58971`，首次正式运行下载并校验源码，固定依赖放入 `.runtime/`；需要 CUDA |

STADiffuser 使用 `--mode pilot` 进行独立试运行，结果放在 `pilot/`。正式训练为 AE 500 轮、扩散 1000 轮；非空间配置使用自环图与 `spatial_coord=None`，不做 RNA 总量归一化。

复制区域在训练前就已存在，各模型直接生成全部列。simATAC、simCAS 保留原生计数／分数，STADiffuser 保留原生连续输出，不截断、不增加训练掩码或对齐。

三个传统基线支持 `--reference 已有结果.h5ad` 进行逐元素核对。仅在需要重放历史中间结果时使用 `--replay-root 历史根目录` 并同时提供 `--reference`。

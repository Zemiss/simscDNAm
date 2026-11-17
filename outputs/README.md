# 输出目录

本地运行结果统一放在此目录；服务器对应 `/home/sccasimp/data/xb/outputs/simscDNAm/`。代码目录只保存入口和配置。

```text
outputs/
  preprocessing/<数据集>/       # 共用训练／留出 H5AD、复制区域表、质控和划分记录
  model/<数据集>/               # 主模型训练、候选、checkpoint、映射及正式生成结果
  inference/<数据集>/simscDNAm/  # 独立推理，含原生成、加掩码和最终 H5AD
  baseline/<数据集>/<基线>/     # 各基线模型的独立输出、日志和核验记录
  evaluate/<数据集>/A1/         # A1 表格与 figures/
  evaluate/<数据集>/A2/         # A2 表格、辅助向量与 figures/
  evaluate/<数据集>/A3/         # A3 表格、坐标与 figures/
  archive/                     # 整理前的历史结果及实验记录
  reports/                     # 已发布的小型指标表与图，随代码保存
  verification/                # 本轮 Jupyter 验证记录，仅本地保存
```

`<数据集>` 默认是 `liu2023/full`；子集可使用 `liu2023/GSE215342`。重新运行时使用新目录，避免覆盖历史记录。STADiffuser 正式产物直接放在其基线目录，试运行放在其中的 `pilot/`。

大矩阵、权重和运行缓存不提交 Git；本目录跟踪此说明与 `reports/` 中的小型结果。`archive/results/` 保留原 results，`archive/runs/` 保留原 output，`archive/experiments/` 保留旧模型探索 notebook，`archive/docs/` 保留旧实验记录。

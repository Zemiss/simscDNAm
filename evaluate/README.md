# A1／A2／A3 独立评估

每个目录包含 `evaluate.py`、`metrics.py` 和 `plotting.py`，分别负责执行、指标和图形；输入核验与生命周期管理共用 `runtime.py`。

```bash
python -m evaluate.A1.evaluate --config evaluate/config.yaml
python -m evaluate.A2.evaluate --config evaluate/config.yaml
python -m evaluate.A3.evaluate --config evaluate/config.yaml
```

服务器使用 `--config evaluate/config.server.yaml`。相对输入、输出和展示清单路径以 YAML 所在目录解析。配置中只保留本轮已有生成数据的方法。三个命令互不依赖执行顺序，均支持 `--help`。

输入是共用预处理导出的真实训练、真实留出及各方法生成 H5AD。均含真实 `obs['Subclass']`，完整区域轴名称与顺序一致；末尾固定 1000 个复制区域。真实复制值与来源相同，合成复制值使用模型实际输出。

| 实验 | 计算与图形 |
|---|---|
| A1 | 自然区域类型加权 1-WD、类型细胞均值 MSE、区域均值 Pearson；多区域边缘分布、单区域按类型分布 |
| A2 | 实际 1000 对 Pearson；小提琴及正式均值柱状图；自然区域 Spearman 辅助统计 |
| A3 | 真实参照 ANOVA 最多 50000 个自然区域、PCA 最多 50 维、30近邻 miLISI；逐方法联合 UMAP |

结果写入 `output_dir/<实验>/`，包括表格、`figures/*.png` 与 `*.pdf`、抽样细胞、输入诊断和 `run_summary.json`。绘图使用无窗口后端，可直接在服务器执行。

`reference_h5ad` 默认使用独立训练参照，与留出 ID 不得重叠；设为 null 时使用全部评估真实数据拟合，并记录为非独立留出评估。A1/A3 排除复制尾部。A3 按类型取来源逆 Simpson 指数中位数再等权平均；UMAP 只作展示，各面板坐标不可直接比较。

A1 需要 `a1_regions_csv` 展示清单，包含原region、展示名称、基因组版本、文献来源、单区域展示、展示细胞类型；恰有一行指定单区域展示，类型以 `|` 分隔。默认 [Liu2023 清单](A1/marker_regions_liu2023.csv) 沿用 mm10 Tle4 训练选区；更换数据须替换清单。A2/A3 不读取该清单。

A2 少于三个共同有限观测或恒定的配对记缺失，并报告有效对数。合成值不截断。`read_in_memory=false` 使用只读 backed 分块读取，设为 true 则载入内存，口径一致。输出目录不得包含输入；失败会写入运行记录并关闭 H5AD。

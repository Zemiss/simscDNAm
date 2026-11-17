# A1／A2／A3 独立评估

三个目录各自包含 `evaluate.ipynb` 和对应 `metrics.py`。修改共用 [config.yaml](config.yaml)，分别通过 Jupyter 运行，互不依赖执行顺序。服务器将首个配置单元的 `CONFIG_PATH` 改为 `PROJECT / "evaluate/config.server.yaml"`。

输入是预处理导出的真实训练、真实留出和各方法生成 H5AD。矩阵为细胞 × 区域，含真实 `obs['Subclass']`，末尾固定 1000 个复制区域。真实和合成区域名及顺序必须完全一致；真实复制值与来源相同，合成复制值使用模型实际输出。

| 入口 | 计算与图形 | 输出子目录 |
|---|---|---|
| [A1/evaluate.ipynb](A1/evaluate.ipynb) | 自然区域类型加权 1-WD、类型细胞均值 MSE、区域均值 Pearson；多区域边缘分布、单区域按类型分布 | `A1/` |
| [A2/evaluate.ipynb](A2/evaluate.ipynb) | 实际 1000 对 Pearson；小提琴及正式均值柱状图；自然区域 Spearman 辅助统计 | `A2/` |
| [A3/evaluate.ipynb](A3/evaluate.ipynb) | 真实参照 ANOVA 最多 50000 个自然区域、PCA 最多 50 维、30近邻 miLISI；逐方法联合 UMAP | `A3/` |

每个实验在 `output_dir/<实验>/` 写入表格、`figures/*.png` 和 `*.pdf`、`selected_cells.csv`、输入诊断与 `run_summary.json`，状态以该 JSON 为准。共享随机种子与抽样协议，默认每种合成方法无放回抽取与真实留出等量的细胞。A1/A2 不执行 A3 的 PCA、UMAP 或近邻计算。

`reference_h5ad` 默认是独立训练参照，与留出细胞 ID 不得重叠；设为 null 时使用全部真实评估数据拟合参照，并记录为非独立留出评估。A1/A3 排除复制尾部。A3 按类型汇总来源逆 Simpson 指数的中位数再等权平均。UMAP 仅展示，各面板坐标不可直接比较。合成值不截断。

A1 绘图需要 `a1_regions_csv`，包含原region、展示名称、基因组版本、文献来源、单区域展示、展示细胞类型；恰有一行指定单区域展示，类型以 `|` 分隔。默认 [Liu2023 清单](A1/marker_regions_liu2023.csv) 的 mm10 Tle4 窗口沿用原训练数据筛选，换数据时需更换清单。A2/A3 不读取此清单。A2 少于三个共同有限观测或恒定的配对记缺失，并报告有效配对数。

`read_in_memory=false` 使用只读 backed 分块读取；内存充足可设 true，两种方式采用同一定义。输出目录不得包含输入文件，异常时记录失败并关闭 H5AD。

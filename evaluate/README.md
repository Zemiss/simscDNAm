# A1、A2、A3 统一评估

修改 config.yaml 的输入路径，在 simscDNAm 环境的 Jupyter 中从头运行 A1A2A3.ipynb，即可一次输出三类评估、图表和总表。该目录可随代码仓库使用，也可整体复制；不依赖原 notebooks、训练脚本、既有实验目录或聚类模型。

## 输入

必需输入为一个真实 H5AD，以及一个或多个已训练生成的合成 H5AD。真实和合成数据均须包含训练前追加的末尾 1000 个复制区间，所有文件的区域名称与顺序完全一致。矩阵 X 是细胞乘区域，NaN 表示缺失；零按实际零值处理，稀疏矩阵的隐式零也按零值处理。

所有评估文件须有 obs['Subclass']：真实文件使用原始真实细胞类型，合成文件使用生成时保存的对应类型标签。不要以聚类标签替代，也不根据生成结果反推真实标签。正式主模型名称为`simscDNAm`，包含CVAE、mask与联合秩映射。

region_pairs_csv 建议填写扩维时保存的配对表路径。表中必须恰有 1000 行，索引从 0 开始：

| 字段 | 含义 |
| --- | --- |
| 原region索引 | 复制来源在原始区间中的列索引，1000 个来源互不重复 |
| 复制region索引 | 对应复制列，必须完整覆盖最后 1000 列 |
| 原region | 可选；提供时与实际原区间名称核对 |
| 复制region | 可选；提供时与实际复制列名称核对 |

也接受英文列名 original_index、duplicate_index、original_region、duplicate_region。没有配对表时，仅当末尾列采用 原区间名__duplicate_编号 的现有命名规则时自动解析，其他命名会要求提供配对表。真实文件中复制值和缺失位置必须与来源完全一致；合成复制列保留模型实际输出，绝不现场重新复制。

reference_h5ad 可填独立的真实训练参照，用于 ANOVA、中位数和 PCA 拟合，也须有同一完整区域轴和复制列。它与评估真实数据不得有细胞 ID 重叠。若为 null，则使用 real_h5ad 的全部细胞拟合参照，并明确记录为“评估真实数据拟合参照，非独立留出评估”。不需要额外训练生成模型。

a1_regions_csv 指定展示区间证据清单，genome_assembly 声明输入数据的基因组版本；不再随机选择区间。清单必须包含原region、展示名称、基因组版本、文献来源、单区域展示、展示细胞类型。区域必须存在于原始轴且不重复，基因组版本须一致；单区域展示列须恰有一个 1，该行展示细胞类型以 `|` 分隔，使用文件中保存的 Subclass 标签。缺少指定类型时停止，不合并或替换标签。请为其他物种、基因组版本或区域轴另备清单。

evaluation_cells 为 null 时使用全部评估真实细胞，各方法固定种子无放回抽取相同数量；合成细胞不足会停止，不重复采样。所有指标共用同一批抽样细胞；训练参照不抽样。若限制 evaluation_cells，未给独立训练参照时仍用全部真实数据拟合参照。

## 运行

本地在仓库根目录执行：

```powershell
conda activate simscDNAm
jupyter lab evaluate/A1A2A3.ipynb
```

选择 Python (simscDNAm) 内核。若该内核尚未注册，先在已激活的环境内运行：

```powershell
python -m ipykernel install --user --name simscdnam --display-name "Python (simscDNAm)"
```

也可通过 Jupyter 批量执行：

```powershell
python -m jupyter nbconvert --execute --to notebook evaluate/A1A2A3.ipynb --output A1A2A3.executed.ipynb --output-dir tmp --ExecutePreprocessor.kernel_name=simscdnam --ExecutePreprocessor.timeout=-1
```

仅在缺少依赖时，在 simscDNAm 环境安装 requirements.txt。配置中的相对路径均相对于 config.yaml。本地请将 output_dir 改为本地路径，例如 ../output/A1A2A3；实际数据文件不需要加入 Git。

服务器使用 simscDNAm 虚拟环境和现有 Jupyter。代码工作目录为 /home/sccasimp/program/xb/simscDNAm/evaluate，读取数据遵循 /home/sccasimp/data/xb/simscDNAm 的约束；输出只写 /home/sccasimp/data/xb/outputs/simscDNAm 下。默认配置中的示例文件名需替换为实际文件；不包含服务器密码。

## 评估口径

| 项目 | 使用区间 | 计算方式与方向 |
| --- | --- | --- |
| A1 加权 1-WD | 仅前面的原始区间 | 每个真实类型内逐区间计算一维 Wasserstein，先平均有效区间，再按评估真实类型比例加权；越小越好 |
| A1 加权 MSE | 仅前面的原始区间 | 每个细胞先平均其有限区间值，再求类型内细胞均值，计算真实与合成类型均值之差的平方，最后按真实比例加权；越小越好 |
| A1 加权 Pearson | 仅前面的原始区间 | 类型内逐区间均值向量的 Pearson，再按真实类型比例加权；越接近 1 越好 |
| A2 主指标 | 原区间与最后 1000 个复制区间 | 实际输出的 1000 对 Pearson 均值；越接近 1 越好。少于三个共同有限观测或恒定对记缺失，并报告有效对数 |
| A2 辅助统计 | 仅原始区间 | 按真实参照 Subclass ANOVA 选最多 2000 个区间；用参照中位数填补后计算所有区间对 Spearman；原始向量全部保存，不绘图 |
| A3 miLISI | 仅原始区间 | 按真实参照 Subclass ANOVA 选最多 50000 个区间，仅在真实参照拟合最多 50 维 PCA；真实与合成联合取 30 近邻，计算来源逆 Simpson 指数，按真实类型取中位数后等权平均；越接近 2 混合越充分 |
| A3 逐方法联合 UMAP | 同一真实参照 PCA 空间 | 每种方法分别将评估真实和合成 PCA 坐标合并拟合 UMAP；各面板的坐标不能直接比较 |

A1 缺少某一真实类型时，相应加权指标保留 NaN，不重新分配权重；有效类型权重和逐类型明细同时输出。常量均值向量的 Pearson 也记为缺失。额外合成类型仍参与 A2、A3，A1 按真实数据中存在的类型评分。

A1、A3 从读取特征开始就排除最后 1000 列，包括 A3 的填补、ANOVA、PCA、miLISI 和 UMAP。A3 及 A2 辅助分布共享原始区间的参照中位数与 ANOVA 结果。小数据下自动降低实际区域数和 PCA 维数，并记录实际值。合成数值不截断；缺失比例及超出 0–1 的比例单独报告。

## 输出

| 文件 | 内容 |
| --- | --- |
| A1_A2_A3_summary.csv | 每种合成方法一行的总表 |
| A1_summary.csv、A1_type_details.csv | A1 加权指标和逐类型明细 |
| A2_pearson_summary.csv、A2_duplicate_correlations.csv | 1000 对 Pearson 汇总及明细，含真实参考 |
| A2_natural_Spearman_summary.csv、A2_Spearman_编号.npy | 原始区间 Spearman 汇总和完整向量；编号映射写在汇总表 |
| A2_natural_selected_regions.csv、A3_selected_regions.csv | 实际入选的原始区域和 ANOVA 分数 |
| A3_miLISI.csv、A3_type_details.csv、A3_miLISI_cells.csv | A3 总分、各类型中位数和逐细胞来源混合指数 |
| A3_PCA_coordinates.csv | 真实参照、评估真实和各方法在同一 PCA 中的坐标 |
| A3_joint_coordinates.csv | 逐方法真实／合成联合 UMAP 坐标 |
| selected_cells.csv、region_pairs.csv | 各指标共用的抽样细胞和经校验的复制映射 |
| input_diagnostics.csv、reference_natural_medians.npy | 原始／复制区间数值诊断和原始区间填补中位数 |
| figures/ | 四组 PNG/PDF：多区域边缘分布、单区域按细胞类型分布、A2 小提琴＋正式均值柱状图、A3 真实／合成联合 UMAP |
| A1_plot_regions.csv、A1_plot_histograms.csv、A1_type_histograms.csv | A1 展示证据清单；两张分布图的逐方法分箱概率、有限值数和越界比例 |
| run_summary.json | 完成或失败状态、参照模式、参数、依赖版本、实际维数和输入文件记录 |

输出目录须独立，不能包含任何输入文件（包括输入配对表），防止重跑时覆盖输入。请以 run_summary.json 的“状态”为准；失败运行不会被标记为完成。大数据的 ANOVA、PCA 和 UMAP 需要时间，流程会打印阶段进度。`read_in_memory=false`以只读backed方式打开；内存充足时可设为true，一次载入以避免反复解压。两种方式的指标相同，正常完成或发生异常均关闭读取对象。

## A1/A2 图形样式

A1 多区域图的行为区域、列为方法；单区域图的行为方法、列为 Subclass，每格叠加该类型的真实和合成分布。横轴统一为 0–1、30 个等宽分箱；灰色为真实数据，方法颜色在各图中保持一致。每个分布按该区域和分组的全部有限原值归一化，超出 0–1 的部分在图内单独标注比例，不压入边缘分箱。缺失值不填补，分组图标注实际有限观测数。该展示范围不改变原数据或评估指标。

A2 复制区间图左侧展示有效 Pearson 分布及中位数，真实参考恒定为 1 时显示短横线；右侧直接使用正式汇总中的有效配对 Pearson 均值，标注四位小数和有效对数。A3 仅画当前真实／合成联合 UMAP，图上 miLISI 使用正式 PCA 空间得分。PNG 以 300 dpi 保存，PDF 保留可选取的文字。

## Liu2023 默认展示证据

清单 [a1_marker_regions_liu2023.csv](a1_marker_regions_liu2023.csv) 使用 [Liu 等，Nature 2023](https://doi.org/10.1038/s41586-023-06805-y) 图 1d 的 Tle4 示例区域；原论文 [交互视图](https://tinyurl.com/fig1d) 明确给出 mm10 的 chr19:14,300,000–14,600,000 和 chr19:15,100,000–15,300,000 两个浏览范围，并对照 L6 CT 与 Pvalb。原始 GEO 注释也确认 mm10。当前 H5AD 未保存 mCG/mCH context 元数据，因此图内只写 Methylation level。

将两个文献浏览范围分别平分，共四段；在全部 50 个 10 kb 候选窗口上只使用本轮 4,740 个真实训练细胞，纳入训练总数至少 25 的 19 个 Subclass。各类有效观测至少 10 且覆盖率至少 50%，Kruskal–Wallis 后在全部 50 个候选上 BH 校正 q<0.01，并要求 Pvalb−L6CT 均值差至少 0.2。每段按该训练均值差最大选一个，随后四者中差值最大的窗口用于单区域图；没有读取合成或留出表现参与筛选。五个展示类型固定为 L6 CT、L6b、L2/3 IT、Pvalb、Oligo。

入选窗口为 chr19_14350001_14360000、chr19_14580001_14590000、chr19_15150001_15160000、chr19_15210001_15220000，坐标名按 1-based inclusive 解释；与文献浏览范围比较时将起点减一转换为 0-based half-open。训练 Pvalb−L6CT 均值差分别为 0.515、0.703、0.344、0.387，BH q 均小于 1e-81。极小 P/Q 用 log10 保存，避免浮点下溢被误写成 P=0。

文献依据精确到 Tle4 示例浏览区域；上述 10 kb 窗口的显著性来自本轮训练数据，并非声称整个 10 kb 是原文发表的精确 DMR。单细胞分组检验也不替代生物学重复层面的推断。完整候选统计、选择口径和本轮图表在 `output/Liu2023全量训练_20261107/figures_markers/`。

## 验证

```powershell
conda activate simscDNAm
python -m pytest evaluate/test_evaluation.py evaluate/test_plotting.py evaluate/test_notebook.py -q
```

单元测试核对 SciPy 数值、缺失值、类型缺失、严格区域轴和抽样。端到端测试在 Jupyter 内核中构造包含 1000 个复制区间的 H5AD，分别验证独立训练参照和仅真实／合成输入两种模式：修改合成复制尾部后 A1、PCA、miLISI 保持一致，A2 显著变化；同时检查图表、输入未变及正常／异常时的文件关闭。

本轮四方法的完整比较和PNG/PDF见[results/liu2023](../results/liu2023/README.md)。默认配置与主模型、baseline的默认输出目录相配；更换数据时需同时修改输入和marker清单。

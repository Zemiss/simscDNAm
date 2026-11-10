# simscDNAm主模型

唯一正式入口为[train.ipynb](train.ipynb)。完整方法包含CVAE及条件潜空间扩散、训练缺失掩码、联合秩映射三个阶段。科学流程和映射函数在notebook内；网络、输入与底层生成保留在同目录模块。

## 配置

- `DATASETS`：每个数据集的`training_h5ad`、`target_h5ad`、`region_pairs_csv`。真实训练输入须已扩维；目标支持原始轴或完整扩维轴，矩阵仅用于最终独立评估，其Subclass提供生成配额。
- `RUN_ROOT`：新的输出父目录；`RUN_DATASETS`选择要运行的数据集。
- `CONFIG`、`CASES`：保持原网络和四个内部候选，内部重构／多样性验收后按内部miLISI选择；外层真实值不参与选模。
- `PRETRAINED_RUNS={}`：从头训练。复用示例：`{"liu2023/full": Path("已有训练目录")}`。

复用目录须含`config.json`、`internal_reference.npz`、`region_pairs.csv`、`CVAE_training_mask/{model.pt,data.h5ad}`、`work/{values.float32.npy,input_cell_audit.csv}`。入口逐块核对当前训练值、NaN、细胞ID、Subclass、区域轴与旧训练缓存；目标类型配额变化时拒绝复用生成矩阵。最终真实留出从当前配置的目标文件重建。

## mask与映射

mask donor只从同Subclass内部fit细胞抽取，保留原生成的有限值。映射随后会改变这些有限值。

1. 按Subclass分别将fit真实值转为平均秩正态分数，缺失中心填0并标准化。
2. PCA学习最多32个联合因子；同一回归计算所有区域的载荷，不根据复制配对表特殊处理任何列。
3. 当前生成批次转秩、投影至真实联合因子，白化后按真实因子协方差着色，再重构区域分数。固定投影强度1，原CVAE的列内排序并非保持约束。
4. 按重构分数在同类有限生成细胞中的分位点，逐区域取fit真实经验分位数。NaN位置保持，不额外复制合成列。

实际维数受fit和内部验证类型样本量限制。fit不足5个或内部验证不足3个的类型保持原值；应用批次同类型不足3个也保持原值。映射结果依赖同类型整批细胞，不能逐细胞单独调用后期待同一结果。

此实现借鉴[Iman与Conover的边缘／秩相关分离思路](https://doi.org/10.1080/03610918208812265)，采用高维低秩近似。高复制Pearson反映本后处理约束；它不是对原论文的逐行复现，也不保证任意数据都达到本轮指标。

## 输出

每个`RUN_ROOT/<数据集>/`包含：

- `simscDNAm/data.h5ad`：CVAE＋mask＋对齐的正式结果，保留完整区域轴和mask donor。
- `simscDNAm/model.pt`：原CVAE checkpoint原样副本。
- `simscDNAm/alignment/`：联合因子、经验分位数、拟合身份和类型明细；与checkpoint共同构成完整生成流程。
- `simscDNAm/alignment_summary.json`、`completed.json`：映射及完成记录。
- `HeldoutReal/data.h5ad`、`region_pairs.csv`：本轮当前目标的扩维真实留出及固定配对表，供独立评估。
- `CVAE_training_mask/`及内部候选文件：新训练时保留的中间产物；复用模式直接读取源目录中的已完成mask结果。

训练输入和源码身份变化时拒绝继续原目录；相同身份可恢复。映射参数、控制JSON和导出checkpoint用临时文件完成后原子替换。训练输出不自动运行baseline；四方法比较统一从`evaluate/A1A2A3.ipynb`执行。

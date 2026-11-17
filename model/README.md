# 主模型

`src/model.py` 实现条件 VAE、FiLM 潜空间去噪器、cosine 日程、v 预测损失和 DDIM；`src/config.py` 保存超参数。科学流程函数在 notebook 内实现。

[script/train.ipynb](script/train.ipynb) 从预处理输出读取训练集，保留原四个内部候选和内部验证选模；真实留出矩阵不参与训练或选模。修改 `DATASETS`、`RUN_DATASETS`、`RUN_ROOT` 后运行。产物写入 `outputs/model/<数据集>/`，正式模型在其中 `simscDNAm/`：`model.pt`、`alignment/` 与 `data.h5ad`。

[script/inference.ipynb](script/inference.ipynb) 读取完整训练目录、同一训练 H5AD 和目标类型配额。先核对训练缓存，再执行 DDIM → 同 Subclass 内部 fit 掩码 → 已拟合联合秩映射。结果写入 `outputs/inference/<数据集>/simscDNAm/`。需要同时保留 checkpoint、映射参数、`work/` 训练缓存、细胞审计和 `internal_reference.npz`；单独权重不能还原完整方法。

映射最多 32 个联合因子、固定投影强度 1。fit 不足 5 个或内部验证不足 3 个的类型沿用原生成值；新批次同类型不足 3 个也沿用原值。映射依赖同类型整批细胞。复制区域与自然区域使用同一规则，不单独重写。

使用 train.ipynb 的 PRETRAINED_RUNS 复用训练时，新目录的完成记录会保存源训练缓存目录；独立推理自动读取该来源，仍逐值核验当前训练 H5AD。请保留源目录的 work/ 与 internal_reference.npz。

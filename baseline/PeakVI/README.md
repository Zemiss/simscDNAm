# PeakVI 官方基线

在 `simscDNAm` 环境中运行 [`基线_PeakVI.ipynb`](../../notebooks/基线_PeakVI.ipynb)。固定依赖为 `scvi-tools==1.4.2`，不自动安装依赖或更改官方模型代码。原有环境已验证的组合为 Python 3.11、PyTorch 2.5.1、AnnData 0.12.16；正式服务器以运行报告记录的实际版本为准。

训练只用共用预处理生成的填补训练矩阵，直接调用默认 `PEAKVI.setup_anndata(adata)`、`PEAKVI(adata)` 和 `model.train()`；设备参数除外。最多 500 轮、早停、网络规模、优化器及内部训练验证划分保持官方默认。`Subclass` 只用于生成时匹配训练供体，不作为模型条件。全部区域均参与训练与输出。

官方重建目标为 `(x > 0)`，故该基线测量 scATAC 概率模型在当前输入上的表现，不拟合连续甲基化比例损失。输入比例原值保留，不自行转计数或改写损失；官方非整数计数提示不被隐藏。

官方 `PEAKVAE` 无可用的 `sample`，适配器按目标 `Subclass` 有放回抽取同类别的训练供体，调用 `get_normalized_accessibility(indices=供体行, use_z_mean=False, return_numpy=True)`。输出是随机后验解码的可及性概率，不是先验样本或伯努利计数。生成后仅 `clip(0, 1)` 并存储为 float32；不增加缺失值、不做测试分布对齐。目标文件只读取 `obs`。

通过 `BASELINE_TRAINING`、`BASELINE_TARGET`、`BASELINE_OUTPUT`、`BASELINE_SEED`、`BASELINE_DEVICE` 设置路径、种子和设备。服务器输出根目录为 `/home/sccasimp/data/xb/outputs/simscDNAm`，输出包含 `data.h5ad`、`donor_cells.csv`、`verification.json` 和 `model/`。共用填补缓存位于训练文件同级的 `imputed_baselines/`，该目录同样必须位于允许的输出根目录。27–31 万区域会令官方默认网络很大，正式运行须具备相应内存和显存；适配器不会自动缩小模型。

服务端命令行入口会在启动 Notebook 内核时加载 `baseline/PeakVI/.runtime/packages` 中的固定版本依赖。服务器保留共享环境现有的 PyTorch，PeakVI 1.4.2 及其兼容版本依赖单独安装在该目录。

接口检查：`python -m unittest discover -s checks -p test_peakvi_interface.py -v`。此检查隔离神经网络训练，并真实读写小型 h5ad；不代表真实数据训练或评估已经完成。

2026-11-18 已用独立人工数据验证官方完整流程：24 个训练细胞、2000 个区域（1000 个原区域及其复制）、8% 缺失，训练中位数填补后按官方默认训练完整 500 轮，再生成 4×2000 个后验概率。使用本地 RTX 4060 Laptop GPU，训练及写出用时约 9.5 秒，输出范围为 0.45836–0.91561，全部有限，目标顺序、区域轴及同类别供体检查通过；临时文件已删除。该检查验证 API 可运行，不推断真实全区域数据的资源需求或模型表现。

官方依据：[PEAKVI API](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.PEAKVI.html)、[1.4.2 模型源码](https://github.com/scverse/scvi-tools/blob/1.4.2/src/scvi/model/_peakvi.py)、[1.4.2 模块源码](https://github.com/scverse/scvi-tools/blob/1.4.2/src/scvi/module/_peakvae.py)。

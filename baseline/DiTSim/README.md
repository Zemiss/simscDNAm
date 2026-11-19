# DiTSim 官方基线

训练与生成实现在 [基线_DiTSim.ipynb](../../notebooks/基线_DiTSim.ipynb)，使用 `simscDNAm` 内核。命令行入口由同目录的 `inference.py` 调用 notebook。

固定官方提交：[52231d81ee7d98c6222ad51539624e1204ebf6a9](https://github.com/BioX-NKU/DiTSim/tree/52231d81ee7d98c6222ad51539624e1204ebf6a9)。`sources.json` 固定逐文件 SHA256；`vendor/` 受项目 gitignore 保护，只缓存九个必要源码／环境／许可证文件，首次运行缺少时从固定提交下载并校验。

仅使用统一预处理的训练逐区域中位数填补。正式训练保留完整细胞和区域轴，包括末尾 1000 个复制区域。目标文件只提供区域轴、细胞 ID 和 `Subclass` 配额。官方 loader 对稀疏格式使用 `mat.A`，因此输入只转为稠密 `numpy.ndarray float32`；官方模型、损失、优化器、训练器和采样实现不修改。实际标签数传入两个模型的 `num_classes`，使条件嵌入维度与数据一致。

AE 使用官方 quick start 参数：潜变量 256、隐藏层 [1024,1024]、条件嵌入 128、`reshape=True`、`cfg=True`，200 轮、`end_loss=0.01`、批大小 32、学习率 5e-4、weight decay 0.01、余弦损失比例 0、每 10 轮保存。官方阈值分支仅保存而不 break，保留此行为。

DiT 使用输入 256、patch 2、hidden 768、depth 4、8 个 heads，2400 轮、批大小 32、学习率 1e-4、weight decay 0.01、每 100 轮保存。2400 来自官方 `quick_start.py`，README 中的 1600 轮示例未采用。生成调用官方 `sampling(sample_dict, sampler='ddim', acc_rate=10)`；其中已有 `clamp(min=0)`，随后仅做统一 `clip(0,1)`，拒绝 NaN／Inf。

生成文件恢复完整训练 `var` 与列名，保持官方采样的行顺序，核对全部 Subclass 配额，赋予独立 synthetic 细胞 ID。不根据真实值调整生成结果，不强制复制列相等。

依赖需在运行前准备；notebook 不自动安装。必要直接依赖为 torch、与 torch 匹配的 torchvision、timm、numpy、pandas、scipy、anndata、scanpy、tqdm。原始 `ditsim.yml` 的参考版本为 torch 2.4.1、timm 1.0.15、numpy 1.24.3、pandas 2.0.3、scipy 1.10.1、anndata 0.9.2、scanpy 1.9.8、tqdm 4.66.5；其中 torchvision 0.20.0 与 torch 2.4.1 的官方组合不匹配，应以当前 torch 的官方配套 torchvision 为准。现有 torch 2.5.1 对应 torchvision 0.20.1。导入模型时 timm 还会间接依赖 torchvision；不需要 tool.py 中的 episcanpy。若预先把依赖安装到本目录 `.runtime/packages/`，`load_official()` 会局部加入该私有目录，不修改全局环境。PyTorch 2.6+ 通过环境变量 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 兼容官方 checkpoint 中的 NumPy 对象，不改 vendor 中的 torch.load。

环境变量：`BASELINE_TRAINING`、`BASELINE_TARGET`、`BASELINE_OUTPUT`、`BASELINE_SEED`、`BASELINE_DEVICE`。默认完整正式训练；`BASELINE_PILOT=1` 才运行小规模一轮接口试跑，写入独立 `pilot/` 并标为非正式。共享填补缓存位于训练文件旁的 `imputed_baselines/`；服务器输出受项目路径检查限制。

完整输入维度约 273k–310k 时，官方 AE 的输入／输出全连接层会产生约 5.6–6.4 亿参数，AdamW 及训练激活需要较大显存；此实现不缩减模型或特征以绕过资源限制。若显存不足，应保留失败信息后由用户决定运行资源。

2026-11-18 已完成一次真实官方模型的人工数据 pilot 闭环：在本地 RTX 4060 Laptop GPU 上，使用官方 AE 隐藏层 [1024,1024]、latent 256 和 DiT hidden 768、depth 4、heads 8，各训练一轮，再调用官方 DDIM（acc_rate=10）。人工连续输入为 6×272，既有 pilot 规则选取 4×256 训练，最终生成 3×256，Subclass 配额为 Pilot_A=2、Pilot_B=1，总耗时 27.80 秒。完整恢复的区域元数据一致，生成结果逐元素等于官方采样结果的 clip(0,1)；官方采样结果最大值 2.0113，45 个大于一的值被裁剪。九个 vendor 文件的 SHA256 均未改变。验证环境为 torch 2.5.1、torchvision 0.20.1、私有目录中的 timm 1.0.15、numpy 1.26.4、pandas 2.3.3、scipy 1.17.1、anndata 0.12.16、scanpy 1.11.5。此检查只验证真实模型接口，不涉及正式数据，不构成正式实验结果；本地日志与 JSON 证据缓存在 `.runtime/real_pilot_20261118/`，正式 200／2400 轮实验尚未执行。

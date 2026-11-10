# 三种全维 baseline

本目录保留 FittedMarginal、simATAC、simCAS。算法和生成函数直接位于各 notebook 内，不依赖旧实验notebook。

## 输入和输出

使用 Jupyter 的 `simscDNAm` 内核，从上到下运行相应 notebook。

- `TRAINING_PATH`：真实训练 H5AD，全部原始区间后已经追加 1000 个复制区间。文件需含 `obs["Subclass"]`；细胞 ID、区域名必须唯一。
- 末尾列按 `原区域名__duplicate_0000` 至 `原区域名__duplicate_0999` 命名。所有细胞的复制值及 NaN 位置必须与对应原列完全一致。稀疏隐式零按零处理。
- `TARGET_OBS_PATH`：目标类别配额来源，只读取 `obs["Subclass"]`，不读取该文件的矩阵，也不重新划分训练集。沿用种子 42 打乱标签的生成顺序。
- 输出为 `OUTPUT_ROOT/<方法>/data.h5ad`，包含全部原始列及全部复制列。生成后不追加或替换复制列。另保存 `region_pairs.csv` 和 `verification.json`。
- FittedMarginal 保留观测比例和缺失值；simATAC、simCAS 保留原生计数或分数，允许大于 1。

当前默认输入是历史扩维训练产物：

```text
/home/sccasimp/data/xb/outputs/simscDNAm/liu2023/duplicate_regions_1000_seed42/f434e3385d87/methods/Truth/data.h5ad
```

该输入包含 4740 个训练细胞和 273544 个区间，其中 272544 个是原始区间，1000 个是种子 42 无放回随机选区后追加的副本。历史目标配额来自 1190 个留出细胞。新产物单独写入：

```text
/home/sccasimp/data/xb/outputs/simscDNAm/baseline_v1_20261110/
```

## 两种运行方式

`RUN_MODE = "run"` 从头执行模型。更换训练数据时，应同时修改目标配额来源及输出目录，将 `REFERENCE_PATH = None`；若已有该新数据的旧 notebook 结果，可指定该结果继续执行完整一致性核对。

`RUN_MODE = "replay"` 用于检查本次整理是否保持已有结果：先逐元素核对历史训练矩阵及目标标签，再重新组装 simATAC 的逐类中间输出，或从 simCAS 官方原始生成矩阵按原种子重新抽样。simCAS 还会逐元素核对中位数填补后的完整模型输入，并核对生成参数。该模式不重新拟合官方模型，也不复制历史最终输出文件。

三个正式入口均默认使用 `run`，`REFERENCE_PATH=None`。simCAS 历史完整拟合约需 5.2 小时；切换为 `run` 后会实际调用官方生成函数。simATAC 需要现有 R4 和 simATAC 包；simCAS 需要服务器 `third_party/simCAS`、R4 及已安装的 Python 依赖。

## 验证口径

1. 检查训练输入中的全部复制值、缺失位置和完整区域轴。
2. 在服务器Jupyter顺序执行；可选replay仅适用于已经逐值核验的对应训练输入和历史中间产物。
3. 与历史扩维结果逐元素比较全部 1190 × 273544 个数值及 NaN 位置，同时核对细胞顺序、Subclass、区域名称和顺序。
4. 保存执行输出和一致性报告；任何不一致均报错，不使用放宽容差或修改生成列的方式使结果通过。

本地检查：在 `simscDNAm` conda 环境运行 `python -m pytest baseline/test_notebooks.py -q`。测试直接读取 notebook 中的校验函数，覆盖密集/稀疏输入、零值、缺失位置、错误复制映射，以及原始区间和末尾区间的输出改动。

移植时已核对服务器保存的对应原实现；旧本地实验文件已移入项目外清理备份。FittedMarginal 保持区间块大小 512 及原随机流，simATAC 保留原实验的小类别估计回退，simCAS 保留 `rand_seed=2022`、`lib_simu="real"` 等参数。三个baseline的算法代码未修改。

## 与simscDNAm统一比较

正式比较在`evaluate/A1A2A3.ipynb`中执行，方法名固定使用`simscDNAm`、`FittedMarginal`、`simATAC`、`simCAS`。本仓库的Liu2023结果复用既有三个baseline全维产物并统一重新评估，未重新拟合数小时的simCAS。

首次运行simATAC需设置可用的Rscript与`R_LIBRARY`；未安装包时可将算法单元中的`INSTALL_MISSING=True`，安装官方`bowang-lab/simATAC`及依赖到指定库。simCAS需配置`OFFICIAL_SIMCAS_ROOT`和`R_HOME`，其官方Python/R依赖须另外安装，根Python依赖清单不代替官方完整运行环境。

simATAC、simCAS的X保留原生计数／分数，可能大于1；本协议不做输出截断或事后归一化。两者与甲基化比例的数值域差异会影响1-WD，结果解释须结合`input_diagnostics.csv`。

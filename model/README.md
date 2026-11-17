# 主模型

`script/` 仅提供命令行入口，具体函数放在 `src/`。

| 模块 | 职责 |
|---|---|
| `src/model.py` | 条件 VAE、FiLM 去噪器、cosine 日程、v 预测损失和 DDIM |
| `src/config.py` | TrainingConfig 超参数及 JSON 读取 |
| `src/training.py` | 内部参照、候选训练和选模诊断 |
| `src/generation.py` | 按完整区域轴分块生成与 H5AD 导出 |
| `src/postprocessing.py` | 同类型训练缺失掩码、联合秩映射及经验分位数对齐 |
| `src/pipeline.py` | 训练、内部选模、恢复与正式产物导出 |
| `src/inference.py` | 复用 checkpoint、训练缓存与映射的独立推理 |

## 训练

```bash
python -m model.script.train --dataset liu2023/full --device cuda:0
python -m model.script.train --help
```

默认读取共用预处理目录，使用原四个内部候选；真实留出矩阵不参与训练或选模。`--training`、`--target`、`--pairs` 指定输入，`--output-root` 指定训练输出父目录。`--config` 可读取 JSON 中的 TrainingConfig 字段，省略字段沿用配置类默认值。

产物写入 `outputs/model/<数据集>/`，正式模型位于其中 `simscDNAm/`，包含 `model.pt`、`alignment/` 与 `data.h5ad`。输入、配置或源码签名变化时拒绝恢复到原目录，请使用新的输出目录。

## 推理与训练复用

```bash
python -m model.script.inference --dataset liu2023/full --device cuda:0
python -m model.script.inference --trained-run 训练目录 --training 原训练.h5ad --target 目标.h5ad --output 新目录
python -m model.script.train --dataset liu2023/full --reuse 既有训练目录 --output-root 新训练父目录
```

独立推理先逐值核对原训练 H5AD 与训练缓存，再执行 DDIM → 同 Subclass 内部 fit 掩码 → 已拟合联合秩映射。默认输出 `outputs/inference/<数据集>/simscDNAm/`，非空目录拒绝覆盖。

需要同时保留 checkpoint、映射参数、`work/` 训练缓存、细胞审计和 `internal_reference.npz`。训练复用会在新目录记录源缓存位置，独立推理自动读取该来源；请保留源训练缓存。旧版本的 v 预测权重及映射仍可加载，继续训练时的源码签名基于当前 Python 文件。

映射最多 32 个联合因子，固定投影强度 1。fit 不足 5 个或内部验证不足 3 个的类型沿用原生成值；新批次同类型不足 3 个也沿用原值。映射依赖同类型整批细胞，复制区域与自然区域走同一规则。

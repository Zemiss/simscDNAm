# Liu2023：新增 STADiffuser 非空间 baseline

既有四方法得分取自仓库冻结的30近邻结果，本次只新增训练和评估STADiffuser。
原始数据、4740/1190划分、1000个复制区域及A1/A2/A3协议保持一致；留出ID哈希在训练前核对。
STADiffuser保留官方网络和MSE，仅使用训练中位数填补、自环图及无空间条件的Subclass生成；输出无截断、mask或秩映射。

| 方法 | A1 1-WD↓ | A1 MSE↓ | A1 Pearson↑ | A2 Pearson↑ | A3 miLISI↑ |
|---|---:|---:|---:|---:|---:|
| simscDNAm | 0.0359395 | 1.36804e-05 | 0.921216 | 0.999896 | 1.66503 |
| FittedMarginal | 0.0457714 | 1.16342e-05 | 0.876301 | 0.0687495 | 1.53475 |
| simATAC | 0.596839 | 0.000628933 | 0.385547 | 0.000442474 | 1.17401 |
| simCAS | 0.795515 | 0.0764871 | 0.613972 | 0.277128 | 1.53773 |
| STADiffuser | 0.166957 | 0.00131529 | 0.860625 | 0.996416 | 1.06712 |

训练、生成及评估日志分别见 full/、full_execution.log、evaluation_execution.log。
STADiffuser图表位于evaluation/figures/，各方法UMAP分别拟合，不能跨面板直接比较坐标。

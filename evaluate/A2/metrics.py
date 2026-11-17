"""A2 评估指标，沿用原正式定义。"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata


def paired_pearson(original, copied):
    """逐区间对忽略非有限细胞；少于三个共同观测或任一列恒定时记缺失。"""
    if original.shape != copied.shape:
        raise ValueError("原区间和复制区间矩阵形状不一致")
    records = []
    for left, right in zip(original.T, copied.T):
        finite = np.isfinite(left) & np.isfinite(right)
        x, y = left[finite], right[finite]
        value = float(pearsonr(x, y).statistic) if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
        records.append({"有效细胞数": len(x), "Pearson": value})
    return pd.DataFrame(records)


def spearman_vector(values):
    ranks = rankdata(values, axis=0, method="average")
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.corrcoef(ranks, rowvar=False)
    return correlations[np.triu_indices(values.shape[1], k=1)]

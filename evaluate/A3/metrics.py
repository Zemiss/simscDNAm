"""A3 评估指标，沿用原正式定义。"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def milisi(reference_rep, reference_labels, candidate_rep, candidate_labels, n_neighbors=30):
    """PCA 空间中计算来源混合，按真实类型取中位数后做等权平均。"""
    combined = np.vstack([reference_rep, candidate_rep])
    if len(combined) <= n_neighbors or n_neighbors < 1:
        raise ValueError("合并细胞数必须大于 miLISI 的近邻数")
    source = np.r_[np.repeat("真实", len(reference_rep)), np.repeat("合成", len(candidate_rep))]
    labels = np.r_[reference_labels, candidate_labels]
    raw = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(combined).kneighbors(combined, return_distance=False)
    indices = np.array([row[row != index][:n_neighbors] for index, row in enumerate(raw)])
    p_real = (indices < len(reference_rep)).mean(axis=1)
    values = 1 / (p_real ** 2 + (1 - p_real) ** 2)
    cells = pd.DataFrame({"来源": source, "细胞类型": labels, "miLISI": values})
    groups = cells.groupby("细胞类型", sort=True)["miLISI"].median().rename("miLISI中位数").reset_index()
    return float(groups["miLISI中位数"].mean()), cells, groups

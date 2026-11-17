"""A1 评估指标，沿用原正式定义。"""
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.stats import pearsonr
from data.evaluation import dense_block, read_labels


@lru_cache(maxsize=128)
def quantile_transport_plan(reference_count, candidate_count):
    edges = np.unique(np.r_[np.arange(reference_count + 1) / reference_count,
                            np.arange(candidate_count + 1) / candidate_count])
    midpoints = (edges[:-1] + edges[1:]) / 2
    left = np.minimum((midpoints * reference_count).astype(int), reference_count - 1)
    right = np.minimum((midpoints * candidate_count).astype(int), candidate_count - 1)
    return np.diff(edges), left, right


def column_wasserstein_1(reference, candidate):
    reference_counts = np.isfinite(reference).sum(axis=0)
    candidate_counts = np.isfinite(candidate).sum(axis=0)
    left = np.sort(np.where(np.isfinite(reference), reference, np.inf), axis=0)
    right = np.sort(np.where(np.isfinite(candidate), candidate, np.inf), axis=0)
    distances = np.full(reference.shape[1], np.nan)
    for n_left, n_right in np.unique(np.column_stack([reference_counts, candidate_counts]), axis=0):
        if n_left == 0 or n_right == 0:
            continue
        columns = np.flatnonzero((reference_counts == n_left) & (candidate_counts == n_right))
        weights, i_left, i_right = quantile_transport_plan(int(n_left), int(n_right))
        distances[columns] = (np.abs(left[i_left[:, None], columns] - right[i_right[:, None], columns])
                              * weights[:, None]).sum(axis=0)
    return distances


def finite_means(values, axis):
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    return np.divide(np.where(finite, values, 0).sum(axis=axis), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def a1_metrics(reference, candidate, reference_rows, candidate_rows, natural_count,
               label_key="Subclass", chunk_size=512):
    """按真实类型比例加权：逐区间 1-WD、类型细胞均值 MSE、区间均值 Pearson。"""
    labels = [read_labels(item, rows, label_key) for item, rows in
              [(reference, reference_rows), (candidate, candidate_rows)]]
    levels, counts = np.unique(labels[0], return_counts=True)
    state = {level: {"distances": [], "means": [[], []]} for level in levels}
    cell_sums = [np.zeros(len(reference_rows)), np.zeros(len(candidate_rows))]
    cell_counts = [np.zeros(len(reference_rows)), np.zeros(len(candidate_rows))]
    for start in range(0, natural_count, chunk_size):
        columns = slice(start, min(start + chunk_size, natural_count))
        blocks = [dense_block(reference, reference_rows, columns), dense_block(candidate, candidate_rows, columns)]
        for index, block in enumerate(blocks):
            finite = np.isfinite(block)
            cell_sums[index] += np.where(finite, block, 0).sum(axis=1)
            cell_counts[index] += finite.sum(axis=1)
        for level in levels:
            groups = [block[label == level] for block, label in zip(blocks, labels)]
            distances = column_wasserstein_1(*groups)
            state[level]["distances"].append(distances[np.isfinite(distances)])
            for index, group in enumerate(groups):
                state[level]["means"][index].append(finite_means(group, axis=0))
    cell_means = [np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
                  for total, count in zip(cell_sums, cell_counts)]
    details = []
    for level, count in zip(levels, counts):
        item = state[level]
        distances = np.concatenate(item["distances"])
        means = [np.concatenate(parts) for parts in item["means"]]
        common = np.isfinite(means[0]) & np.isfinite(means[1])
        x, y = means[0][common], means[1][common]
        pearson = float(pearsonr(x, y).statistic) if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
        type_means = []
        for values, label in zip(cell_means, labels):
            values = values[(label == level) & np.isfinite(values)]
            type_means.append(values.mean() if len(values) else np.nan)
        details.append({"细胞类型": level, "真实类型权重": count / len(reference_rows),
                        "真实细胞数": int(count), "合成细胞数": int((labels[1] == level).sum()),
                        "1-WD": distances.mean() if distances.size else np.nan,
                        "MSE": (type_means[1] - type_means[0]) ** 2, "Pearson": pearson,
                        "有效区域数": int(distances.size), "真实类型均值": type_means[0],
                        "合成类型均值": type_means[1]})
    details = pd.DataFrame(details)
    weights = details["真实类型权重"].to_numpy()
    summary = {}
    for column in ["1-WD", "MSE", "Pearson"]:
        values = details[column].to_numpy()
        summary["加权 " + column] = float(np.sum(weights * values)) if np.isfinite(values).all() else np.nan
    valid = np.isfinite(details[["1-WD", "MSE", "Pearson"]]).all(axis=1)
    summary["有效细胞类型权重"] = float(weights[valid].sum())
    return summary, details

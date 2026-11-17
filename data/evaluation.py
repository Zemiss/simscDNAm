"""统一评估的数据读取、复制配对和真实参照预处理。"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


def validate_region_axis(real, item):
    """指标依赖严格对应的区域轴，禁止静默取交集或重排末尾复制区间。"""
    if not item.var_names.equals(real.var_names):
        raise ValueError("输入数据的区域名称或顺序与真实数据不一致")


def sample_rows(item, count, seed=42):
    if count < 3 or item.n_obs < count:
        raise ValueError(f"评估至少需要 3 个细胞，当前数据不能无放回抽取 {count} 个细胞")
    if item.n_obs == count:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(item.n_obs, count, replace=False))


def dense_block(item, rows, columns):
    """支持原始、稀疏及只读 backed H5AD；保留 NaN 和原始数值。"""
    rows = np.arange(item.n_obs)[rows]
    columns = np.arange(item.n_vars)[columns]
    if not len(rows) or not len(columns):
        return np.empty((len(rows), len(columns)), dtype=np.float64)
    if not item.isbacked:
        matrix = item.X[np.ix_(rows, columns)]
        return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float64)
    # HDF5 只使用一个非连续轴；按行分块，避免一次读取全数据。
    result = np.empty((len(rows), len(columns)), dtype=np.float64)
    start, stop = int(columns.min()), int(columns.max()) + 1
    for offset in range(0, len(rows), 32):
        selected = rows[offset:offset + 32]
        matrix = item[selected, start:stop].X
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        result[offset:offset + len(selected)] = np.asarray(matrix)[:, columns - start]
    return result


def read_labels(item, rows, label_key="Subclass"):
    """只读取保存的真实类型标签；不构建、推断或替换为聚类标签。"""
    if label_key not in item.obs or item.obs[label_key].iloc[rows].isna().any():
        raise ValueError(f"数据缺少完整的真实类型标签 obs[{label_key!r}]")
    labels = item.obs[label_key].iloc[rows].astype(str).to_numpy()
    if np.any(np.char.strip(labels.astype(str)) == ""):
        raise ValueError("真实类型标签不能为空")
    return labels


def region_pairs(real, duplicate_count=1000, pair_path=None):
    """验证原区间到末尾复制区间的映射；不凭随机种子猜测对应关系。"""
    natural_count = real.n_vars - duplicate_count
    if duplicate_count <= 0 or natural_count < duplicate_count or not real.var_names.is_unique:
        raise ValueError("区域名称必须唯一，原始区域数必须不少于复制区域数")
    original_names = real.var_names[:natural_count]
    copied_names = real.var_names[natural_count:]
    if pair_path is None:
        indices = []
        for name in copied_names:
            match = re.fullmatch(r"(.+)__duplicate_\d+", str(name))
            if match is None or match.group(1) not in original_names:
                raise ValueError("无法从复制区间名称确定来源，请提供 region_pairs.csv")
            indices.append(original_names.get_loc(match.group(1)))
        pairs = pd.DataFrame({"原region索引": indices,
                              "复制region索引": np.arange(natural_count, real.n_vars)})
    else:
        pairs = pd.read_csv(Path(pair_path)).rename(columns={
            "original_index": "原region索引", "duplicate_index": "复制region索引",
            "original_region": "原region", "duplicate_region": "复制region",
        })
    required = ["原region索引", "复制region索引"]
    if not set(required).issubset(pairs.columns) or len(pairs) != duplicate_count:
        raise ValueError(f"复制对应表必须包含原region索引、复制region索引，共 {duplicate_count} 行")
    pairs = pairs.copy()
    for column in required:
        values = pd.to_numeric(pairs[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("复制对应表的区域索引必须为从 0 开始的整数")
        pairs[column] = values.astype(np.int64)
    pairs = pairs.sort_values("复制region索引").reset_index(drop=True)
    if not np.array_equal(pairs["复制region索引"], np.arange(natural_count, real.n_vars)):
        raise ValueError(f"复制对应表必须恰好覆盖最后 {duplicate_count} 个区间")
    indices = pairs["原region索引"].to_numpy()
    if np.any(indices < 0) or np.any(indices >= natural_count) or len(np.unique(indices)) != duplicate_count:
        raise ValueError("复制来源必须是互不重复的原始区间索引")
    for column, expected in [("原region", original_names[indices]), ("复制region", copied_names)]:
        if column in pairs and not np.array_equal(pairs[column].astype(str), expected.astype(str)):
            raise ValueError(f"复制对应表的 {column} 名称与实际区域轴不一致")
        pairs[column] = expected.to_numpy(dtype=str)
    return pairs[["原region索引", "复制region索引", "原region", "复制region"]]


def validate_real_duplicates(real, pairs):
    """真实数据的复制值及缺失位置必须与来源相同；合成复制列不做此限制。"""
    for start in range(0, real.n_obs, 64):
        rows = np.arange(start, min(start + 64, real.n_obs))
        left = dense_block(real, rows, pairs["原region索引"].to_numpy())
        right = dense_block(real, rows, pairs["复制region索引"].to_numpy())
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError("真实数据的最后复制区间与来源值或缺失位置不一致")


def natural_reference_statistics(item, rows, natural_count, label_key="Subclass", chunk_size=512):
    """仅在真实参照的原始区间计算中位数和按真实类型分组的 ANOVA。"""
    labels = read_labels(item, rows, label_key)
    levels = np.unique(labels)
    if len(levels) < 2 or len(levels) >= len(rows):
        raise ValueError("ANOVA 需要至少两种真实类型，且细胞数大于类型数")
    fill = np.full(natural_count, np.nan)
    scores = np.full(natural_count, np.nan)
    for start in range(0, natural_count, chunk_size):
        stop = min(start + chunk_size, natural_count)
        values = dense_block(item, rows, slice(start, stop))
        available = np.isfinite(values).any(axis=0)
        values[~np.isfinite(values)] = np.nan
        medians = np.full(stop - start, np.nan)
        medians[available] = np.nanmedian(values[:, available], axis=0)
        fill[start:stop] = medians
        values = values[:, available]
        np.copyto(values, medians[available][None, :], where=~np.isfinite(values))
        grand_mean = values.mean(axis=0)
        between = np.zeros(values.shape[1])
        within = np.zeros(values.shape[1])
        for level in levels:
            group = values[labels == level]
            mean = group.mean(axis=0)
            between += len(group) * (mean - grand_mean) ** 2
            within += ((group - mean) ** 2).sum(axis=0)
        numerator = between / (len(levels) - 1)
        denominator = within / (len(rows) - len(levels))
        local = np.divide(numerator, denominator, out=np.full(values.shape[1], np.nan), where=denominator > 0)
        local[(denominator == 0) & (numerator > 0)] = np.inf
        scores[start:stop][available] = local
    if not np.isfinite(fill).any():
        raise ValueError("真实参照的原始区间没有有效观测")
    fill[~np.isfinite(fill)] = np.median(fill[np.isfinite(fill)])
    return fill, scores


def select_features(scores, count):
    """按 ANOVA 排序，最多选择 count 个有效原始区间，结果保持原始列序。"""
    eligible = np.flatnonzero(np.isfinite(scores) | np.isposinf(scores))
    if count < 2 or eligible.size < 2:
        raise ValueError("至少需要两个可用于 ANOVA 排序的原始区间")
    order = np.argsort(scores[eligible], kind="stable")[-min(count, len(eligible)):]
    return np.sort(eligible[order])


def imputed_features(item, rows, selected, fill):
    """所有数据共享真实参照的逐区间填充值，不截断生成值。"""
    values = dense_block(item, rows, selected)
    np.copyto(values, fill[selected][None, :], where=~np.isfinite(values))
    return values.astype(np.float32)

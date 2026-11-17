"""基线共用的完整区域轴读取与结果核对。"""
import hashlib
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def close_backed(item):
    if getattr(item, "file", None) is not None:
        item.file.close()


def dense32(values):
    return np.asarray(values.toarray() if sp.issparse(values) else values, dtype=np.float32)


def hash_text(values):
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_training(path):
    item = ad.read_h5ad(path, backed="r")
    try:
        names = item.var_names.astype(str).to_numpy()
        original_count = item.n_vars - 1000
        if original_count < 1000 or not item.var_names.is_unique or not item.obs_names.is_unique:
            raise ValueError("训练输入必须包含原始区间及末尾 1000 个唯一的复制区间，细胞 ID 必须唯一")
        originals = pd.Index(names[:original_count])
        copied_names = names[original_count:]
        suffixes = [f"__duplicate_{i:04d}" for i in range(1000)]
        if not all(name.endswith(suffix) for name, suffix in zip(copied_names, suffixes)):
            raise ValueError("最后 1000 列缺少顺序正确的复制区间名称")
        source_names = [name[:-len(suffix)] for name, suffix in zip(copied_names, suffixes)]
        indices = originals.get_indexer(source_names)
        if (indices < 0).any() or len(np.unique(indices)) != 1000:
            raise ValueError("复制区间无法唯一对应到原始区间")
        labels = item.obs.get("Subclass")
        if labels is None or labels.isna().any() or labels.astype(str).str.strip().eq("").any():
            raise ValueError("训练输入缺少有效的 Subclass")
        matrix = np.empty(item.shape, dtype=np.float32)
        for start in range(0, item.n_obs, 32):
            block = dense32(item.X[start:start + 32])
            if np.isinf(block).any() or np.any(np.isfinite(block) & ((block < 0) | (block > 1))):
                raise ValueError("真实训练数据必须是 [0, 1] 内的比例或 NaN")
            if np.any(np.isfinite(block).sum(axis=1) == 0):
                raise ValueError("训练输入存在全缺失细胞")
            if not np.array_equal(block[:, indices], block[:, original_count:], equal_nan=True):
                raise ValueError("训练数据的复制区间数值或缺失位置与原始区间不一致")
            matrix[start:start + len(block)] = block
        pairs = pd.DataFrame({"原region索引": indices,
            "复制region索引": np.arange(original_count, item.n_vars),
            "原region": source_names, "复制region": copied_names})
        return matrix, item.obs.copy(), item.var.copy(), pairs
    finally:
        close_backed(item)


def compare_h5ad(candidate_path, reference_path):
    candidate = ad.read_h5ad(candidate_path, backed="r")
    try:
        reference = ad.read_h5ad(reference_path, backed="r")
        try:
            if candidate.shape != reference.shape:
                raise AssertionError("输出形状与既有结果 不一致")
            if not candidate.var_names.equals(reference.var_names):
                raise AssertionError("完整区域名称或顺序与既有结果 不一致")
            if not candidate.obs_names.equals(reference.obs_names):
                raise AssertionError("细胞 ID 或顺序与既有结果 不一致")
            pd.testing.assert_series_equal(candidate.obs["Subclass"].astype(str),
                reference.obs["Subclass"].astype(str), check_names=False)
            for start in range(0, candidate.n_obs, 32):
                left = candidate.X[start:start + 32]
                right = reference.X[start:start + 32]
                left = left.toarray() if sp.issparse(left) else np.asarray(left)
                right = right.toarray() if sp.issparse(right) else np.asarray(right)
                if not np.array_equal(left, right, equal_nan=True):
                    raise AssertionError(f"完整输出数值或缺失位置不一致，起始行：{start}")
            return {"一致": True, "核对细胞数": candidate.n_obs,
                    "核对区间数": candidate.n_vars, "核对元素数": candidate.n_obs * candidate.n_vars,
                    "比较标准": "逐元素完全一致，NaN 位置一致"}
        finally:
            close_backed(reference)
    finally:
        close_backed(candidate)

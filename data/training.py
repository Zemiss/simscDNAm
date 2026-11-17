"""完整扩维训练输入；保留 notebook 的缺失填补和内部验证划分。"""
from __future__ import annotations
from pathlib import Path
import hashlib
import os
import re
import uuid
import warnings
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def _close_backed(adata: ad.AnnData) -> None:
    if getattr(adata, "file", None) is not None:
        adata.file.close()


def _dense_float32(block) -> np.ndarray:
    if sp.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=np.float32)


def inspect_inputs(paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    records: list[dict] = []
    reference_names: np.ndarray | None = None
    reference_var: pd.DataFrame | None = None

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在： {path}")
        item = ad.read_h5ad(path, backed="r")
        try:
            if "Subclass" not in item.obs.columns:
                raise KeyError(f"缺少 obs['Subclass']： {path}")
            names = item.var_names.astype(str).to_numpy()
            if reference_names is None:
                reference_names = names.copy()
                reference_var = item.var.copy()
            elif not np.array_equal(reference_names, names):
                raise ValueError(f"特征名称或顺序不一致： {path}")
            records.append(
                {
                    "source_dataset": path.parent.name,
                    "path": str(path),
                    "n_cells": int(item.n_obs),
                    "n_features": int(item.n_vars),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
        finally:
            _close_backed(item)

    assert reference_names is not None and reference_var is not None
    return pd.DataFrame(records), reference_names, reference_var


def materialize_inputs(
    paths: list[Path],
    work_dir: Path,
    expected_features: np.ndarray,
    row_block: int = 16,
) -> tuple[np.memmap, pd.DataFrame, np.ndarray, float]:
    source_info, _, _ = inspect_inputs(paths)
    total_rows = int(source_info["n_cells"].sum())
    n_features = len(expected_features)
    matrix_path = work_dir / "values.float32.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(total_rows, n_features)
    )

    metadata_parts: list[pd.DataFrame] = []
    finite_counts = np.zeros(total_rows, dtype=np.int64)
    finite_total = 0
    row_offset = 0

    for path in paths:
        item = ad.read_h5ad(path, backed="r")
        source = path.parent.name
        try:
            local_names = item.var_names.astype(str).to_numpy()
            if not np.array_equal(expected_features, local_names):
                raise ValueError(f"读取期间特征轴发生变化： {path}")
            labels = item.obs["Subclass"]
            local_obs = pd.DataFrame(
                {
                    "cell_id": [f"{source}::{name}" for name in item.obs_names.astype(str)],
                    "Subclass": labels.astype("string").fillna("").astype(str).to_numpy(),
                    "source_dataset": source,
                }
            )
            metadata_parts.append(local_obs)

            for start in range(0, item.n_obs, row_block):
                stop = min(start + row_block, item.n_obs)
                block = _dense_float32(item.X[start:stop])
                finite = np.isfinite(block)
                if np.isinf(block).any() or np.any(finite & ((block < 0) | (block > 1))):
                    raise ValueError(f"甲基化值不合法： {path.name}, rows {start}:{stop}")
                first, last = row_offset + start, row_offset + stop
                matrix[first:last] = block
                finite_counts[first:last] = finite.sum(axis=1)
                finite_total += int(finite.sum())
        finally:
            row_offset += int(item.n_obs)
            _close_backed(item)

    matrix.flush()
    metadata = pd.concat(metadata_parts, ignore_index=True)
    valid_label = metadata["Subclass"].str.strip().ne("").to_numpy()
    retained_rows = np.flatnonzero(valid_label & (finite_counts > 0)).astype(np.int64)
    if len(retained_rows) < 10:
        raise ValueError("固定质控后有效细胞少于 10 个")
    missing_fraction = 1.0 - finite_total / float(total_rows * n_features)
    metadata["finite_features"] = finite_counts
    metadata["retained"] = False
    metadata.loc[retained_rows, "retained"] = True
    metadata.to_csv(work_dir / "input_cell_audit.csv", index=False)
    return matrix, metadata, retained_rows, float(missing_fraction)


def hash_text(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stratified_group_split(
    metadata: pd.DataFrame,
    retained_rows: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    retained = metadata.loc[retained_rows, ["source_dataset", "Subclass"]].copy()
    train_parts: list[np.ndarray] = []
    heldout_parts: list[np.ndarray] = []
    grouped = retained.groupby(["source_dataset", "Subclass"], sort=True, observed=True)
    for group_index, (_, group) in enumerate(grouped):
        rows = group.index.to_numpy(dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, group_index]))
        rows = rng.permutation(rows)
        if len(rows) == 1:
            n_test = 0
        else:
            n_test = min(len(rows) - 1, max(1, int(round(len(rows) * test_fraction))))
        heldout_parts.append(rows[:n_test])
        train_parts.append(rows[n_test:])
    train_rows = np.sort(np.concatenate(train_parts)).astype(np.int64)
    heldout_rows = np.sort(np.concatenate([x for x in heldout_parts if len(x)])).astype(np.int64)
    if np.intersect1d(train_rows, heldout_rows).size:
        raise AssertionError("训练与留出细胞重叠")
    if not np.array_equal(np.sort(np.concatenate([train_rows, heldout_rows])), np.sort(retained_rows)):
        raise AssertionError("训练与留出划分未覆盖全部有效细胞")
    return train_rows, heldout_rows


def internal_subclass_split(
    truth_positions: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_index, label in enumerate(np.sort(np.unique(labels[truth_positions]))):
        positions = truth_positions[labels[truth_positions] == label]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 10_000, class_index]))
        positions = rng.permutation(positions)
        n_validation = 0 if len(positions) == 1 else min(
            len(positions) - 1, max(1, int(round(len(positions) * validation_fraction)))
        )
        validation_parts.append(positions[:n_validation])
        train_parts.append(positions[n_validation:])
    fit_positions = np.sort(np.concatenate(train_parts)).astype(np.int64)
    validation_positions = np.sort(
        np.concatenate([x for x in validation_parts if len(x)])
    ).astype(np.int64)
    if len(validation_positions) == 0:
        raise ValueError("内部验证集为空")
    return fit_positions, validation_positions


def write_real_h5ad(
    path: Path,
    matrix: np.memmap,
    rows: np.ndarray,
    metadata: pd.DataFrame,
    var: pd.DataFrame,
    feature_names: np.ndarray,
    input_row_block: int = 16,
    feature_block: int = 4096,
) -> None:
    obs = metadata.loc[rows, ["cell_id", "Subclass", "source_dataset"]].copy().set_index("cell_id")
    partial = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    shell = ad.AnnData(
        X=sp.csr_matrix((len(rows), len(feature_names)), dtype=np.float32),
        obs=obs,
        var=var.copy(),
    )
    shell.var_names = pd.Index(feature_names.astype(str))
    shell.write_h5ad(partial, compression="gzip")
    with h5py.File(partial, "r+", locking=False) as handle:
        del handle["X"]
        dataset = handle.create_dataset(
            "X",
            shape=(len(rows), len(feature_names)),
            dtype="float32",
            chunks=(min(input_row_block, len(rows)), min(feature_block, len(feature_names))),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.nan,
        )
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"
        for start in range(0, len(rows), input_row_block):
            stop = min(start + input_row_block, len(rows))
            dataset[start:stop] = np.asarray(matrix[rows[start:stop]], dtype=np.float32)
    if path.exists():
        path.unlink()
    partial.replace(path)


def validate_real_h5ad(
    path: Path,
    expected_rows: int,
    feature_names: np.ndarray,
) -> None:
    item = ad.read_h5ad(path, backed="r")
    try:
        if item.shape != (expected_rows, len(feature_names)):
            raise AssertionError(f"真实数据形状不符合预期： {path}: {item.shape}")
        if not np.array_equal(item.var_names.astype(str).to_numpy(), feature_names.astype(str)):
            raise AssertionError(f"特征轴不一致： {path}")
    finally:
        _close_backed(item)


def fit_feature_medians(
    matrix: np.memmap,
    train_rows: np.ndarray,
    feature_block: int,
) -> np.ndarray:
    medians = np.empty(matrix.shape[1], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for start in range(0, matrix.shape[1], feature_block):
            stop = min(start + feature_block, matrix.shape[1])
            block = np.asarray(matrix[train_rows, start:stop], dtype=np.float32)
            medians[start:stop] = np.nanmedian(block, axis=0).astype(np.float32)
    finite_medians = medians[np.isfinite(medians)]
    if finite_medians.size == 0:
        raise ValueError("训练矩阵没有有限甲基化观测")
    global_median = float(np.median(finite_medians))
    medians[~np.isfinite(medians)] = global_median
    return medians


def fit_feature_means(
    matrix: np.memmap,
    train_rows: np.ndarray,
    medians: np.ndarray,
    feature_block: int,
) -> np.ndarray:
    means = np.empty(matrix.shape[1], dtype=np.float32)
    for start in range(0, matrix.shape[1], feature_block):
        stop = min(start + feature_block, matrix.shape[1])
        block = np.array(matrix[train_rows, start:stop], dtype=np.float32, copy=True)
        missing = ~np.isfinite(block)
        if missing.any():
            block[missing] = medians[start:stop][np.nonzero(missing)[1]]
        means[start:stop] = block.mean(axis=0, dtype=np.float64).astype(np.float32)
    return means


class FullMethylationDataset(Dataset):
    def __init__(
        self,
        matrix: np.memmap,
        retained_rows: np.ndarray,
        positions: np.ndarray,
        medians: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.matrix = matrix
        self.rows = np.asarray(retained_rows[positions], dtype=np.int64)
        self.medians = np.asarray(medians, dtype=np.float32)
        self.labels = np.asarray(labels[positions], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = np.array(self.matrix[int(self.rows[index])], dtype=np.float32, copy=True)
        observed = np.isfinite(values)
        if not observed.any():
            raise ValueError(f"Cell row {self.rows[index]} 没有观测特征")
        values[~observed] = self.medians[~observed]
        return (
            torch.from_numpy(values),
            torch.from_numpy(observed),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


# 复制对应关系口径与 evaluate/preprocessing.py 一致。
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


def load_training_input(path, work_dir, pair_path=None, row_block=16):
    """输入已扩维的训练 H5AD；不再划分外层留出集，不重新复制列。"""
    path, work_dir = Path(path), Path(work_dir)
    item = ad.read_h5ad(path, backed="r")
    try:
        pairs = region_pairs(item, pair_path=pair_path)
        validate_real_duplicates(item, pairs)
        input_obs = item.obs.copy()
    finally:
        _close_backed(item)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_table, names, var = inspect_inputs([path])
    matrix, metadata, retained, missing_fraction = materialize_inputs(
        [path], work_dir, names, row_block=row_block)
    # 输入已是准备好的训练集，不再添加原始多数据集合并时的路径前缀。
    metadata["cell_id"] = input_obs.index.astype(str).to_numpy()
    if "source_dataset" in input_obs:
        metadata["source_dataset"] = input_obs["source_dataset"].astype(str).to_numpy()
    metadata.to_csv(work_dir / "input_cell_audit.csv", index=False)
    return {"matrix": matrix, "cell_metadata": metadata, "retained_rows": retained,
            "feature_names": names, "feature_var": var, "region_pairs": pairs,
            "source_table": source_table, "missing_fraction": missing_fraction}


def load_target_labels(path, feature_names, class_names):
    """留出数据仅提供真实 Subclass 配额，允许原始轴或同一扩维轴。"""
    item = ad.read_h5ad(path, backed="r")
    try:
        axis = item.var_names.astype(str).to_numpy()
        if not (np.array_equal(axis, feature_names) or np.array_equal(axis, feature_names[:-1000])):
            raise ValueError("留出数据的特征轴必须与训练数据的原始轴或扩维轴一致")
        if "Subclass" not in item.obs or item.obs["Subclass"].isna().any() or item.n_obs == 0:
            raise ValueError("留出数据缺少有效的 Subclass 标签")
        labels = item.obs["Subclass"].astype(str).to_numpy(dtype=str)
        if np.any(np.char.strip(labels) == "") or len(np.setdiff1d(labels, class_names)):
            raise ValueError("留出数据的 Subclass 必须非空且存在于训练数据中")
        return labels
    finally:
        _close_backed(item)

"""基线共用的完整区域轴读取与结果核对。"""
import hashlib
import json
import uuid
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


def prepare_imputed_training(training_path, target_path, cache_dir):
    """仅用真实训练中位数填补；目标文件只提供区域轴、ID和类型配额。"""
    from data.paths import check_output

    training_path, target_path = Path(training_path).resolve(), Path(target_path).resolve()
    cache_dir = check_output(cache_dir)
    records = [{"路径": str(path), "字节数": path.stat().st_size,
                "修改时间纳秒": path.stat().st_mtime_ns} for path in (training_path, target_path)]
    identity = {"输入": records, "填补规则": "真实训练集逐区域中位数；全缺失列用训练有限中位数的中位数"}
    training_header = ad.read_h5ad(training_path, backed="r")
    target = ad.read_h5ad(target_path, backed="r")
    try:
        if not training_header.var_names.equals(target.var_names):
            raise ValueError("训练与目标区域轴不一致")
        if len(training_header.obs_names.intersection(target.obs_names)):
            raise ValueError("训练与目标细胞 ID 重叠")
        target_obs = target.obs.copy()
        labels = target_obs.get("Subclass")
        if labels is None or labels.isna().any() or labels.astype(str).str.strip().eq("").any():
            raise ValueError("目标缺少真实 Subclass 配额")
        if not set(labels.astype(str)).issubset(set(training_header.obs["Subclass"].astype(str))):
            raise ValueError("目标含训练中不存在的 Subclass")
    finally:
        close_backed(training_header)
        close_backed(target)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path, report_path = cache_dir / "TrainingImputed.h5ad", cache_dir / "imputation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["来源"] != identity:
            raise ValueError("填补缓存的输入来源已变化，拒绝复用")
        if output_path.is_file():
            return output_path, target_obs, report
    matrix, obs, var, pairs = read_training(training_path)
    medians = np.full(matrix.shape[1], np.nan, dtype=np.float32)
    missing_count = 0
    for start in range(0, matrix.shape[1], 1024):
        block = matrix[:, start:start + 1024]
        available = np.isfinite(block).any(axis=0)
        missing_count += int((~np.isfinite(block)).sum())
        local = np.full(block.shape[1], np.nan, dtype=np.float32)
        local[available] = np.nanmedian(block[:, available], axis=0)
        medians[start:start + block.shape[1]] = local
    available = np.isfinite(medians)
    fallback = float(np.median(medians[available]))
    medians[~available] = fallback
    for start in range(0, matrix.shape[1], 1024):
        block = matrix[:, start:start + 1024]
        np.copyto(block, medians[start:start + block.shape[1]][None, :], where=~np.isfinite(block))
    if not np.array_equal(matrix[:, pairs["原region索引"]], matrix[:, pairs["复制region索引"]]):
        raise AssertionError("填补后真实复制列不一致")
    publication_id = uuid.uuid4().hex
    partial = cache_dir / f".TrainingImputed.{publication_id}.h5ad.partial"
    ad.AnnData(matrix, obs=obs, var=var).write_h5ad(partial, compression="gzip", compression_opts=1)
    partial.replace(output_path)
    partial_medians = cache_dir / f".training_medians.{publication_id}.npy"
    np.save(partial_medians, medians)
    partial_medians.replace(cache_dir / "training_medians.npy")
    report = {"来源": identity, "状态": "完成", "训练细胞数": matrix.shape[0], "区域数": matrix.shape[1],
              "填补元素数": missing_count, "全缺失列数": int((~available).sum()),
              "全缺失列填充值": fallback, "训练细胞ID哈希": hash_text(obs.index), "区域轴哈希": hash_text(var.index)}
    partial_report = cache_dir / f".imputation.{publication_id}.json.partial"
    partial_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_report.replace(report_path)
    return output_path, target_obs, report


def clip_generated_matrix(values):
    """仅裁剪有限生成值至 0–1；不改变行列、缺失状态或做额外映射。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("生成矩阵须为非空、有限的二维数值矩阵")
    report = {"裁剪前最小值": float(values.min()), "裁剪前最大值": float(values.max()),
              "小于零的数目": int((values < 0).sum()), "大于一的数目": int((values > 1).sum()),
              "总元素数": values.size, "操作": "仅 clip(0,1)"}
    return np.clip(values, 0, 1), report

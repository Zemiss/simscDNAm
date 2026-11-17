"""simCAS 完整区域轴训练与生成。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.paths import PROJECT, OUTPUTS, check_output
from baseline.common import parse_arguments

def run(training_path, target_obs_path, output_dir, seed=42, reference_path=None, replay_root=None):
    METHOD = 'simCAS'
    RUN_MODE = "replay" if replay_root is not None else "run"
    ALLOWED_OUTPUT_ROOT = OUTPUTS
    HISTORICAL = Path(replay_root) if replay_root is not None else OUTPUTS / "archive/legacy_baselines"
    TRAINING_PATH, TARGET_OBS_PATH = Path(training_path), Path(target_obs_path)
    OUTPUT_ROOT = Path(output_dir).parent
    REFERENCE_PATH = Path(reference_path) if reference_path is not None else None
    SEED = seed
    FEATURE_BLOCK = 512
    ROW_BLOCK = 32
    import hashlib
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from data.baseline import close_backed, dense32, hash_text, read_training, compare_h5ad


    import importlib.util
    import json
    import math
    import os
    import shutil
    import subprocess
    import sys
    import time
    import uuid
    import warnings
    import h5py
    from scipy.io import mmread, mmwrite

    assert "simscdnam" in sys.executable.lower(), "请使用 simscDNAm Python 环境"
    if RUN_MODE not in {"run", "replay"}:
        raise ValueError("RUN_MODE 只能是 run 或 replay")
    METHOD_DIR = Path(output_dir)
    check_output(METHOD_DIR)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = METHOD_DIR / "data.h5ad"
    if OUTPUT_PATH.resolve() in {TRAINING_PATH.resolve(), TARGET_OBS_PATH.resolve()}:
        raise ValueError("输出路径不能覆盖输入")
    input_records = {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
                     for path in (TRAINING_PATH, TARGET_OBS_PATH)}
    matrix, training_obs, feature_var, region_pairs = read_training(TRAINING_PATH)
    feature_names = feature_var.index.astype(str).to_numpy()
    train_rows = np.arange(len(matrix), dtype=np.int64)
    train_labels = training_obs["Subclass"].astype(str).to_numpy()
    target_item = ad.read_h5ad(TARGET_OBS_PATH, backed="r")
    try:
        # 仅读取目标类别配额，不读取留出集 X，不重新划分训练集。
        if target_item.obs["Subclass"].isna().any():
            raise ValueError("目标类别含缺失值")
        heldout_labels = target_item.obs["Subclass"].astype(str).to_numpy()
    finally:
        close_backed(target_item)
    if not len(heldout_labels) or set(heldout_labels) - set(train_labels):
        raise ValueError("生成目标为空或存在训练集没有的类别")
    target_labels = heldout_labels.copy()
    np.random.default_rng(SEED).shuffle(target_labels)
    N_TARGET = len(target_labels)
    heldout_rows = np.arange(N_TARGET)
    target_obs = pd.DataFrame({"Subclass": target_labels, "origin": "Synthetic"},
        index=pd.Index([f"synthetic_{i:06d}" for i in range(1, N_TARGET + 1)], name="cell_id"))
    AXIS_HASH = hash_text(feature_names)
    INPUT_SIGNATURE = hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()
    SPLIT_HASH = hash_text(training_obs.index)
    TARGET_HASH = hash_text(target_labels)
    region_pairs.to_csv(METHOD_DIR / "region_pairs.csv", index=False)
    if RUN_MODE == "replay":
        if REFERENCE_PATH is None:
            raise ValueError("replay 必须指定历史结果")
        # 历史缓存只有在训练矩阵、细胞、特征轴和目标标签顺序完全相同时才允许使用。
        compare_h5ad(TRAINING_PATH, HISTORICAL / "methods/Truth/data.h5ad")
        old = ad.read_h5ad(REFERENCE_PATH, backed="r")
        try:
            if not old.var_names.equals(feature_var.index) or not old.obs_names.equals(target_obs.index):
                raise ValueError("历史缓存的目标细胞或区域轴不同")
            np.testing.assert_array_equal(old.obs["Subclass"].astype(str).to_numpy(), target_labels)
        finally:
            close_backed(old)
    print({"方法": METHOD, "模式": RUN_MODE, "训练细胞": len(matrix),
           "原始区间": len(feature_names) - 1000, "追加复制区间": 1000,
           "完整输出形状": [N_TARGET, len(feature_names)]})


    def fit_training_feature_medians(matrix, rows, feature_block):
        medians = np.empty(matrix.shape[1], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for start in range(0, matrix.shape[1], feature_block):
                stop = min(start + feature_block, matrix.shape[1])
                block = np.asarray(matrix[rows, start:stop], dtype=np.float32)
                medians[start:stop] = np.nanmedian(block, axis=0).astype(np.float32)
        finite = medians[np.isfinite(medians)]
        if finite.size == 0:
            raise ValueError("Training 中没有有限的特征值")
        medians[~np.isfinite(medians)] = np.float32(np.median(finite))
        return medians

    def filled_matrix_rows(matrix, rows, medians):
        values = np.array(matrix[rows], dtype=np.float32, copy=True)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = medians[np.nonzero(missing)[1]]
        return values

    def write_simcas_input(path, matrix, rows, labels, medians, var, names):
        obs = pd.DataFrame(
            {"celltype": labels},
            index=pd.Index([f"training_{i:06d}" for i in range(len(rows))], name="cell_id"),
        )
        partial = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
        shell = ad.AnnData(
            X=sp.csr_matrix((len(rows), len(names)), dtype=np.float32),
            obs=obs,
            var=var.copy(),
        )
        shell.var_names = pd.Index(names.astype(str))
        shell.write_h5ad(partial, compression="gzip")
        with h5py.File(partial, "r+", locking=False) as handle:
            del handle["X"]
            output = handle.create_dataset(
                "X",
                shape=(len(rows), len(names)),
                dtype="float32",
                chunks=(min(16, len(rows)), min(4096, len(names))),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            output.attrs["encoding-type"] = "array"
            output.attrs["encoding-version"] = "0.2.0"
            for start in range(0, len(rows), 16):
                stop = min(start + 16, len(rows))
                output[start:stop] = filled_matrix_rows(matrix, rows[start:stop], medians)
        if path.exists():
            path.unlink()
        partial.replace(path)

    SIMCAS_INPUT_PATH = METHOD_DIR / "training_continuous_beta_as_counts.h5ad"
    training_feature_medians = fit_training_feature_medians(matrix, train_rows, FEATURE_BLOCK)
    write_simcas_input(SIMCAS_INPUT_PATH, matrix, train_rows, train_labels,
                       training_feature_medians, feature_var, feature_names)


    OFFICIAL_SIMCAS_ROOT = PROJECT / "baseline/simCAS/vendor"
    R_HOME = Path(os.environ.get("R_HOME", "/home/sccasimp/anaconda3/envs/R4/lib/R"))
    training_counts = pd.Series(train_labels).value_counts().sort_index()
    heldout_counts = pd.Series(heldout_labels).value_counts().sort_index()
    cell_scale = max(1.0, float((heldout_counts / training_counts).max()) * 1.10)
    raw_path = METHOD_DIR / "simcas_raw.h5ad"
    started = time.perf_counter()
    if RUN_MODE == "replay":
        cached_root = HISTORICAL / "methods/simCAS"
        cached_input = ad.read_h5ad(cached_root / "_work/training_continuous_beta_as_counts.h5ad", backed="r")
        try:
            if cached_input.shape != matrix.shape or not cached_input.var_names.equals(feature_var.index):
                raise ValueError("simCAS 缓存输入区域轴不一致")
            np.testing.assert_array_equal(cached_input.obs["celltype"].astype(str).to_numpy(), train_labels)
            for start in range(0, len(matrix), 32):
                expected = filled_matrix_rows(matrix, train_rows[start:start + 32], training_feature_medians)
                np.testing.assert_array_equal(dense32(cached_input.X[start:start + 32]), expected)
        finally:
            close_backed(cached_input)
        summary = json.loads((cached_root / "run_summary.json").read_text(encoding="utf-8"))
        for key, expected in {"rand_seed": 2022, "simu_type": "cell_type", "correct_iter": 2,
            "activation": "sigmod", "lib_simu": "real", "distribution": "Poisson", "cell_scale": cell_scale}.items():
            if summary[key] != expected:
                raise ValueError(f"simCAS 缓存参数不一致：{key}")
        generated = ad.read_h5ad(cached_root / "_work/simcas_raw.h5ad")
    else:
        source = OFFICIAL_SIMCAS_ROOT / "code/simCAS.py"
        if not source.is_file() or not R_HOME.is_dir():
            raise FileNotFoundError("缺少已安装的官方 simCAS 或 R4 运行时")
        os.environ["R_HOME"] = str(R_HOME)
        os.environ["PATH"] = str(R_HOME.parent / "bin") + os.pathsep + os.environ["PATH"]
        specification = importlib.util.spec_from_file_location("official_simCAS", source)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        generated = module.simCAS_generate(adata_dir=str(SIMCAS_INPUT_PATH),
            n_peak=len(feature_names), rand_seed=2022, simu_type="cell_type", correct_iter=2,
            activation="sigmod", lib_simu="real", distribution="Poisson", cell_scale=cell_scale)
        generated.var = feature_var.copy()
        generated.var_names = pd.Index(feature_names)
        generated.write_h5ad(raw_path, compression="gzip")
    if generated.n_vars != len(feature_names):
        raise ValueError("simCAS 原始输出未保留全部区间")
    generated_labels = generated.obs["celltype"].astype(str).to_numpy()
    generated_values = generated.X.data if sp.issparse(generated.X) else np.asarray(generated.X)
    if not np.isfinite(generated_values).all():
        raise ValueError("simCAS 原始输出包含非有限值")
    rng = np.random.default_rng(2022)
    selected_by_class = {}
    for label, target_count in heldout_counts.items():
        available = np.flatnonzero(generated_labels == label)
        if len(available) < target_count:
            raise ValueError(f"simCAS 生成的类别细胞数不足：{label}")
        selected_by_class[label] = rng.choice(available, size=int(target_count), replace=False)
    queues = {label: iter(rng.permutation(indices).tolist()) for label, indices in selected_by_class.items()}
    selected = np.fromiter((next(queues[label]) for label in target_labels), dtype=np.int64, count=N_TARGET)
    synthetic = generated[selected].copy()
    synthetic.obs = target_obs.copy()
    synthetic.var = feature_var.copy()
    synthetic.var_names = pd.Index(feature_names)
    partial = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.{uuid.uuid4().hex}.partial")
    synthetic.write_h5ad(partial, compression="gzip")
    partial.replace(OUTPUT_PATH)
    del generated, generated_values, synthetic
    print({"缩放系数": cell_scale, "输出细胞": N_TARGET, "耗时秒": time.perf_counter() - started})


    item = ad.read_h5ad(OUTPUT_PATH, backed="r")
    try:
        if item.shape != (N_TARGET, len(feature_names)) or not item.var_names.equals(feature_var.index):
            raise AssertionError("生成结果未保留完整区域轴")
        if not item.obs_names.equals(target_obs.index):
            raise AssertionError("生成结果细胞顺序不一致")
        np.testing.assert_array_equal(item.obs["Subclass"].astype(str).to_numpy(), target_labels)
        minimum, maximum, observed = math.inf, -math.inf, 0
        for start in range(0, item.n_obs, ROW_BLOCK):
            block = dense32(item.X[start:start + ROW_BLOCK])
            if np.isinf(block).any() or (METHOD != "FittedMarginal" and not np.isfinite(block).all()):
                raise AssertionError("生成结果含无效数值")
            finite = block[np.isfinite(block)]
            if finite.size:
                minimum, maximum = min(minimum, float(finite.min())), max(maximum, float(finite.max()))
                observed += int(finite.size)
        if not observed or minimum < 0 or (METHOD == "FittedMarginal" and maximum > 1):
            raise AssertionError("生成结果的数值范围不正确")
    finally:
        close_backed(item)
    comparison = {"一致": None, "说明": "未配置历史结果，仅完成输入与全维输出检查"}
    if REFERENCE_PATH is not None:
        comparison = compare_h5ad(OUTPUT_PATH, REFERENCE_PATH)
    for path, before in input_records.items():
        if [Path(path).stat().st_size, Path(path).stat().st_mtime_ns] != before:
            raise AssertionError("运行期间输入文件发生变化")
    verification = {"方法": METHOD, "运行模式": RUN_MODE, "训练形状": list(matrix.shape),
        "输出形状": [N_TARGET, len(feature_names)], "原始区间": len(feature_names) - 1000,
        "训练前复制区间": 1000, "生成后复制": False, "数值最小值": minimum,
        "数值最大值": maximum, "有限元素数": observed, "输入保持不变": True,
        "训练矩阵SHA256": INPUT_SIGNATURE, "区域轴SHA256": AXIS_HASH,
        "目标标签SHA256": TARGET_HASH, "历史结果": str(REFERENCE_PATH),
        "一致性检查": comparison, "输出文件": str(OUTPUT_PATH)}
    (METHOD_DIR / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame([{"方法": METHOD, "运行模式": RUN_MODE, "输出细胞": N_TARGET,
        "输出区间": len(feature_names), "与既有结果逐元素一致": comparison["一致"]}]))
    del matrix
    print("输出：", OUTPUT_PATH)

    return OUTPUT_PATH

def main(argv=None):
    args = parse_arguments('simCAS', argv)
    return run(args.training, args.target, args.output, args.seed, args.reference, args.replay_root)


if __name__ == "__main__":
    main()

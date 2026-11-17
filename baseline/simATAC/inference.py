"""simATAC 完整区域轴训练与生成。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.paths import PROJECT, OUTPUTS, check_output
from baseline.common import parse_arguments

def run(training_path, target_obs_path, output_dir, seed=42, reference_path=None, replay_root=None):
    METHOD = 'simATAC'
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


    SIMATAC_PATH = OUTPUT_PATH
    INTERMEDIATE_DIR = (HISTORICAL / "methods/simATAC/intermediate" if RUN_MODE == "replay"
                        else METHOD_DIR / "intermediate")
    R_LIBRARY = PROJECT / "baseline/simATAC/.runtime/r_library"
    INSTALL_MISSING = False
    if RUN_MODE == "run":
        INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    def training_feature_medians(values, rows, block_size=FEATURE_BLOCK):
        medians = np.empty(values.shape[1], dtype=np.float32)
        for start in range(0, values.shape[1], block_size):
            stop = min(start + block_size, values.shape[1])
            columns = np.arange(start, stop, dtype=np.int64)
            block = np.asarray(values[np.ix_(rows, columns)], dtype=np.float32)
            with np.errstate(all="ignore"):
                medians[start:stop] = np.nanmedian(block, axis=0)
        fallback = float(np.nanmedian(medians))
        if not np.isfinite(fallback):
            raise ValueError("Training 中没有可用于填补的有限值")
        medians[~np.isfinite(medians)] = fallback
        return medians

    def resolve_rscript():
        candidates = (
            shutil.which("Rscript"),
            "/home/sccasimp/anaconda3/envs/R4/bin/Rscript",
            "/home/sccasimp/anaconda3/envs/deeplearningR/bin/Rscript",
        )
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return str(candidate)
        raise FileNotFoundError("找不到 simATAC 所需的 Rscript")

    SIMATAC_R = r"""
    args <- commandArgs(trailingOnly=TRUE)
    input_path <- args[[1]]
    output_path <- args[[2]]
    n_cells <- as.integer(args[[3]])
    private_library <- args[[4]]
    install_missing <- identical(args[[5]], "1")
    simulation_seed <- as.integer(args[[6]])
    dir.create(private_library, recursive=TRUE, showWarnings=FALSE)
    .libPaths(c(private_library, .libPaths()))
    if (!requireNamespace("simATAC", quietly=TRUE)) {
      if (!install_missing) stop("未安装 simATAC；请将 INSTALL_MISSING 设为 True")
      if (!requireNamespace("remotes", quietly=TRUE)) {
        install.packages("remotes", repos="https://cloud.r-project.org", lib=private_library)
      }
      remotes::install_github("bowang-lab/simATAC", lib=private_library,
                              dependencies=TRUE, upgrade="never")
    }
    counts <- methods::as(Matrix::readMM(input_path), "dgCMatrix")

    simulate_with_empirical_library <- function(counts, n_cells) {
      keep <- Matrix::colSums(counts) > 0
      counts <- counts[, keep, drop=FALSE]
      if (ncol(counts) < 1) stop("有效 Training 细胞少于 1 个")
      lib_size <- Matrix::colSums(counts)
      lib_median <- stats::median(lib_size)
      estimation_counts <- counts
      if (ncol(estimation_counts) == 1) {
        estimation_counts <- cbind(estimation_counts, estimation_counts)
      }
      estimation_lib_size <- Matrix::colSums(estimation_counts)
      norm_counts <- methods::as(
        estimation_counts %*% Matrix::Diagonal(x=lib_median / estimation_lib_size),
        "dgCMatrix"
      )
      object <- simATAC::newsimATACCount()
      object <- simATAC::setParameters(
        object, nBins=nrow(counts), default=FALSE
      )
      estimate_nonzero <- getFromNamespace("simATACEstNonZeroPro", "simATAC")
      estimate_bin_mean <- getFromNamespace("simATACEstBinMean", "simATAC")
      object <- estimate_nonzero(norm_counts, object)
      object <- tryCatch(
        estimate_bin_mean(norm_counts, object),
        error=function(error) {
          nonzero <- Matrix::rowSums(norm_counts != 0) / ncol(norm_counts)
          bin_mean <- Matrix::rowSums(norm_counts) / ncol(norm_counts)
          usable <- is.finite(bin_mean) & nonzero < 0.8
          constant_mean <- mean(bin_mean[usable])
          if (!is.finite(constant_mean) || constant_mean <= 0) {
            constant_mean <- mean(bin_mean[is.finite(bin_mean) & bin_mean > 0])
          }
          if (!is.finite(constant_mean) || constant_mean <= 0) {
            constant_mean <- sqrt(.Machine$double.eps)
          }
          message(paste0("bin_mean_fallback=constant: ", conditionMessage(error)))
          simATAC::setParameters(
            object, mean.coef0=constant_mean, mean.coef1=0, mean.coef2=0
          )
        }
      )
      log_library <- log2(lib_size + 1)
      library_mean <- mean(log_library)
      library_sd <- stats::sd(log_library)
      if (!is.finite(library_sd) || library_sd <= 0) {
        library_sd <- sqrt(.Machine$double.eps)
      }
      object <- simATAC::setParameters(
        object,
        nCells=n_cells,
        lib.mean1=library_mean,
        lib.sd1=library_sd,
        lib.mean2=library_mean,
        lib.sd2=library_sd,
        lib.prob=0.5
      )
      simATAC::simATACSimulate(object=object, verbose=TRUE)
    }

    set.seed(simulation_seed)
    simulated <- tryCatch(
      simATAC::simATACGenerate(
        count=counts, default=FALSE, verbose=TRUE, nCells=n_cells
      ),
      error=function(error) {
        message_text <- conditionMessage(error)
        message(paste0("estimation_fallback=single_gaussian: " , message_text))
        set.seed(simulation_seed)
        simulate_with_empirical_library(counts, n_cells)
      }
    )
    result <- methods::as(
      SummarizedExperiment::assay(simulated, "counts"), "dgCMatrix"
    )
    Matrix::writeMM(result, output_path)
    """

    def create_simatac_h5ad(path, obs, var, names):
        partial = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
        shell = ad.AnnData(
            X=sp.csr_matrix((len(obs), len(names)), dtype=np.float32),
            obs=obs.copy(),
            var=var.copy(),
        )
        shell.var_names = pd.Index(names.astype(str))
        shell.uns["baseline"] = "simATAC"
        shell.uns["native_output"] = True
        shell.uns["input_representation"] = "training-median-filled beta used directly"
        shell.uns["output_representation"] = "native simATAC counts or scores; not beta proportions"
        shell.write_h5ad(partial, compression="gzip")
        with h5py.File(partial, "r+", locking=False) as handle:
            del handle["X"]
            output = handle.create_dataset(
                "X",
                shape=(len(obs), len(names)),
                dtype="float32",
                chunks=(min(ROW_BLOCK, len(obs)), min(4096, len(names))),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fillvalue=0.0,
            )
            output.attrs["encoding-type"] = "array"
            output.attrs["encoding-version"] = "0.2.0"
        return partial

    def validate_simatac_output(path):
        expected_counts = pd.Series(heldout_labels).value_counts().sort_index().rename_axis(None)
        observed = 0
        minimum = math.inf
        maximum = -math.inf
        with h5py.File(path, "r", locking=False) as handle:
            data = handle["X"]
            if data.shape != (N_TARGET, len(feature_names)):
                raise ValueError(f"simATAC 输出形状错误：{data.shape}")
            for start in range(0, N_TARGET, ROW_BLOCK):
                block = np.asarray(data[start:start + ROW_BLOCK], dtype=np.float32)
                if not np.isfinite(block).all():
                    raise ValueError("simATAC 输出包含非有限值")
                if np.any(block < 0):
                    raise ValueError("simATAC 输出包含负值")
                observed += block.size
                minimum = min(minimum, float(block.min()))
                maximum = max(maximum, float(block.max()))
        item = ad.read_h5ad(path, backed="r")
        try:
            if not np.array_equal(
                item.var_names.astype(str).to_numpy(), feature_names.astype(str)
            ):
                raise ValueError("simATAC 输出特征轴不一致")
            generated_counts = (
                item.obs["Subclass"].astype(str).value_counts().sort_index().rename_axis(None)
            )
            pd.testing.assert_series_equal(generated_counts, expected_counts)
            if not bool(item.uns["native_output"]):
                raise ValueError("simATAC 输出缺少原生结果标记")
        finally:
            close_backed(item)
        return {
            "observed": int(observed),
            "finite_min": minimum,
            "finite_max": maximum,
        }

    def run_simatac():
        started = time.perf_counter()
        rscript = resolve_rscript()
        feature_medians = training_feature_medians(matrix, train_rows)
        target_obs = pd.DataFrame(
            {"Subclass": target_labels, "origin": "Synthetic"},
            index=pd.Index(
                [f"synthetic_{i:06d}" for i in range(1, N_TARGET + 1)],
                name="cell_id",
            ),
        )
        fit_by_label = {
            label: train_rows[train_labels == label] for label in np.unique(train_labels)
        }
        target_by_label = {
            label: np.flatnonzero(target_labels == label)
            for label in np.unique(target_labels)
        }
        partial = create_simatac_h5ad(
            SIMATAC_PATH, target_obs, feature_var, feature_names
        )
        reuse_intermediates = RUN_MODE == "replay"
        fallback_labels = []
        reused_labels = []
        with h5py.File(partial, "r+", locking=False) as handle:
            output = handle["X"]
            for group_index, label in enumerate(sorted(target_by_label)):
                source_rows = fit_by_label.get(label)
                target_rows = target_by_label[label]
                if source_rows is None or len(source_rows) == 0:
                    raise ValueError(f"simATAC Training 缺少目标类别：{label}")
                input_path = INTERMEDIATE_DIR / f"group-{group_index:03d}-input.mtx"
                output_path = INTERMEDIATE_DIR / f"group-{group_index:03d}-output.mtx"
                expected_shape = (len(target_rows), len(feature_names))
                generated = None
                if reuse_intermediates and output_path.is_file():
                    cached = mmread(output_path)
                    cached = cached.toarray() if sp.issparse(cached) else np.asarray(cached)
                    cached = np.asarray(cached, dtype=np.float32).T
                    if (
                        cached.shape == expected_shape
                        and np.isfinite(cached).all()
                        and not np.any(cached < 0)
                    ):
                        generated = cached
                        reused_labels.append(label)
                if generated is None and RUN_MODE == "replay":
                    raise ValueError(f"历史 simATAC 中间结果缺失或无效：{label}")
                if generated is None:
                    local = np.asarray(matrix[source_rows], dtype=np.float32)
                    local = np.where(np.isfinite(local), local, feature_medians)
                    if not np.isfinite(local).all():
                        raise ValueError(f"simATAC 类别 {label} 的输入仍含非有限值")
                    mmwrite(input_path, sp.csc_matrix(local.T))
                    del local
                    completed = subprocess.run(
                        [
                            rscript, "-", str(input_path), str(output_path),
                            str(len(target_rows)), str(R_LIBRARY),
                            "1" if INSTALL_MISSING else "0",
                            str(SEED + group_index),
                        ],
                        input=SIMATAC_R, text=True, capture_output=True, check=False,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"simATAC 类别 {label} 运行失败：{completed.stderr[-2000:]}"
                        )
                    process_output = completed.stdout + completed.stderr
                    if "estimation_fallback=single_gaussian" in process_output:
                        fallback_labels.append(label)
                    generated = mmread(output_path)
                    generated = generated.toarray() if sp.issparse(generated) else np.asarray(generated)
                    generated = np.asarray(generated, dtype=np.float32).T
                if generated.shape != expected_shape:
                    raise RuntimeError(
                        f"simATAC 类别 {label} 输出形状错误："
                        f"{generated.shape}，预期 {expected_shape}"
                    )
                if not np.isfinite(generated).all() or np.any(generated < 0):
                    raise ValueError(f"simATAC 类别 {label} 输出不是有限非负矩阵")
                output[target_rows, :] = generated
                print({
                    "Subclass": label,
                    "training_cells": int(len(source_rows)),
                    "synthetic_cells": int(len(target_rows)),
                })
        verified = validate_simatac_output(partial)
        if SIMATAC_PATH.exists():
            SIMATAC_PATH.unlink()
        partial.replace(SIMATAC_PATH)
        summary = {
            "method": "simATAC",
            "input_signature": INPUT_SIGNATURE,
            "feature_axis_hash": AXIS_HASH,
            "split_hash": SPLIT_HASH,
            "seed": SEED,
            "shape": [N_TARGET, len(feature_names)],
            "subclass_counts": pd.Series(target_labels).value_counts().sort_index().to_dict(),
            "native_output": True,
            "library_size_fallback": "single empirical Gaussian after simATAC estimation failure",
            "library_size_fallback_subclasses": fallback_labels,
            "reused_intermediate_subclasses": reused_labels,
            "input_representation": "training-median-filled beta used directly",
            "output_representation": "native simATAC counts or scores; not beta proportions",
            "finite_min": verified["finite_min"],
            "finite_max": verified["finite_max"],
            "elapsed_seconds": time.perf_counter() - started,
            "data": str(SIMATAC_PATH),
        }
        (METHOD_DIR / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return SIMATAC_PATH

    run_simatac()

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
    args = parse_arguments('simATAC', argv)
    return run(args.training, args.target, args.output, args.seed, args.reference, args.replay_root)


if __name__ == "__main__":
    main()

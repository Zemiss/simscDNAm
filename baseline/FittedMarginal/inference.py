"""FittedMarginal 完整区域轴训练与生成。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.paths import PROJECT, OUTPUTS, check_output
from baseline.common import parse_arguments

def run(training_path, target_obs_path, output_dir, seed=42, reference_path=None, replay_root=None):
    METHOD = 'FittedMarginal'
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


    def create_chunked_h5ad(path,obs,var,names):
        partial=path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial')
        shell=ad.AnnData(X=sp.csr_matrix((len(obs),len(names)),dtype=np.float32),obs=obs.copy(),var=var.copy())
        shell.var_names=pd.Index(names.astype(str))
        shell.write_h5ad(partial,compression='gzip')
        with h5py.File(partial,'r+',locking=False) as handle:
            del handle['X']
            data=handle.create_dataset('X',shape=(len(obs),len(names)),dtype='float32',
                chunks=(min(ROW_BLOCK,len(obs)),min(FEATURE_BLOCK,len(names))),
                compression='gzip',compression_opts=4,shuffle=True,fillvalue=np.nan)
            data.attrs['encoding-type']='array'; data.attrs['encoding-version']='0.2.0'
        return partial

    def exact_observation_mask(n_rows,probabilities,rng):
        quota=np.rint(np.clip(probabilities,0,1)*n_rows).astype(np.int64)
        offsets=rng.integers(0,max(n_rows,1),size=len(probabilities))
        ranks=(np.arange(n_rows)[:,None]+offsets[None,:])%max(n_rows,1)
        return ranks<quota[None,:]

    def run_fitted_marginal():
        method='FittedMarginal'; root=METHOD_DIR; root.mkdir(parents=True,exist_ok=True)
        final=root/'data.h5ad'
        started=time.perf_counter(); partial=create_chunked_h5ad(final,target_obs,feature_var,feature_names)
        observed=0; minimum=math.inf; maximum=-math.inf
        sources={x:train_rows[train_labels==x] for x in np.unique(train_labels)}
        targets={x:np.flatnonzero(target_labels==x) for x in np.unique(target_labels)}
        with h5py.File(partial,'r+',locking=False) as handle:
            out=handle['X']
            for bi,start in enumerate(range(0,len(feature_names),FEATURE_BLOCK)):
                stop=min(start+FEATURE_BLOCK,len(feature_names)); width=stop-start
                block_output=np.full((N_TARGET,width),np.nan,dtype=np.float32)
                for gi,label in enumerate(sorted(targets)):
                    source=np.asarray(matrix[sources[label],start:stop],dtype=np.float64)
                    finite=np.isfinite(source); nobs=finite.sum(0)
                    zeros=((source==0)&finite).sum(0); ones=((source==1)&finite).sum(0)
                    cont=finite&(source>0)&(source<1); nc=cont.sum(0)
                    values=np.where(cont,source,0); sums=values.sum(0); squares=(values*values).sum(0)
                    mean=np.divide(sums,nc,out=np.full(width,0.5),where=nc>0)
                    var=np.maximum(np.divide(squares,nc,out=np.zeros(width),where=nc>0)-mean*mean,0)
                    common=np.divide(mean*(1-mean),var,out=np.zeros(width),where=var>1e-10)-1
                    valid=(nc>=2)&(var>1e-10)&(common>0)&np.isfinite(common)&(mean>0)&(mean<1)
                    alpha=np.where(valid,mean*common,1); beta=np.where(valid,(1-mean)*common,1)
                    pobs=nobs/len(source); p0=np.divide(zeros,nobs,out=np.zeros(width),where=nobs>0)
                    p1=np.divide(ones,nobs,out=np.zeros(width),where=nobs>0)
                    dst=targets[label]; rng=np.random.default_rng(SEED+20000+bi*1009+gi)
                    obs=exact_observation_mask(len(dst),pobs,rng); mix=rng.random((len(dst),width))
                    beta_values=rng.beta(alpha[None,:],beta[None,:],size=(len(dst),width))
                    continuous=np.where(valid[None,:],beta_values,mean[None,:])
                    block=np.full((len(dst),width),np.nan,dtype=np.float32)
                    z=obs&(mix<p0[None,:]); o=obs&(mix>=p0[None,:])&(mix<(p0+p1)[None,:]); c=obs&~z&~o
                    block[z]=0; block[o]=1; block[c]=continuous[c].astype(np.float32)
                    if np.isinf(block).any(): raise ValueError(f'Infinite {method} block')
                    block_output[dst]=block
                    local=block[np.isfinite(block)]
                    if local.size:
                        observed+=local.size; minimum=min(minimum,float(local.min())); maximum=max(maximum,float(local.max()))
                out[:,start:stop]=block_output
                if bi%50==0: print(f'{method}: {start:,}:{stop:,}')
        partial.replace(final)
        return final

    run_fitted_marginal()

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
    args = parse_arguments('FittedMarginal', argv)
    return run(args.training, args.target, args.output, args.seed, args.reference, args.replay_root)


if __name__ == "__main__":
    main()

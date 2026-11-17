"""主模型训练、内部选模与正式结果导出。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc, hashlib, json, shutil
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from model.src.config import TrainingConfig
from data.training import load_training_input, load_target_labels, internal_subclass_split, fit_feature_medians, validate_real_duplicates
from model.src.model import load_cvae_model


from data.paths import PROJECT, OUTPUTS, check_output, require_environment
from model.src.generation import write_synthetic_h5ad, validate_synthetic_h5ad
from model.src.postprocessing import (write_training_mask_h5ad, finalize_model, copy_checkpoint, write_json)
from model.src.training import fit_selection_reference, selection_rank, train_candidate

CASES = (("joint_beta_1e4", 1e-4, False), ("frozen_beta_1e4", 1e-4, True),
         ("frozen_beta_1e5", 1e-5, True), ("frozen_beta_1e6", 1e-6, True))

def guard_run_identity(directory, identity):
    """签名在读取/覆盖缓存之前检查；不同输入、代码或配置不允许恢复。"""
    directory = Path(directory)
    path = directory / "run_identity.json"
    normalized = json.loads(json.dumps(identity, sort_keys=True, ensure_ascii=False))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != normalized:
        raise ValueError("运行签名变化，请使用新输出目录；禁止复用旧模型或合成矩阵")
    if not path.exists() and any(directory.glob("*/diagnostics.json")):
        raise ValueError("已有候选缺少运行签名，请使用新输出目录")
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, normalized)

def write_status(dataset_root, phase, **extra):
    payload = {"阶段": phase, "更新时间": datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra}
    partial = dataset_root / "status.json.partial"
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(dataset_root / "status.json")


def expanded_heldout(raw_path,output_path,training,pairs):
    """原始轴追加真实复制列；完整轴核验后保留，合成列不在此处理。"""
    output_path=Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'真实留出输出已存在：{output_path}')
    raw=ad.read_h5ad(raw_path,backed='r')
    try:
        expanded=raw.var_names.equals(training.var_names)
        if not expanded and not raw.var_names.equals(training.var_names[:-1000]):
            raise ValueError('留出与训练的原始或扩维区域轴不一致')
        if len(raw.obs_names.intersection(training.obs_names)):
            raise ValueError('真实训练与外层留出细胞重叠')
        if expanded:
            validate_real_duplicates(raw,pairs)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        shell=ad.AnnData(X=sp.csr_matrix((raw.n_obs,training.n_vars),dtype=np.float32),
                         obs=raw.obs.copy(),var=training.var.copy())
        partial=output_path.with_suffix('.h5ad.partial')
        shell.write_h5ad(partial)
        original=pairs['原region索引'].to_numpy(dtype=int)
        with h5py.File(partial,'r+') as handle:
            del handle['X']
            values=handle.create_dataset('X',shape=(raw.n_obs,training.n_vars),dtype='float32',
                chunks=(min(16,raw.n_obs),min(4096,training.n_vars)),compression='gzip',compression_opts=4)
            values.attrs['encoding-type'],values.attrs['encoding-version']='array','0.2.0'
            for start in range(0,raw.n_obs,16):
                block=raw.X[start:start+16]
                block=np.asarray(block.toarray() if sp.issparse(block) else block,dtype=np.float32)
                values[start:start+len(block)]=block if expanded else np.column_stack([block,block[:,original]])
        partial.replace(output_path)
    finally:
        raw.file.close()


def validate_pretrained_inputs(training_path,cached_path,audit_path,masked_path,target_path):
    """逐值核对既有训练缓存，同时确认当前生成配额仍适用。"""
    truth=ad.read_h5ad(training_path,backed='r')
    masked=ad.read_h5ad(masked_path,backed='r')
    cached=np.load(cached_path,mmap_mode='r')
    try:
        audit=pd.read_csv(audit_path)
        labels=truth.obs['Subclass'].astype(str).to_numpy()
        if (truth.shape!=cached.shape or not truth.var_names.equals(masked.var_names)
            or not np.array_equal(truth.obs_names.astype(str),audit['cell_id'].astype(str))
            or not np.array_equal(labels,audit['Subclass'].astype(str))):
            raise ValueError('训练缓存的细胞、标签或区域轴与当前训练输入不一致')
        for start in range(0,truth.n_obs,64):
            block=truth.X[start:start+64]
            values=np.asarray(block.toarray() if sp.issparse(block) else block,dtype=np.float32)
            if not np.array_equal(values,cached[start:start+64],equal_nan=True):
                raise ValueError('当前训练矩阵与checkpoint原训练缓存不一致')
        target=load_target_labels(target_path,truth.var_names.to_numpy(),np.unique(labels))
        counts=pd.Series(target).value_counts().sort_index()
        generated=masked.obs['Subclass'].astype(str).value_counts().sort_index()
        if not (counts.index.equals(generated.index) and np.array_equal(counts.to_numpy(),generated.to_numpy())):
            raise ValueError('当前目标类型配额与已有生成结果不一致，请从头生成')
        return labels,truth.var_names.to_numpy()
    finally:
        truth.file.close()
        masked.file.close()
        cached._mmap.close()

def run_dataset(dataset, training_path, target_path, pair_path, run_root, config, device,
                pretrained_run=None, cases=None):
    cases = CASES if cases is None else cases
    training_path, target_path, pair_path = map(Path, (training_path, target_path, pair_path))
    check_output(Path(run_root) / dataset)
    dataset_root = run_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    if pretrained_run is not None:
        return run_pretrained_dataset(dataset, pretrained_run, dataset_root, training_path, target_path)
    inputs = [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
              for p in (training_path, target_path, pair_path)]
    guard_run_identity(dataset_root, {"config": config.to_dict(), "cases": cases, "inputs": inputs,
        "源码SHA256": release_source_hashes()})
    if (dataset_root / "completed.json").exists():
        print("复用签名一致的已完成数据集：", dataset, flush=True)
        return json.loads((dataset_root / "completed.json").read_text(encoding="utf-8"))
    write_status(dataset_root, "读取训练数据")
    data = load_training_input(training_path, dataset_root / "work", pair_path, config.INPUT_ROW_BLOCK)
    matrix = data["matrix"]
    try:
        retained, metadata = data["retained_rows"], data["cell_metadata"]
        labels_text = metadata.loc[retained, "Subclass"].to_numpy(dtype=str)
        classes = np.sort(np.unique(labels_text))
        labels = np.searchsorted(classes, labels_text)
        fit, validation = internal_subclass_split(np.arange(len(retained)), labels_text,
            config.INTERNAL_VALIDATION_FRACTION, config.SEED)
        medians = fit_feature_medians(matrix, retained[fit], config.FEATURE_BLOCK)
        labels_by_row = np.full(matrix.shape[0], -1, dtype=int)
        labels_by_row[retained] = labels
        reference = fit_selection_reference(matrix, retained[fit], retained[validation], labels_by_row,
            natural_count=matrix.shape[1]-1000)
        np.savez_compressed(dataset_root / "internal_reference.npz", selected=reference["selected"],
            fill=reference["fill"], pca_mean=reference["pca"].mean_, pca_components=reference["pca"].components_,
            fit_rows=retained[fit], validation_rows=retained[validation])
        names, var = data["feature_names"], data["feature_var"]
        data["region_pairs"].to_csv(dataset_root / "region_pairs.csv", index=False)
        (dataset_root / "config.json").write_text(json.dumps({"配置": config.to_dict(),
            "候选": cases, "选模": "只用内部验证，真实留出矩阵不参与", "标签": "Subclass",
            "输入": str(training_path), "输入区域数": int(matrix.shape[1]),
            "内部训练细胞": int(len(fit)), "内部验证细胞": int(len(validation)),
            "评分区域数": int(len(reference["selected"])), "评分PC数": int(reference["pca"].n_components_)},
            ensure_ascii=False, indent=2), encoding="utf-8")
        records = []
        for name, beta, frozen in cases:
            candidate = dataset_root / name
            if (candidate / "diagnostics.json").exists():
                record = json.loads((candidate / "diagnostics.json").read_text(encoding="utf-8"))
            else:
                write_status(dataset_root, "训练", 候选=name)
                record = train_candidate(matrix, retained, medians, labels, classes, fit, validation,
                    beta, candidate, device, {"natural_features": matrix.shape[1]-1000,
                    "duplicate_features": 1000}, config, selection_reference=reference, frozen_vae=frozen)
            record["case"] = name
            records.append(record)
            matched = [r for r in records if float(r["beta_target"]) == 1e-4]
            if len(matched) == 2 and len({r["vae_state_before_diffusion"] for r in matched}) != 1:
                raise AssertionError("联合和冻结对照没有使用相同的预训练初始化")
            pd.DataFrame(records).to_csv(dataset_root / "internal_selection.csv", index=False)
            gc.collect()
        eligible = [record for record in records if record["selection_eligible"]]
        if not eligible:
            raise RuntimeError("所有候选均未通过原重构/多样性验收；不提升正式模型")
        selected = min(eligible, key=selection_rank)
        write_json(dataset_root / "selected.json", selected)
        # 外层留出仅在选模完成后读取；固定每种生成方法的类别配额。
        target_labels = load_target_labels(target_path, names, classes)
        np.random.default_rng(config.SEED).shuffle(target_labels)
        target_classes = np.searchsorted(classes, target_labels)
        final_names = [selected["case"]]
        for name in final_names:
            record = next(record for record in records if record["case"] == name)
            model, checkpoint = load_cvae_model(Path(record["model_path"]), device)
            destination = dataset_root / name / "data.h5ad"
            if not destination.exists():
                write_synthetic_h5ad(destination, model, target_classes, target_labels, var, names,
                    checkpoint["sampling_steps"], device, config.SEED,
                    config.GENERATION_ROW_BLOCK, config.FEATURE_BLOCK)
            validate_synthetic_h5ad(destination, names, len(target_labels))
            del model
            torch.cuda.empty_cache()
        raw_path = dataset_root / selected["case"] / "data.h5ad"
        masked_path = dataset_root / "CVAE_training_mask/data.h5ad"
        if not masked_path.exists():
            write_training_mask_h5ad(raw_path, masked_path, matrix,
                metadata['Subclass'].to_numpy(dtype=str), retained[fit], config.SEED)
        checkpoint_path = masked_path.parent / 'model.pt'
        if not checkpoint_path.exists():
            copy_checkpoint(selected['model_path'], checkpoint_path)
        raw_path.unlink()
        mapped_path = finalize_model(dataset_root, matrix, metadata['Subclass'].to_numpy(dtype=str),
            retained[fit], retained[validation], names, masked_path, checkpoint_path)
        training = ad.read_h5ad(training_path, backed="r")
        try:
            heldout_path = dataset_root / "HeldoutReal/data.h5ad"
            if not heldout_path.exists():
                expanded_heldout(target_path, heldout_path, training, data["region_pairs"])
        finally:
            training.file.close()
        report = {"状态": "完成", "入选候选": selected["case"], "选模依据": "内部验证 miLISI",
                  "方法": "simscDNAm", "模型": str(mapped_path.parent / "model.pt"),
                  "数据": str(mapped_path), "训练输入": str(training_path), "真实留出": str(heldout_path),
                  "因子上限": 32, "投影强度": 1.0}
        write_json(dataset_root / "completed.json", report)
        write_status(dataset_root, "完成", **report)
        return report
    finally:
        matrix._mmap.close()

def run_pretrained_dataset(dataset,source,dataset_root,training_path,target_path):
    """复用已完成的CVAE＋mask产物，重新拟合正式对齐并另存最终模型。"""
    source=Path(source)
    training_path, target_path = Path(training_path), Path(target_path)
    configuration=json.loads((source/'config.json').read_text(encoding='utf-8'))
    if Path(configuration['输入']).resolve()!=training_path.resolve():
        raise ValueError('checkpoint来源与配置的训练输入不一致')
    masked_path=source/'CVAE_training_mask/data.h5ad'
    checkpoint=source/'CVAE_training_mask/model.pt'
    paths=[training_path,target_path,masked_path,checkpoint,source/'internal_reference.npz',
           source/'work/values.float32.npy',source/'work/input_cell_audit.csv']
    guard_run_identity(dataset_root,{'方法':'simscDNAm','模式':'reuse','源目录':str(source),
        '输入':[{'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths],
        '因子上限':32,'投影强度':1.,'源码SHA256':release_source_hashes()})
    if (dataset_root/'completed.json').exists():
        return json.loads((dataset_root/'completed.json').read_text(encoding='utf-8'))
    with np.load(source/'internal_reference.npz') as reference:
        fit,validation=reference['fit_rows'].copy(),reference['validation_rows'].copy()
    labels,names=validate_pretrained_inputs(training_path,source/'work/values.float32.npy',
        source/'work/input_cell_audit.csv',masked_path,target_path)
    matrix=np.load(source/'work/values.float32.npy',mmap_mode='r')
    try:
        mapped=finalize_model(dataset_root,matrix,labels,fit,validation,names,masked_path,checkpoint)
    finally:
        matrix._mmap.close()
    shutil.copy2(source/'region_pairs.csv',dataset_root/'region_pairs.csv')
    heldout_path=dataset_root/'HeldoutReal/data.h5ad'
    truth=ad.read_h5ad(training_path,backed='r')
    try:
        if not heldout_path.exists():
            expanded_heldout(target_path,heldout_path,truth,pd.read_csv(dataset_root/'region_pairs.csv'))
    finally:
        truth.file.close()
    report={'状态':'完成','方法':'simscDNAm','模式':'reuse','数据':str(mapped),
            '模型':str(mapped.parent/'model.pt'),'训练输入':str(training_path),
            '训练缓存目录':str(source.resolve()),
            '真实留出':str(heldout_path),'源checkpoint':str(checkpoint),
            '因子上限':32,'投影强度':1.,'内部fit细胞数':len(fit),'内部验证细胞数':len(validation)}
    write_json(dataset_root/'completed.json',report)
    write_status(dataset_root,'完成',**report)
    return report

def release_source_hashes():
    names = ('model/src/model.py', 'model/src/config.py', 'model/src/training.py',
             'model/src/generation.py', 'model/src/postprocessing.py', 'model/src/pipeline.py',
             'data/training.py', 'evaluate/A3/metrics.py')
    return {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in names}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="训练 simscDNAm 模型并导出正式生成数据")
    parser.add_argument('--dataset', default='liu2023/full', help='数据集标识')
    parser.add_argument('--training', type=Path, help='预处理后的训练 H5AD')
    parser.add_argument('--target', type=Path, help='目标配额 H5AD')
    parser.add_argument('--pairs', type=Path, help='预处理保存的复制区域表')
    parser.add_argument('--output-root', type=Path, default=OUTPUTS / 'model', help='训练输出父目录')
    parser.add_argument('--config', type=Path, help='可选：TrainingConfig JSON 超参数')
    parser.add_argument('--reuse', type=Path, help='复用既有 CVAE 与训练掩码结果的源训练目录')
    parser.add_argument('--device', help='计算设备，例如 cuda:0 或 cpu')
    args = parser.parse_args(argv)
    require_environment()
    prepared = OUTPUTS / 'preprocessing' / args.dataset
    config = TrainingConfig.from_json(args.config) if args.config else TrainingConfig(BETA_TARGETS=(1e-4, 1e-5, 1e-6))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(8)
    report = run_dataset(args.dataset, args.training or prepared / 'TrainingReal.h5ad',
        args.target or prepared / 'HeldoutReal.h5ad', args.pairs or prepared / 'region_pairs.csv',
        args.output_root, config, device, pretrained_run=args.reuse)
    print(json.dumps(report, ensure_ascii=False, indent=2))

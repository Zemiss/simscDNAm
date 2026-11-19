"""checkpoint、训练掩码和已拟合联合秩映射的独立推理。"""
from pathlib import Path
import json
import anndata as ad
import numpy as np
import pandas as pd
import torch
from data.training import load_target_labels
from model.src.model import load_cvae_model
from data.training import _dense_float32


from data.paths import OUTPUTS, check_output, require_environment
from model.src.config import TrainingConfig
from model.src.generation import write_synthetic_h5ad, validate_synthetic_h5ad
from model.src.postprocessing import write_training_mask_h5ad, apply_joint_mapping, feature_axis_hash, write_json

def run_inference(trained_run, training_path, target_obs_path, output_dir, seed=42, device=None, *, sampling_steps=None):
    trained_run, training_path, target_obs_path, output_dir = map(Path, (trained_run, training_path, target_obs_path, output_dir))
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(8)
    check_output(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("推理输出目录非空，请使用新的 output_dir")
    model_root = trained_run / "simscDNAm"
    if not (trained_run / "completed.json").is_file():
        raise ValueError("训练尚未完成，不能运行正式推理")
    training_report = json.loads((trained_run / "completed.json").read_text(encoding="utf-8"))
    cache_run = Path(training_report.get("训练缓存目录", str(trained_run)))
    model, checkpoint = load_cvae_model(model_root / "model.pt", device)
    if sampling_steps is None:
        sampling_steps = min(TrainingConfig.DDIM_SAMPLING_STEPS, model.diffusion_timesteps)
    elif not 1 <= sampling_steps <= model.diffusion_timesteps:
        raise ValueError("DDIM 采样步数必须位于扩散日程范围内")
    training = ad.read_h5ad(training_path, backed="r")
    matrix = None
    try:
        matrix = np.load(cache_run / "work/values.float32.npy", mmap_mode="r")
        audit = pd.read_csv(cache_run / "work/input_cell_audit.csv")
        if (training.shape != matrix.shape
                or training.n_vars != checkpoint["input_dim"]
                or feature_axis_hash(training.var_names) != json.loads((model_root / "alignment/metadata.json").read_text(encoding="utf-8"))["区域轴SHA256"]
                or not np.array_equal(training.obs_names.astype(str), audit["cell_id"].astype(str))
                or not np.array_equal(training.obs["Subclass"].astype(str), audit["Subclass"].astype(str))):
            raise ValueError("训练输入与 checkpoint、缓存细胞或标签不一致")
        for start in range(0, training.n_obs, 32):
            if not np.array_equal(_dense_float32(training.X[start:start+32]), matrix[start:start+32], equal_nan=True):
                raise ValueError("当前训练输入与 checkpoint 原训练缓存不同")
        with np.load(cache_run / "internal_reference.npz") as reference:
            fit_rows = reference["fit_rows"].copy()
        names = training.var_names.to_numpy(dtype=str)
        labels = load_target_labels(target_obs_path, names, checkpoint["class_names"])
        np.random.default_rng(seed).shuffle(labels)
        class_lookup = {str(name): index for index, name in enumerate(checkpoint["class_names"])}
        indices = np.array([class_lookup[label] for label in labels], dtype=np.int64)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path, masked_path = output_dir / "raw.h5ad", output_dir / "masked.h5ad"
        write_synthetic_h5ad(raw_path, model, indices, labels, training.var, names,
            sampling_steps, device, seed)
        validate_synthetic_h5ad(raw_path, names, len(labels))
        write_training_mask_h5ad(raw_path, masked_path, matrix,
            training.obs["Subclass"].astype(str).to_numpy(), fit_rows, seed)
        apply_joint_mapping(masked_path, output_dir / "data.h5ad", model_root / "alignment")
        write_json(output_dir / "run_summary.json", {"状态": "完成", "方法": "simscDNAm",
            "训练目录": str(trained_run), "目标配额来源": str(target_obs_path),
            "生成细胞数": len(labels), "数据": str(output_dir / "data.h5ad"), "种子": seed,
            "DDIM采样步数": sampling_steps})
        print("推理完成：", output_dir / "data.h5ad")
    finally:
        training.file.close()
        if matrix is not None:
            matrix._mmap.close()

    return output_dir / "data.h5ad"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="使用完整 simscDNAm 模型进行独立推理")
    parser.add_argument('--dataset', default='liu2023/full', help='数据集标识')
    parser.add_argument('--trained-run', type=Path, help='包含完成记录、模型和映射的训练目录')
    parser.add_argument('--training', type=Path, help='原训练 H5AD，逐值核对缓存')
    parser.add_argument('--target', type=Path, help='只提供目标 Subclass 配额的 H5AD')
    parser.add_argument('--output', type=Path, help='新的独立推理输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--sampling-steps', type=int, help=f'DDIM 采样步数，默认 {TrainingConfig.DDIM_SAMPLING_STEPS}')
    parser.add_argument('--device', help='计算设备，例如 cuda:0 或 cpu')
    args = parser.parse_args(argv)
    require_environment()
    prepared = OUTPUTS / 'preprocessing' / args.dataset
    run_inference(args.trained_run or OUTPUTS / 'model' / args.dataset,
        args.training or prepared / 'TrainingReal.h5ad', args.target or prepared / 'HeldoutReal.h5ad',
        args.output or OUTPUTS / 'inference' / args.dataset / 'simscDNAm', args.seed, args.device,
        sampling_steps=args.sampling_steps)

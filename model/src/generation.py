"""按训练区域轴分块生成并导出完整矩阵。"""
from pathlib import Path
import math, uuid
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from model.src.model import LatentDiffusionVAE, sample_latent_ddim
from data.training import _close_backed, _dense_float32
import os


def write_synthetic_h5ad(
    output_path: Path,
    model: LatentDiffusionVAE,
    synthetic_class_indices: np.ndarray,
    synthetic_subclass: np.ndarray,
    var: pd.DataFrame,
    feature_names: np.ndarray,
    sampling_steps: int,
    device: torch.device,
    seed: int,
    row_block: int = 8,
    feature_block: int = 4096,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = len(synthetic_subclass)
    obs = pd.DataFrame(
        {"origin": "Synthetic", "generator": "End-to-end conditional latent diffusion VAE", "Subclass": synthetic_subclass},
        index=pd.Index([f"synthetic_{i:06d}" for i in range(1, n_samples + 1)], name="cell_id"),
    )
    partial = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    shell = ad.AnnData(
        X=sp.csr_matrix((n_samples, len(feature_names)), dtype=np.float32),
        obs=obs,
        var=var.copy(),
    )
    shell.var_names = pd.Index(feature_names.astype(str))
    shell.write_h5ad(partial, compression="gzip")
    model.eval()
    with h5py.File(partial, "r+", locking=False) as handle:
        del handle["X"]
        dataset = handle.create_dataset(
            "X", shape=(n_samples, len(feature_names)), dtype="float32",
            chunks=(min(row_block, n_samples), min(feature_block, len(feature_names))),
            compression="gzip", compression_opts=4, shuffle=True,
        )
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"
        with torch.no_grad():
            for start in range(0, n_samples, row_block):
                stop = min(start + row_block, n_samples)
                condition = torch.from_numpy(
                    synthetic_class_indices[start:stop]
                ).long().to(device)
                latent = sample_latent_ddim(
                    model, condition, sampling_steps, seed + start
                )
                values = model.decode(latent).cpu().numpy().astype(np.float32)
                dataset[start:stop] = values
    partial.replace(output_path)


def validate_synthetic_h5ad(
    path: Path, feature_names: np.ndarray, expected_count: int
) -> dict[str, float | int]:
    item = ad.read_h5ad(path, backed="r")
    try:
        if item.shape != (expected_count, len(feature_names)):
            raise AssertionError(f"合成数据形状不符合预期： {item.shape}")
        if not np.array_equal(item.var_names.astype(str).to_numpy(), feature_names.astype(str)):
            raise AssertionError("合成数据与输入特征轴不一致")
        minimum, maximum = math.inf, -math.inf
        for start in range(0, item.n_obs, 32):
            block = _dense_float32(item.X[start:min(start + 32, item.n_obs)])
            if not np.isfinite(block).all():
                raise AssertionError("合成数据包含非有限值")
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
        if minimum < 0 or maximum > 1:
            raise AssertionError("合成值超出 [0, 1]")
        return {
            "n_cells": int(item.n_obs), "n_features": int(item.n_vars),
            "minimum": minimum, "maximum": maximum,
        }
    finally:
        _close_backed(item)

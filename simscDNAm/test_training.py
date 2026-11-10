"""独立 miLISI 实验的训练隔离和评估参照检查。"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def experiment():
    path = Path(__file__).parent / "train.ipynb"
    assert path.is_file(), "缺少保持原架构的 miLISI 实验 notebook"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "milisi_notebook_test"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell.get("metadata", {}).get("definitions"):
            source = "".join(cell["source"])
            exec(compile(ast.parse(source), str(path), "exec"), namespace)
    return namespace


def test_reference_uses_only_fit_cells_and_natural_regions(experiment):
    rng = np.random.default_rng(42)
    labels = np.repeat([0, 1], 24)
    matrix = rng.uniform(0, 1, (48, 1016)).astype(np.float32)
    matrix[:, :4] += labels[:, None] * 0.5
    fit = np.r_[0:16, 24:40]
    validation = np.setdiff1d(np.arange(48), fit)
    first = experiment["fit_selection_reference"](
        matrix, fit, validation, labels, natural_count=16, top_regions=8, components=4)
    changed = matrix.copy()
    changed[validation] = 999
    changed[:, 16:] = -999
    second = experiment["fit_selection_reference"](
        changed, fit, validation, labels, natural_count=16, top_regions=8, components=4)
    assert max(first["selected"]) < 16
    np.testing.assert_array_equal(first["selected"], second["selected"])
    np.testing.assert_array_equal(first["fill"], second["fill"])
    np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)
    assert first["fit_cells"] == len(fit)
    assert first["validation_rep"].shape == (16, 4)


def test_selection_prefers_validated_milisi_after_eligibility(experiment):
    rank = experiment["selection_rank"]
    base = {"selection_eligible": True, "selection_milisi": 1.2, "selection_score": 0.1}
    higher = {**base, "selection_milisi": 1.8, "selection_score": 5.0}
    collapsed = {**higher, "selection_eligible": False, "selection_milisi": 2.0}
    assert rank(higher) < rank(base) < rank(collapsed)


def test_resume_rejects_changed_config_before_overwriting_artifacts(experiment, tmp_path):
    guard = experiment["guard_run_identity"]
    original = {"beta": 1e-4, "cases": [("baseline", False)]}
    guard(tmp_path, original)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"existing-checkpoint")
    guard(tmp_path, original)
    with pytest.raises(ValueError, match="运行签名"):
        guard(tmp_path, {**original, "beta": 1e-5})
    assert checkpoint.read_bytes() == b"existing-checkpoint"
    assert json.loads((tmp_path / "run_identity.json").read_text())["beta"] == 1e-4


def test_frozen_training_keeps_vae_and_checkpoint_architecture(experiment, tmp_path):
    from simscDNAm.config import TrainingConfig
    from simscDNAm.model import LatentDiffusionVAE, load_cvae_model

    rng = np.random.default_rng(42)
    labels = np.repeat([0, 1], 80)
    natural = np.clip(0.2 + labels[:, None] * 0.5 + rng.normal(0, 0.04, (160, 1024)), 0, 1).astype(np.float32)
    matrix = np.column_stack([natural, natural[:, :1000]])
    fit = np.r_[0:60, 80:140]
    validation = np.setdiff1d(np.arange(160), fit)
    medians = np.median(matrix[fit], axis=0)
    reference = experiment["fit_selection_reference"](
        matrix, fit, validation, labels, natural_count=1024, top_regions=32, components=4)
    config = TrainingConfig(HIDDEN_DIM=16, LATENT_DIM=4, CONDITION_DIM=8,
        DENOISER_BLOCKS=1, TRAIN_BATCH_SIZE=32, VAE_PRETRAIN_EPOCHS=16,
        DIFFUSION_WARMUP_EPOCHS=2, JOINT_EPOCHS=2, DIFFUSION_TIMESTEPS=10,
        DDIM_SAMPLING_STEPS=2, SELECTION_INTERVAL=1, SELECTION_FEATURES=16)
    result = experiment["train_candidate"](matrix, np.arange(160), medians,
        labels, np.array(["A", "B"]), fit, validation, 1e-4,
        tmp_path / "candidate", torch.device("cpu"), {}, config,
        selection_reference=reference, frozen_vae=True)
    assert result["vae_state_before_diffusion"] == result["vae_state_after_training"]
    assert result["training_stages"] == ["vae_pretrain", "diffusion_warmup"]
    assert 1 <= result["selection_milisi"] <= 2
    model, checkpoint = load_cvae_model(Path(result["model_path"]), torch.device("cpu"))
    original = LatentDiffusionVAE(2024, 2, 4, 8, 16, 0.1, 10, 1)
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == {
        k: tuple(v.shape) for k, v in original.state_dict().items()}
    assert checkpoint["input_dim"] == 2024

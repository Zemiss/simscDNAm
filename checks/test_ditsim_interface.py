"""DiTSim 官方参数、填补输入和生成配额的轻量接口检查。"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "基线_DiTSim.ipynb"


def notebook_namespace():
    assert NOTEBOOK.is_file(), "缺少 DiTSim 基线 notebook"
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if "baseline-definitions" in cell.get("metadata", {}).get("tags", [])]
    assert cells, "缺少可独立检查的函数定义 cell"
    namespace = {"__name__": "ditsim_notebook_test"}
    for cell in cells:
        source = "".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node, ast.Expr):
                assert isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "sys.path.insert"
            else:
                assert isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign))
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.fixture
def fixture_data(tmp_path):
    matrix = np.asarray([[0.1, 0.5, 0.1], [0.2, 0.6, 0.2], [0.3, 0.7, 0.3]], dtype=np.float32)
    obs = pd.DataFrame({"Subclass": ["乙", "甲", "乙"]}, index=["train1", "train2", "train3"])
    var = pd.DataFrame({"region_source_index": [0, 1, 0]}, index=["r0", "r1", "r0_copy"])
    path = tmp_path / "filled_training.h5ad"
    ad.AnnData(matrix, obs=obs, var=var).write_h5ad(path)
    target = pd.DataFrame({"Subclass": ["甲", "乙", "甲", "甲"]}, index=["held1", "held2", "held3", "held4"])
    return path, target, matrix, var


@pytest.mark.parametrize("pilot", [False, True])
def test_official_training_parameters_and_full_features(tmp_path, fixture_data, pilot):
    namespace = notebook_namespace()
    filled_path, target, matrix, var = fixture_data
    events = []

    class AutoEncoder:
        def __init__(self, **kwargs):
            events.append(("ae_model", kwargs))

    class AutoencoderTrainer:
        def __init__(self, **kwargs):
            events.append(("ae_init", kwargs))

        def train(self, **kwargs):
            events.append(("ae_train", kwargs))

    class DiT:
        def __init__(self, **kwargs):
            events.append(("dit_model", kwargs))

    class DitTrainer:
        def __init__(self, **kwargs):
            events.append(("dit_init", kwargs))

        def load_ae(self, path):
            events.append(("load_ae", path))

        def train(self, **kwargs):
            events.append(("dit_train", kwargs))

        def sampling(self, sample_dict, **kwargs):
            events.append(("sampling", sample_dict, kwargs))
            labels = [label for label, count in sample_dict.items() for _ in range(count)]
            values = np.tile(np.asarray([-0.2, 0.375, 1.2], dtype=np.float32), (len(labels), 1))
            result = ad.AnnData(values, obs=pd.DataFrame({"Subclass": labels}, index=[str(i) for i in range(len(labels))]))
            result.write_h5ad(kwargs["result_savepath"])

    namespace.update(AutoEncoder=AutoEncoder, AutoencoderTrainer=AutoencoderTrainer, DiT=DiT, DitTrainer=DitTrainer)
    clip_calls = []

    original_clip = namespace["clip_generated_matrix"]

    def clip(values):
        clip_calls.append(np.asarray(values).copy())
        return original_clip(values)

    namespace["clip_generated_matrix"] = clip
    output = namespace["train_and_sample"](filled_path, target, tmp_path / "result", seed=17, device="cpu", pilot=pilot)
    calls = {event[0]: event[1] for event in events}
    assert calls["ae_model"] == dict(input_dim=3, latent_dim=256, hidden_dim=[1024, 1024], embed_dim=128, reshape=True, cfg=True, num_classes=2)
    assert calls["dit_model"] == dict(input_size=256, patch_size=2, hidden_size=768, depth=4, num_heads=8, num_classes=2)
    for name in ("ae_init", "dit_init"):
        training = calls[name]["adata"]
        np.testing.assert_array_equal(training.X, matrix)
        pd.testing.assert_frame_equal(training.var, var)
        assert type(training.X) is np.ndarray
        assert calls[name]["label_name"] == "Subclass"
        assert calls[name]["device"] == "cpu"
    assert calls["ae_init"]["adata"] is calls["dit_init"]["adata"]
    ae = calls["ae_train"]
    assert {key: ae[key] for key in ("epochs", "end_loss", "batch_size", "lr", "weight_decay", "cs_loss_ratio", "save_freq")} == dict(epochs=1 if pilot else 200, end_loss=0.01, batch_size=32, lr=5e-4, weight_decay=0.01, cs_loss_ratio=0.0, save_freq=10)
    dit = calls["dit_train"]
    assert {key: dit[key] for key in ("epochs", "batch_size", "lr", "weight_decay", "save_freq")} == dict(epochs=1 if pilot else 2400, batch_size=32, lr=1e-4, weight_decay=0.01, save_freq=100)
    sampling = next(event for event in events if event[0] == "sampling")
    expected_quotas = {"甲": 2 if pilot else 3, "乙": 1}
    assert sampling[1] == expected_quotas
    assert sampling[2]["sampler"] == "ddim"
    assert sampling[2]["acc_rate"] == 10
    assert len(clip_calls) == 1
    generated = ad.read_h5ad(output)
    np.testing.assert_array_equal(generated.X, np.tile([0.0, 0.375, 1.0], (sum(expected_quotas.values()), 1)))
    assert generated.X.dtype == np.float32
    pd.testing.assert_frame_equal(generated.var, var)
    assert generated.obs["Subclass"].value_counts().to_dict() == expected_quotas
    assert not len(generated.obs_names.intersection(target.index))


def test_sparse_container_adaptation_preserves_values(tmp_path, fixture_data):
    namespace = notebook_namespace()
    filled_path, target, matrix, var = fixture_data
    filled = ad.read_h5ad(filled_path)
    filled.X = sparse.csr_matrix(filled.X)
    filled.write_h5ad(filled_path)
    actual = namespace["read_filled_training"](filled_path)
    assert type(actual.X) is np.ndarray
    assert actual.X.dtype == np.float32
    np.testing.assert_array_equal(actual.X, matrix)
    pd.testing.assert_frame_equal(actual.var, var)


def test_quota_rejects_untrained_subclass(fixture_data):
    namespace = notebook_namespace()
    filled_path, target, _, _ = fixture_data
    target.loc["held1", "Subclass"] = "未训练类型"
    with pytest.raises(ValueError, match="未训练类型"):
        namespace["sample_quotas"](ad.read_h5ad(filled_path).obs, target)


def test_filled_input_must_be_finite(fixture_data):
    namespace = notebook_namespace()
    filled_path, _, _, _ = fixture_data
    filled = ad.read_h5ad(filled_path)
    filled.X[0, 0] = np.nan
    filled.write_h5ad(filled_path)
    with pytest.raises(ValueError, match="有限"):
        namespace["read_filled_training"](filled_path)


def test_sources_fixed_and_vendor_unchanged():
    notebook_namespace()
    manifest = json.loads((ROOT / "baseline" / "DiTSim" / "sources.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "52231d81ee7d98c6222ad51539624e1204ebf6a9"
    assert set(manifest["sha256"]) == {"LICENSE", "ae.py", "ae_trainer.py", "diffusion.py", "dit_trainer.py", "ditsim.yml", "loaders.py", "quick_start.py", "transformer.py"}
    for path, expected in manifest["sha256"].items():
        source_path = ROOT / "baseline" / "DiTSim" / "vendor" / path
        if source_path.is_file():
            assert hashlib.sha256(source_path.read_bytes()).hexdigest() == expected


def test_official_sampling_retains_full_width_beyond_latent_dimension(tmp_path):
    namespace = notebook_namespace()
    vendor = ROOT / "baseline" / "DiTSim" / "vendor"
    if not (vendor / "dit_trainer.py").is_file():
        pytest.skip("未缓存官方源码；正式入口会下载并逐文件校验")
    torch = namespace["torch"]
    from torch.utils.data import DataLoader, Dataset

    # 直接编译未改动的官方 class 和 sampling 方法；只用轻量模型替身避开 GPU 训练。
    original_loader = ast.parse((vendor / "loaders.py").read_text(encoding="utf-8"))
    dataset_class = next(node for node in original_loader.body if isinstance(node, ast.ClassDef) and node.name == "ATACDataset")
    trainer_class = next(node for node in ast.parse((vendor / "dit_trainer.py").read_text(encoding="utf-8")).body if isinstance(node, ast.ClassDef) and node.name == "DitTrainer")
    sampling = next(node for node in trainer_class.body if isinstance(node, ast.FunctionDef) and node.name == "sampling")

    def local_loader(dataset, batch_size, num_workers, shuffle):
        assert num_workers == 2 and not shuffle
        return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)

    official = dict(np=np, pd=pd, ad=ad, torch=torch, nn=torch.nn, Dataset=Dataset, DataLoader=local_loader)
    exec(compile(ast.Module(body=[dataset_class, sampling], type_ignores=[]), "official_ditsim_sampling", "exec"), official)
    full_width = 310003

    class Decode:
        input_dim = full_width

        def __call__(self, values, labels, dec=False):
            assert dec
            return torch.full((len(labels), 1, full_width), 0.375)

    class Diffusion:
        def sample_ddim(self, model, batch, labels, sub_time_seq):
            assert sub_time_seq == [0, 1, *range(10, 1001, 10)]
            return torch.zeros(batch, 1, 256)

    trainer = SimpleNamespace(device="cpu", device_ids=None, label_name="Subclass", ae=Decode(), diffusion=Diffusion(), model=SimpleNamespace(input_size=256, label_dict={"甲": 0, "乙": 1, "masked": 100}))
    output = tmp_path / "official_sampling.h5ad"
    with pytest.warns(ad.ImplicitModificationWarning, match="Transforming to str index"):
        official["sampling"](trainer, {"甲": 3, "乙": 1}, result_savepath=str(output), sampler="ddim", acc_rate=10)
    result = ad.read_h5ad(output)
    assert result.shape == (4, full_width)
    assert result.obs["Subclass"].value_counts().to_dict() == {"甲": 3, "乙": 1}
    assert not len(result.var.columns)
    np.testing.assert_array_equal(result.X, np.full((4, full_width), 0.375, dtype=np.float32))


def test_notebook_run_prepares_input_and_separates_pilot(monkeypatch, tmp_path):
    namespace = notebook_namespace()
    events = []
    target = pd.DataFrame({"Subclass": ["甲"]}, index=["held1"])
    filled = tmp_path / "filled.h5ad"
    for key, value in {"BASELINE_TRAINING": str(tmp_path / "original.h5ad"), "BASELINE_TARGET": str(tmp_path / "target.h5ad"), "BASELINE_OUTPUT": str(tmp_path / "out"), "BASELINE_SEED": "23", "BASELINE_DEVICE": "cpu", "BASELINE_PILOT": "1"}.items():
        monkeypatch.setenv(key, value)

    def prepare(training, target_path, cache_dir):
        events.append(("prepare", Path(training), Path(target_path), Path(cache_dir)))
        return filled, target, {"filling": "training_median"}

    def execute(filled_path, target_obs, run_dir, seed, device, pilot=False):
        events.append(("train", filled_path, target_obs, Path(run_dir), seed, device, pilot))
        return Path(run_dir) / "data.h5ad"

    namespace.update(prepare_imputed_training=prepare, train_and_sample=execute, load_official=lambda: None, require_environment=lambda: None, check_output=lambda path: Path(path), ROOT=ROOT)
    output = namespace["run_baseline"]()
    assert events[0][0] == "prepare"
    assert events[1][0] == "train"
    assert events[1][1] == filled
    assert events[1][2] is target
    assert events[1][3] == tmp_path / "out" / "pilot"
    assert events[1][4:] == (23, "cpu", True)
    assert output == tmp_path / "out" / "pilot" / "data.h5ad"

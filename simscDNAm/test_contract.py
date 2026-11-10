"""扩维训练入口的输入契约。"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def expanded(tmp_path):
    rng = np.random.default_rng(42)
    original = rng.uniform(0, 1, (32, 1024)).astype(np.float32)
    original[0, 0] = np.nan
    names = [f"region_{i}" for i in range(1024)]
    item = ad.AnnData(
        np.column_stack([original, original[:, :1000]]),
        obs=pd.DataFrame({"Subclass": np.repeat(["A", "B"], 16)},
                         index=[f"cell_{i}" for i in range(32)]),
        var=pd.DataFrame(index=names + [f"{name}__duplicate_{i:04d}" for i, name in enumerate(names[:1000])]),
    )
    path = tmp_path / "expanded.h5ad"
    item.write_h5ad(path)
    return item, path


def test_input_keeps_all_dimensions_and_missingness(expanded, tmp_path):
    from simscDNAm.data import load_training_input
    item, path = expanded
    result = load_training_input(path, tmp_path / "work")
    try:
        np.testing.assert_array_equal(result["matrix"], item.X)
        assert result["matrix"].shape == (32, 2024)
        assert len(result["region_pairs"]) == 1000
        np.testing.assert_array_equal(result["feature_names"], item.var_names)
        np.testing.assert_array_equal(result["cell_metadata"]["cell_id"], item.obs_names)
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "work/input_cell_audit.csv"), result["cell_metadata"])
    finally:
        result["matrix"]._mmap.close()


def test_invalid_tail_is_rejected_before_training(expanded, tmp_path):
    from simscDNAm.data import load_training_input
    item, path = expanded
    item.X[2, -1] += 0.01
    item.write_h5ad(path)
    with pytest.raises(ValueError, match="复制"):
        load_training_input(path, tmp_path / "work")


def test_missing_copy_is_rejected(expanded, tmp_path):
    from simscDNAm.data import load_training_input
    item, path = expanded
    item[:, :1024].copy().write_h5ad(path)
    with pytest.raises(ValueError):
        load_training_input(path, tmp_path / "work")


def test_explicit_map_and_sparse_backed_input(expanded, tmp_path):
    from scipy.sparse import csr_matrix
    from simscDNAm.data import load_training_input
    item, path = expanded
    item.X = csr_matrix(item.X)
    item.var_names = [f"x{i}" for i in range(2024)]
    item.write_h5ad(path)
    mapping = tmp_path / "region_pairs.csv"
    pd.DataFrame({"original_index": np.arange(1000),
                  "duplicate_index": np.arange(1024, 2024)}).to_csv(mapping, index=False)
    result = load_training_input(path, tmp_path / "work", mapping)
    try:
        np.testing.assert_array_equal(result["matrix"], item.X.toarray())
    finally:
        result["matrix"]._mmap.close()


def test_target_feature_axis_and_classes(expanded, tmp_path):
    from simscDNAm.data import load_target_labels
    item, path = expanded
    labels = load_target_labels(path, item.var_names.to_numpy(), np.array(["A", "B"]))
    np.testing.assert_array_equal(labels, item.obs["Subclass"])
    item.var_names = item.var_names[::-1]
    item.write_h5ad(path)
    with pytest.raises(ValueError, match="特征轴"):
        load_target_labels(path, item.var_names[::-1].to_numpy(), np.array(["A", "B"]))

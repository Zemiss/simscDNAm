import anndata as ad
import numpy as np
import pandas as pd

from data.preprocessing import filter_training_subclasses


def test_filter_uses_training_counts_for_both_splits(tmp_path):
    def write(name, labels):
        obs = pd.DataFrame({"Subclass": pd.Categorical(labels)}, index=[f"{name}{i}" for i in range(len(labels))])
        values = np.arange(2 * len(labels), dtype=np.float32).reshape(-1, 2)
        matrix = ad.AnnData(values, obs=obs)
        matrix.write_h5ad(tmp_path / f"{name}.h5ad")
        return matrix

    training = write("train", ["rare", "keep"] * 9 + ["keep"] + ["other"] * 11)
    target = write("test", ["keep", "rare", "unseen", "other", "rare"])
    output = tmp_path / "filtered"
    counts = filter_training_subclasses(
        tmp_path / "train.h5ad", tmp_path / "test.h5ad", output, min_cells=10
    )
    for item, name, expected in (
        (training, "TrainingReal", [label != "rare" for label in training.obs["Subclass"]]),
        (target, "HeldoutReal", [True, False, False, True, False]),
    ):
        kept = ad.read_h5ad(output / f"{name}.h5ad")
        assert kept.obs_names.equals(item.obs_names[expected])
        assert kept.var_names.equals(item.var_names)
        np.testing.assert_array_equal(kept.X, item.X[expected])
    assert counts.loc["keep", "保留"]
    assert not counts.loc["rare", "保留"]
    assert counts.loc["unseen", "训练细胞数"] == 0

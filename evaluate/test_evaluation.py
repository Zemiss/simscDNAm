"""统一评估的数值、输入校验和复制尾部隔离回归测试。"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse
from scipy.stats import pearsonr, wasserstein_distance

from evaluate.preprocessing import (
    dense_block, read_labels, region_pairs, validate_real_duplicates,
    natural_reference_statistics, select_features, imputed_features,
    validate_region_axis, sample_rows,
)
from evaluate.metrics import a1_metrics, column_wasserstein_1, paired_pearson, milisi


@pytest.fixture
def expanded():
    rng = np.random.default_rng(42)
    labels = np.repeat(["A", "B"], 16)
    values = rng.uniform(0.05, 0.95, (32, 1004))
    values[labels == "B", :20] *= 0.2
    values[0, 0] = np.nan
    copied = np.arange(1000)
    names = [f"region_{i}" for i in range(values.shape[1])]
    var_names = names + [f"{names[i]}__duplicate_{j:04d}" for j, i in enumerate(copied)]
    result = ad.AnnData(
        np.column_stack([values, values[:, copied]]),
        obs=pd.DataFrame({"Subclass": labels}, index=[f"real_{i}" for i in range(32)]),
        var=pd.DataFrame(index=var_names),
    )
    return result


def test_pair_mapping_and_real_copy_validation(expanded):
    pairs = region_pairs(expanded)
    assert len(pairs) == 1000
    np.testing.assert_array_equal(pairs["原region索引"], np.arange(1000))
    np.testing.assert_array_equal(pairs["复制region索引"], np.arange(1004, 2004))
    validate_real_duplicates(expanded, pairs)
    expanded.X[0, -1] += 0.1
    with pytest.raises(ValueError, match="复制"):
        validate_real_duplicates(expanded, pairs)


def test_explicit_mapping_validates_names_and_tail(expanded, tmp_path):
    pairs = region_pairs(expanded)
    path = tmp_path / "region_pairs.csv"
    pairs.to_csv(path, index=False)
    pd.testing.assert_frame_equal(region_pairs(expanded, pair_path=path), pairs)
    pairs.loc[0, "原region"] = "wrong"
    pairs.to_csv(path, index=False)
    with pytest.raises(ValueError, match="名称"):
        region_pairs(expanded, pair_path=path)
    pairs = region_pairs(expanded)
    pairs.loc[0, "复制region索引"] = 1003
    pairs.to_csv(path, index=False)
    with pytest.raises(ValueError, match="最后"):
        region_pairs(expanded, pair_path=path)


def test_unknown_copy_names_require_mapping(expanded):
    expanded.var_names = [f"x{i}" for i in range(expanded.n_vars)]
    with pytest.raises(ValueError, match="region_pairs.csv"):
        region_pairs(expanded)


def test_reordered_region_axis_is_rejected(expanded):
    candidate = expanded.copy()
    candidate.var_names = expanded.var_names[::-1]
    with pytest.raises(ValueError, match="顺序"):
        validate_region_axis(expanded, candidate)


def test_sampling_is_reproducible_and_never_upsamples(expanded):
    first = sample_rows(expanded, 12)
    np.testing.assert_array_equal(first, sample_rows(expanded, 12))
    assert len(np.unique(first)) == 12
    with pytest.raises(ValueError, match="无放回"):
        sample_rows(expanded, 33)


@pytest.mark.parametrize("backed_sparse", [False, True])
def test_backed_selection_preserves_rows_columns_and_missing(expanded, tmp_path, backed_sparse):
    if backed_sparse:
        expanded.X = sparse.csr_matrix(expanded.X)
    path = tmp_path / "input.h5ad"
    expanded.write_h5ad(path)
    item = ad.read_h5ad(path, backed="r")
    try:
        rows, columns = np.array([1, 4, 9]), np.array([0, 10, 1004, 2000])
        expected = expanded.X.toarray() if backed_sparse else expanded.X
        np.testing.assert_allclose(dense_block(item, rows, columns), expected[np.ix_(rows, columns)])
    finally:
        item.file.close()


def test_wasserstein_matches_scipy_with_unequal_counts():
    reference = np.array([[0.1, np.nan, 0.5], [0.6, 0.0, np.nan], [0.9, 0.8, np.nan]])
    candidate = np.array([[0.2, 0.4, np.nan], [0.3, np.nan, np.nan]])
    actual = column_wasserstein_1(reference, candidate)
    for j in (0, 1):
        left, right = reference[:, j], candidate[:, j]
        assert actual[j] == pytest.approx(wasserstein_distance(left[np.isfinite(left)], right[np.isfinite(right)]))
    assert np.isnan(actual[2])


def test_a1_matches_existing_metric_definitions(expanded):
    candidate = expanded.copy()
    candidate.X[:, :1004] += 0.01
    summary, details = a1_metrics(expanded, candidate, np.arange(32), np.arange(32), 1004)
    assert summary["加权 1-WD"] == pytest.approx(0.01)
    assert summary["加权 MSE"] == pytest.approx(0.0001)
    assert summary["加权 Pearson"] == pytest.approx(1.0)
    assert len(details) == 2


def test_a1_and_a3_ignore_all_appended_columns(expanded):
    changed = expanded.copy()
    changed.X[:, 1004:] = 100000.0
    rows = np.arange(expanded.n_obs)
    baseline, _ = a1_metrics(expanded, expanded, rows, rows, 1004)
    actual, _ = a1_metrics(expanded, changed, rows, rows, 1004)
    assert actual == baseline
    fill, scores = natural_reference_statistics(expanded, rows, 1004)
    changed_fill, changed_scores = natural_reference_statistics(changed, rows, 1004)
    np.testing.assert_allclose(fill, changed_fill)
    np.testing.assert_allclose(scores, changed_scores, equal_nan=True)
    selected = select_features(scores, 100)
    assert selected.max() < 1004
    np.testing.assert_allclose(imputed_features(expanded, rows, selected, fill),
                               imputed_features(changed, rows, selected, fill))


def test_missing_type_is_not_silently_reweighted(expanded):
    rows = np.arange(32)
    candidate_rows = np.arange(16)
    summary, details = a1_metrics(expanded, expanded, rows, candidate_rows, 1004)
    assert np.isnan(summary["加权 1-WD"])
    assert summary["有效细胞类型权重"] == pytest.approx(0.5)
    assert details.loc[details["细胞类型"] == "B", "合成细胞数"].item() == 0


def test_missing_label_is_rejected(expanded):
    expanded.obs.iloc[0, 0] = None
    with pytest.raises(ValueError, match="标签"):
        read_labels(expanded, np.arange(32))


def test_a2_uses_actual_output_and_reports_invalid_pairs(expanded):
    pairs = region_pairs(expanded)
    original = dense_block(expanded, np.arange(32), pairs["原region索引"].to_numpy())
    copied = dense_block(expanded, np.arange(32), pairs["复制region索引"].to_numpy())
    perfect = paired_pearson(original, copied)
    np.testing.assert_allclose(perfect["Pearson"], 1.0)
    copied[:, 0] = 0.5
    copied[:, 1] = original[::-1, 1]
    copied[2:, 2] = np.nan
    result = paired_pearson(original, copied)
    assert np.isnan(result.loc[0, "Pearson"])
    assert np.isnan(result.loc[2, "Pearson"])
    assert result.loc[1, "Pearson"] == pytest.approx(pearsonr(original[:, 1], copied[:, 1]).statistic)
    assert result.loc[2, "有效细胞数"] == 2


def test_milisi_matches_direct_neighborhood_formula():
    # 坐标无并列距离，直接穷举邻居作为独立数值参照。
    real = np.array([[0.0], [1.3], [4.1], [7.2]])
    synthetic = np.array([[0.2], [2.0], [5.0], [9.4]])
    labels = np.array(["A", "A", "B", "B"])
    score, cells, groups = milisi(real, labels, synthetic, labels, n_neighbors=3)
    combined = np.vstack([real, synthetic])[:, 0]
    distance = np.abs(combined[:, None] - combined[None, :])
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :3]
    p = (nearest < 4).mean(axis=1)
    expected = 1 / (p ** 2 + (1 - p) ** 2)
    np.testing.assert_allclose(cells["miLISI"], expected)
    assert score == pytest.approx(pd.Series(expected).groupby(np.tile(labels, 2)).median().mean())
    assert len(groups) == 2

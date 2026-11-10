"""直接检查 notebook 中的输入约束与逐元素一致性校验。"""
from pathlib import Path

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


ROOT = Path(__file__).parent
METHODS = ('FittedMarginal', 'simATAC', 'simCAS')


def definitions(method):
    path = ROOT / f'{method}.ipynb'
    assert path.is_file(), f'缺少 baseline notebook：{method}'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    scope = {}
    for cell in notebook.cells:
        if '校验函数' in cell.metadata.get('tags', []):
            exec(compile(cell.source, str(path), 'exec'), scope)
    return scope


def real_data():
    values = np.linspace(0, 1, 4008, dtype=np.float32).reshape(4, 1002)
    values[0, 8] = np.nan
    names = [f'region_{i}' for i in range(1002)]
    copied = [f'{names[i]}__duplicate_{i:04d}' for i in range(1000)]
    return ad.AnnData(
        X=np.c_[values, values[:, :1000]],
        obs=pd.DataFrame({'Subclass': ['A', 'B', 'A', 'B']}, index=list('abcd')),
        var=pd.DataFrame(index=names + copied),
    )


@pytest.mark.parametrize('method', METHODS)
def test_training_preserves_full_axis_missingness_and_sparse_zeros(tmp_path, method):
    funcs = definitions(method)
    item = real_data()
    path = tmp_path / 'training.h5ad'
    for sparse in (False, True):
        if sparse:
            item.X = sp.csr_matrix(item.X)
        item.write_h5ad(path)
        matrix, obs, var, pairs = funcs['read_training'](path)
        assert matrix.shape == (4, 2002)
        assert obs.index.equals(item.obs_names)
        assert var.index.equals(item.var_names)
        assert len(pairs) == 1000
        np.testing.assert_array_equal(matrix[:, :1000], matrix[:, 1002:])
        assert np.isnan(matrix[0, 8]) and matrix[0, 0] == 0


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('damage', ['values', 'mask', 'axis'])
def test_rejects_invalid_training_duplicates(tmp_path, method, damage):
    funcs = definitions(method)
    item = real_data()
    if damage == 'values':
        item.X[0, -1] += 0.25
    elif damage == 'mask':
        item.X[0, 1010] = 0
    else:
        item.var_names = list(item.var_names[:-1]) + ['unknown__duplicate_0999']
    path = tmp_path / 'damaged.h5ad'
    item.write_h5ad(path)
    with pytest.raises(ValueError, match='复制'):
        funcs['read_training'](path)


@pytest.mark.parametrize('method', METHODS)
def test_full_output_comparison_catches_tail_and_nan_changes(tmp_path, method):
    funcs = definitions(method)
    item = real_data()
    reference, candidate = tmp_path / 'old.h5ad', tmp_path / 'new.h5ad'
    item.write_h5ad(reference)
    item.write_h5ad(candidate)
    assert funcs['compare_h5ad'](candidate, reference)['一致']
    for value in (0.123, np.nan):
        changed = item.copy()
        changed.X[-1, -1] = value
        changed.write_h5ad(candidate)
        with pytest.raises(AssertionError, match='数值|缺失'):
            funcs['compare_h5ad'](candidate, reference)
    changed = item.copy()
    changed.X = changed.X.astype(np.float64)
    changed.X[-1, -1] += 1e-12
    changed.write_h5ad(candidate)
    with pytest.raises(AssertionError, match='数值'):
        funcs['compare_h5ad'](candidate, reference)
    changed = item.copy()
    changed.X[0, 1] = 0.987
    changed.write_h5ad(candidate)
    with pytest.raises(AssertionError, match='数值'):
        funcs['compare_h5ad'](candidate, reference)

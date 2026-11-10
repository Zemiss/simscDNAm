"""主模型入口只应用同类型内部训练缺失掩码，保留模型数值。"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def model_notebook():
    path = Path(__file__).parent / 'train.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    scope = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('metadata', {}).get('definitions'):
            exec(''.join(cell['source']), scope)
    return scope


def test_mask_donors_only_use_matching_fit_cells(model_notebook):
    assert 'sample_mask_rows' in model_notebook, '主模型入口缺少同类型训练掩码'
    labels = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    fit = np.array([0, 2])
    target = np.array(['B', 'A', 'B', 'A'])
    donor = model_notebook['sample_mask_rows'](labels, fit, target, 42)
    np.testing.assert_array_equal(donor, [2, 0, 2, 0])
    with pytest.raises(ValueError, match='细胞类型'):
        model_notebook['sample_mask_rows'](labels, fit, np.array(['C']), 42)


@pytest.mark.parametrize('sparse', [False, True])
def test_export_preserves_generated_values_zero_and_duplicate_axis(model_notebook, tmp_path, sparse):
    assert 'write_training_mask_h5ad' in model_notebook, '主模型入口缺少训练掩码导出'
    raw = np.array([[0, .2, .4, .9], [.1, .3, .8, .7]], dtype=np.float32)
    training = np.array([[np.nan, 0, 1, np.nan], [1, np.nan, np.nan, 0],
                         [1, 1, 1, 1]], dtype=np.float32)
    labels = np.array(['A', 'B', 'A'])
    names = ['region_0', 'region_1', 'region_0__duplicate_0000', 'region_1__duplicate_0001']
    item = ad.AnnData(sp.csr_matrix(raw) if sparse else raw,
        obs=pd.DataFrame({'Subclass': ['B', 'A']}, index=['s0', 's1']),
        var=pd.DataFrame(index=names))
    source = tmp_path / 'raw.h5ad'
    item.write_h5ad(source)
    original = source.read_bytes()
    output = tmp_path / 'CVAE_training_mask/data.h5ad'
    model_notebook['write_training_mask_h5ad'](source, output, training, labels,
        np.array([0, 1]), seed=42, feature_block=2)
    generated = ad.read_h5ad(output)
    expected = np.where(np.isfinite(training[[1, 0]]), raw, np.nan)
    np.testing.assert_array_equal(generated.X, expected)
    assert generated.X[0, 0] == 0
    assert generated.var_names.tolist() == names
    assert generated.obs['Subclass'].astype(str).tolist() == ['B', 'A']
    assert generated.uns['postprocessing']['method'] == 'training_mask'
    np.testing.assert_array_equal(generated.obs['mask_donor_row'], [1, 0])
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        model_notebook['write_training_mask_h5ad'](source, output, training, labels, np.array([0, 1]))

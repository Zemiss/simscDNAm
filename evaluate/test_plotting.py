"""核对参考风格图中的概率和正式均值，避免重画改变数值含义。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
import numpy as np
import pandas as pd
import pytest
import anndata as ad


def plotting_namespace():
    notebook = nbformat.read(Path(__file__).with_name("A1A2A3.ipynb"), as_version=4)
    namespace = {"plt": plt, "np": np, "pd": pd, "Line2D": Line2D}
    exec(notebook.cells[5].source, namespace)
    return namespace


def test_histogram_reports_outside_mass_without_clipping_values():
    namespace = plotting_namespace()
    assert "probability_histogram" in namespace, "缺少统一分箱及越界比例计算"
    values = np.array([-0.2, 0.0, 0.25, 1.0, 1.3, np.nan])
    original = values.copy()
    probabilities, outside, count = namespace["probability_histogram"](values, np.array([0, 0.5, 1]))
    np.testing.assert_array_equal(probabilities, [0.4, 0.2])
    assert outside == 0.4 and count == 5
    assert probabilities.sum() + outside == 1
    np.testing.assert_array_equal(values, original)


def test_histogram_empty_region_has_no_invented_probability():
    namespace = plotting_namespace()
    assert "probability_histogram" in namespace, "缺少统一分箱及越界比例计算"
    probabilities, outside, count = namespace["probability_histogram"]([np.nan], np.array([0, 0.5, 1]))
    np.testing.assert_array_equal(probabilities, [0, 0])
    assert np.isnan(outside) and count == 0


def test_a2_bars_use_formal_means_and_preserve_negative_values(tmp_path):
    namespace = plotting_namespace()
    assert "plot_duplicate_pearson" in namespace, "缺少复制区域Pearson双面板图"
    frames = [pd.DataFrame({"方法": name, "Pearson": values}) for name, values in [
        ("真实数据", [1, 1, np.nan, 1]),
        ("CVAE_training_mask", [0.2, 0.6, 1, np.nan]),
        ("FittedMarginal", [-0.25, -0.25, -0.25, -0.25])]]
    correlations = pd.concat(frames, ignore_index=True)
    original = correlations.copy(deep=True)
    summary = correlations.groupby("方法", sort=False).agg(
        配对数=("Pearson", "size"), 有效配对数=("Pearson", "count"),
        Pearson均值=("Pearson", "mean")).reset_index()
    captured = []
    namespace["save_figure"] = lambda figure, output, stem: captured.append(figure)
    namespace["plot_duplicate_pearson"](correlations, summary, tmp_path)
    figure = captured[0]
    try:
        assert len(figure.axes) == 2
        np.testing.assert_allclose([patch.get_height() for patch in figure.axes[1].patches], [0.6, -0.25])
        assert figure.axes[1].get_ylim()[0] <= -0.25
        labels = {text.get_text() for text in figure.axes[1].texts}
        assert "0.6000" in labels and "-0.2500" in labels
        pd.testing.assert_frame_equal(correlations, original)
    finally:
        plt.close(figure)


def test_marker_manifest_rejects_wrong_assembly_and_duplicate_tail(tmp_path):
    namespace = plotting_namespace()
    assert 'load_plot_regions' in namespace, '选区必须由显式文献清单提供，不随机回退'
    real = ad.AnnData(np.zeros((3, 3)), var=pd.DataFrame(index=['r0', 'r1', 'r0__duplicate_0000']))
    path = tmp_path / 'markers.csv'
    table = pd.DataFrame({'原region': ['r0'], '展示名称': ['Test marker'], '基因组版本': ['test'],
        '文献来源': ['test fixture'], '单区域展示': [1], '展示细胞类型': ['A|B']})
    table.to_csv(path, index=False)
    selected = namespace['load_plot_regions'](real, 2, path, 'test')
    assert selected['原region索引'].tolist() == [0]
    with pytest.raises(ValueError, match='基因组版本'):
        namespace['load_plot_regions'](real, 2, path, 'mm10')
    table['原region'] = 'r0__duplicate_0000'
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='原始区间'):
        namespace['load_plot_regions'](real, 2, path, 'test')


def test_type_grid_has_methods_as_rows_and_uses_each_saved_type(tmp_path):
    namespace = plotting_namespace()
    assert 'plot_region_by_type' in namespace, '缺少单区域细胞类型分布图'
    namespace['dense_block'] = lambda item, rows, columns: item.X[np.ix_(rows, columns)]
    namespace['read_labels'] = lambda item, rows: item.obs['Subclass'].iloc[rows].to_numpy()
    real = ad.AnnData(np.array([[0.1], [0.2], [0.8], [0.9]]),
        obs=pd.DataFrame({'Subclass': ['A', 'A', 'B', 'B']}, index=list('abcd')),
        var=pd.DataFrame(index=['r0']))
    synth = real.copy()
    synth.X = np.array([[0.15], [1.1], [-0.1], [0.85]])
    selected = pd.DataFrame({'原region索引': [0], '原region': ['r0'], '展示名称': ['Test marker'],
        '单区域展示': [1], '展示细胞类型': ['A|B']})
    captured = []
    namespace['save_figure'] = lambda figure, output, stem: captured.append(figure)
    namespace['plot_region_by_type']({'真实数据': real, 'Test': synth},
        {'真实数据': np.arange(4), 'Test': np.arange(4)}, ['Test'], selected, tmp_path)
    figure = captured[0]
    try:
        assert len(figure.axes) == 2
        table = pd.read_csv(tmp_path / 'A1_type_histograms.csv')
        for level in ['A', 'B']:
            group = table[table['细胞类型'].eq(level) & table['方法'].eq('Test')]
            assert group['有限值数'].iloc[0] == 2
            assert group['越界比例'].iloc[0] == .5
            assert group['概率'].sum() == .5
        assert 'A' in figure.axes[0].get_title() and 'B' in figure.axes[1].get_title()
        assert 'Test' in figure.axes[0].get_ylabel()
    finally:
        plt.close(figure)

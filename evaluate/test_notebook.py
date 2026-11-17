"""通过真实 Jupyter 内核执行统一评估，验证两种真实参照模式及文件关闭。"""

import hashlib
import json
import ast
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pytest
import yaml


def test_server_output_guard_resolves_allowed_root_alias(tmp_path):
    """服务器 /home 数据目录为链接时，合法输出应通过范围检查。"""
    notebook = nbformat.read(Path(__file__).with_name("A1A2A3.ipynb"), as_version=4)
    function = next(node for node in ast.parse(notebook.cells[7].source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_all")
    actual_root = tmp_path / "actual_data"

    class LinkedRoot:
        def __fspath__(self):
            return str(tmp_path / "home_alias")

        def resolve(self):
            return actual_root.resolve()

    namespace = {"Path": lambda value: LinkedRoot(),
                 "resolve_path": lambda value: Path(value).resolve(),
                 "sys": SimpleNamespace(platform="linux")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "evaluation_guard", "exec"), namespace)
    config = {"real_h5ad": str(tmp_path / "missing_real.h5ad"),
              "synthetic_h5ad": {"Test": str(tmp_path / "missing_synthetic.h5ad")},
              "reference_h5ad": None, "region_pairs_csv": None,
              "output_dir": str(actual_root / "allowed_output")}
    with pytest.raises(FileNotFoundError, match="缺少输入文件"):
        namespace["evaluate_all"](config)
    with pytest.raises(ValueError):
        namespace["evaluate_all"]({**config, "output_dir": str(tmp_path / "outside")})


@pytest.mark.parametrize("independent_reference", [True, False])
@pytest.mark.parametrize("read_in_memory", [True, False])
def test_jupyter_end_to_end(tmp_path, independent_reference, read_in_memory):
    root = Path(__file__).resolve().parent.parent
    rng = np.random.default_rng(5)
    natural_count = 1004
    names = [f"region_{i}" for i in range(natural_count)]
    var_names = names + [f"{name}__duplicate_{i:04d}" for i, name in enumerate(names[:1000])]

    def make_real(count, prefix):
        labels = np.repeat(["A", "B", "C"], count // 3)
        values = rng.uniform(0.05, 0.95, (count, natural_count))
        values[labels == "B", :30] *= 0.2
        values[labels == "C", 30:60] *= 0.2
        values[rng.uniform(size=values.shape) < 0.02] = np.nan
        return ad.AnnData(np.column_stack([values, values[:, :1000]]).astype(np.float32),
            obs=pd.DataFrame({"Subclass": labels}, index=[f"{prefix}_{i}" for i in range(count)]),
            var=pd.DataFrame(index=var_names))

    real = make_real(24, "real")
    reference = make_real(36, "train")
    synthetic = real.copy()
    synthetic.obs_names = [f"synthetic_{i}" for i in range(24)]
    corrupted = synthetic.copy()
    corrupted.X[:, natural_count:] = rng.uniform(0, 1, (24, 1000))
    bad_axis = synthetic.copy()
    bad_axis.var_names = bad_axis.var_names[::-1]
    paths = {}
    for name, item in [("real", real), ("train", reference), ("Identity", synthetic),
                       ("ChangedTail", corrupted), ("bad_axis", bad_axis)]:
        paths[name] = tmp_path / f"{name}.h5ad"
        item.write_h5ad(paths[name])
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    marker_path = tmp_path / 'markers.csv'
    pd.DataFrame({'原region': ['region_0', 'region_30'], '展示名称': ['Test 1', 'Test 2'],
        '基因组版本': ['test', 'test'], '文献来源': ['test fixture', 'test fixture'],
        '单区域展示': [1, 0], '展示细胞类型': ['A|B', '']}).to_csv(marker_path, index=False)
    configuration = {"real_h5ad": str(paths["real"]), "read_in_memory": read_in_memory,
        "synthetic_h5ad": {name: str(paths[name]) for name in ("Identity", "ChangedTail")},
        "reference_h5ad": str(paths["train"]) if independent_reference else None,
        "output_dir": str(tmp_path / "results"), "a3_top_regions": 24,
        "pca_components": 6, "spearman_top_regions": 12,
        "a1_regions_csv": str(marker_path), "genome_assembly": "test"}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(configuration, allow_unicode=True), encoding="utf-8")
    notebook = nbformat.read(root / "evaluate/A1A2A3.ipynb", as_version=4)
    notebook.cells[1].source = f"from pathlib import Path\nCONFIG_PATH = Path({str(config_path)!r})"
    # 在整个执行过程中记录文件对象，正常返回后直接检查 finally 的关闭效果。
    notebook.cells[3].source += '''
_original_read_h5ad = ad.read_h5ad
_opened_inputs = []
def _tracked_read_h5ad(*args, **kwargs):
    item = _original_read_h5ad(*args, **kwargs)
    _opened_inputs.append(item)
    return item
ad.read_h5ad = _tracked_read_h5ad
'''
    notebook.cells.append(nbformat.v4.new_code_cell(f'''
assert _opened_inputs and all(not item.file.is_open for item in _opened_inputs)
copy_path = Path(config["output_dir"]) / "region_pairs.csv"
before_copy = copy_path.read_bytes()
before_report = (copy_path.parent / "run_summary.json").read_bytes()
try:
    evaluate_all({{**config, "region_pairs_csv": str(copy_path)}})
except ValueError as error:
    assert "覆盖" in str(error), str(error)
else:
    raise AssertionError("输入输出路径冲突未被拒绝")
assert copy_path.read_bytes() == before_copy
assert (copy_path.parent / "run_summary.json").read_bytes() == before_report
bad_config = {{**config, "synthetic_h5ad": {{"Bad": {str(paths['bad_axis'])!r}}},
              "output_dir": {str(tmp_path / 'failure_results')!r}}}
try:
    evaluate_all(bad_config)
except ValueError as error:
    assert "顺序" in str(error), str(error)
else:
    raise AssertionError("错误区域轴未被拒绝")
assert all(not item.file.is_open for item in _opened_inputs)
ad.read_h5ad = _original_read_h5ad
'''))
    executed = NotebookClient(notebook, timeout=240, kernel_name="simscdnam",
        resources={"metadata": {"path": str(root)}}).execute()
    nbformat.write(executed, tmp_path / "executed.ipynb")
    output = tmp_path / "results"
    summary = pd.read_csv(output / "A1_A2_A3_summary.csv").set_index("方法")
    assert summary["近邻数"].eq(30).all()
    np.testing.assert_allclose(summary["加权 1-WD"], 0, atol=1e-12)
    np.testing.assert_allclose(summary["加权 MSE"], 0, atol=1e-12)
    np.testing.assert_allclose(summary["加权 Pearson"], 1, atol=1e-12)
    assert summary.loc["Identity", "miLISI"] == pytest.approx(summary.loc["ChangedTail", "miLISI"])
    assert summary.loc["Identity", "A2_Pearson均值"] == pytest.approx(1)
    assert abs(summary.loc["ChangedTail", "A2_Pearson均值"]) < 0.15
    assert summary["有效配对数"].tolist() == [1000, 1000]
    assert summary["入选区域数"].tolist() == [24, 24]
    assert summary["PCA维数"].tolist() == [6, 6]
    pc = pd.read_csv(output / "A3_PCA_coordinates.csv")
    pc_columns = [c for c in pc if c.startswith("PC")]
    np.testing.assert_array_equal(pc.loc[pc["数据集"] == "Identity", pc_columns],
                                 pc.loc[pc["数据集"] == "ChangedTail", pc_columns])
    assert pd.read_csv(output / "A3_selected_regions.csv")["原region索引"].max() < natural_count
    assert len(pd.read_csv(output / "A2_duplicate_correlations.csv")) == 3000
    assert len(pd.read_csv(output / "A3_miLISI_cells.csv")) == 96
    stems = ["A1_marginal_regions", "A1_region_by_type", "A2_duplicate_Pearson", "A3_joint_UMAP"]
    assert {path.stem for path in (output / 'figures').iterdir()} == set(stems)
    for stem in stems:
        for extension in ["png", "pdf"]:
            assert (output / "figures" / f"{stem}.{extension}").stat().st_size > 1000
    report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert report["状态"] == "完成"
    assert report["原始区间数"] == natural_count
    assert report["A1_A3排除末尾复制列"]
    assert report["输入文件大小及修改时间未变"]
    assert ("独立训练" in report["参照模式"]) == independent_reference
    failed = json.loads((tmp_path / "failure_results/run_summary.json").read_text(encoding="utf-8"))
    assert failed["状态"] == "失败"
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()} == before

"""验证 PeakVI notebook 适配契约；不运行神经网络训练。"""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NOTEBOOK = ROOT / "notebooks/基线_PeakVI.ipynb"


class PeakVIInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.is_file(), "缺少可执行的 PeakVI notebook")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.namespace = {"__name__": "peakvi_interface_check"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and "baseline-main" not in cell["metadata"].get("tags", []):
                exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), self.namespace)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.filled_path = self.folder / "filled.h5ad"
        self.training = ad.AnnData(
            X=np.array([[0.0, 0.2, 0.9], [0.3, 0.5, 0.0], [0.7, 1.0, 0.4]], dtype=np.float32),
            obs=pd.DataFrame({"Subclass": ["A", "B", "A"]}, index=["train_a", "train_b", "train_c"]),
            var=pd.DataFrame(index=["region_1", "region_2", "region_3"]),
        )
        self.training.write_h5ad(self.filled_path)
        self.target = pd.DataFrame({"Subclass": ["B", "A", "B", "A", "B"]}, index=["t5", "t2", "t9", "t0", "t3"])
        self.prepare = Mock(return_value=(self.filled_path, self.target, {"填补来源": "仅真实训练"}))
        self.namespace["prepare_imputed_training"] = self.prepare
        self.namespace["check_output"] = lambda path: Path(path).resolve()
        self.namespace["require_environment"] = lambda: None
        self.namespace["version"] = lambda name: "1.4.2"
        self.calls = {"setup": [], "models": [], "train": [], "generation": []}
        state = self.calls

        class OfficialPeakVIStub:
            @classmethod
            def setup_anndata(cls, data, **kwargs):
                state["setup"].append((data, kwargs))

            def __init__(self, data, **kwargs):
                state["models"].append((data, kwargs))

            def train(self, **kwargs):
                state["train"].append(kwargs)

            def get_normalized_accessibility(self, **kwargs):
                state["generation"].append(kwargs)
                indices = np.asarray(kwargs["indices"])
                return np.column_stack([indices / 10.0 + 0.25, np.full(len(indices), -0.2), np.full(len(indices), 1.2)])

            def save(self, path, **kwargs):
                state["save"] = (path, kwargs)

        self.scvi = types.SimpleNamespace(model=types.SimpleNamespace(PEAKVI=OfficialPeakVIStub), settings=types.SimpleNamespace())
        self.namespace["scvi"] = self.scvi

    def run_notebook(self, device="cpu"):
        with patch.dict(sys.modules, {"scvi": self.scvi}):
            return self.namespace["run_peakvi"](
                self.folder / "raw_training.h5ad",
                self.folder / "target_metadata_only.h5ad",
                self.folder / "PeakVI",
                seed=17,
                device=device,
            )

    def test_default_official_training_and_only_clipped_posterior_probabilities(self):
        original_reader = ad.read_h5ad
        reads = []

        def read_prepared_only(path, *args, **kwargs):
            reads.append(Path(path))
            self.assertEqual(Path(path), self.filled_path, "适配器不可读取原训练或目标 X")
            return original_reader(path, *args, **kwargs)

        with patch.object(ad, "read_h5ad", side_effect=read_prepared_only):
            output_path = self.run_notebook()
        self.assertEqual(reads, [self.filled_path])
        self.assertEqual(self.calls["setup"][0][1], {})
        self.assertEqual(self.calls["models"][0][1], {})
        np.testing.assert_array_equal(self.calls["models"][0][0].X, self.training.X)
        self.assertEqual(self.calls["train"], [{"accelerator": "cpu", "devices": 1}])
        self.assertEqual(self.scvi.settings.seed, 17)
        self.assertEqual(getattr(self.scvi.settings, "logging_dir", None), self.folder / "PeakVI" / "training_logs")
        generated = original_reader(output_path)
        pd.testing.assert_index_equal(generated.obs_names, self.target.index)
        np.testing.assert_array_equal(generated.obs["Subclass"].astype(str), self.target["Subclass"])
        pd.testing.assert_index_equal(generated.var_names, self.training.var_names)
        donors = pd.read_csv(Path(output_path).parent / "donor_cells.csv")
        indices = donors["训练行索引"].to_numpy()
        np.testing.assert_array_equal(self.training.obs["Subclass"].to_numpy()[indices], self.target["Subclass"])
        np.testing.assert_array_equal(donors["目标细胞"], self.target.index)
        np.testing.assert_allclose(generated.X, np.column_stack([indices / 10.0 + 0.25, np.zeros(len(indices)), np.ones(len(indices))]))
        self.assertEqual(generated.X.dtype, np.float32)
        kwargs = self.calls["generation"][0]
        self.assertEqual(set(kwargs), {"indices", "use_z_mean", "return_numpy"})
        self.assertFalse(kwargs["use_z_mean"])
        self.assertTrue(kwargs["return_numpy"])
        np.testing.assert_array_equal(kwargs["indices"], indices)
        self.prepare.assert_called_once_with(self.folder / "raw_training.h5ad", self.folder / "target_metadata_only.h5ad", self.folder / "imputed_baselines")
        report = json.loads((Path(output_path).parent / "verification.json").read_text(encoding="utf-8"))
        self.assertIn("后验", report["生成方式"])
        self.assertEqual(report["输出形状"], [5, 3])

    def test_repeated_seed_preserves_donor_selection(self):
        first_path = self.run_notebook()
        first = (first_path.parent / "donor_cells.csv").read_bytes()
        self.run_notebook()
        self.assertEqual(first, (first_path.parent / "donor_cells.csv").read_bytes())

    def test_cuda_device_selection_does_not_override_training_defaults(self):
        self.run_notebook("cuda:2")
        self.assertEqual(self.calls["train"], [{"accelerator": "gpu", "devices": [2]}])

    def test_absent_subclass_is_rejected_before_fitting(self):
        self.target.loc["t5", "Subclass"] = "unseen"
        with self.assertRaisesRegex(ValueError, "训练.*类别|类别.*训练"):
            self.run_notebook()
        self.assertEqual(self.calls["models"], [])

    def test_nonfinite_native_output_is_rejected(self):
        def invalid_output(**kwargs):
            values = np.ones((len(kwargs["indices"]), 3))
            values[0, 0] = np.nan
            return values

        self.scvi.model.PEAKVI.get_normalized_accessibility = staticmethod(invalid_output)
        with self.assertRaises(ValueError):
            self.run_notebook()
        self.assertFalse((self.folder / "PeakVI/data.h5ad").exists())


if __name__ == "__main__":
    unittest.main()

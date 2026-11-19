"""PeakVI 的 Python 入口；训练与生成函数定义在实验 Notebook 内。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    import os
    import nbformat
    from nbclient import NotebookClient
    from baseline.common import parse_arguments
    from data.paths import PROJECT
    args = parse_arguments("PeakVI", neural=True, reference=False)
    private_packages = PROJECT / "baseline/PeakVI/.runtime/packages"
    if private_packages.is_dir():
        inherited = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(private_packages), inherited) if part
        )
    os.environ.update(BASELINE_TRAINING=str(args.training), BASELINE_TARGET=str(args.target),
                      BASELINE_OUTPUT=str(args.output), BASELINE_SEED=str(args.seed), BASELINE_DEVICE=args.device)
    notebook = nbformat.read(PROJECT / "notebooks/基线_PeakVI.ipynb", as_version=4)
    client = NotebookClient(notebook, timeout=None, kernel_name="simscdnam", resources={"metadata": {"path": str(PROJECT)}})
    executed = PROJECT / "notebooks/执行记录" / f"PeakVI_{args.dataset.replace('/', '_')}.executed.ipynb"
    executed.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, executed)

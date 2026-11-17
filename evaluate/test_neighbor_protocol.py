"""检查当前代码入口，避免位置参数中的旧邻域设置遗漏。"""

import ast
from pathlib import Path
import re

import nbformat
import yaml


def test_current_code_has_no_legacy_neighborhood_setting():
    root = Path(__file__).resolve().parent.parent
    failures = []
    for directory in ("evaluate", "simscDNAm", "mian", "baseline"):
        for path in (root / directory).rglob("*"):
            if any(part in {".ipynb_checkpoints", "__pycache__", ".git"} for part in path.parts):
                continue
            if path.suffix not in {".py", ".ipynb", ".yaml"}:
                continue
            if path.suffix == ".yaml":
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(config, dict) and "milisi_neighbors" in config:
                    assert config["milisi_neighbors"] == 30, path
                continue
            sources = ([cell.source for cell in nbformat.read(path, as_version=4).cells]
                       if path.suffix == ".ipynb" else [path.read_text(encoding="utf-8")])
            code_sources = ([cell.source for cell in nbformat.read(path, as_version=4).cells
                             if cell.cell_type == "code"]
                            if path.suffix == ".ipynb" else sources)
            for source in sources:
                if re.search(r"(?<!\d)10(?:[- ]NN|[- ]nn|\s*近邻)", source):
                    failures.append(f"{path.relative_to(root)}：旧邻域文字")
            for source in code_sources:
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Call):
                        name = (node.func.id if isinstance(node.func, ast.Name)
                                else node.func.attr if isinstance(node.func, ast.Attribute) else "")
                        if name == "milisi" and len(node.args) >= 5:
                            if isinstance(node.args[4], ast.Constant) and node.args[4].value == 10:
                                failures.append(f"{path.relative_to(root)}：milisi位置参数")
                        for keyword in node.keywords:
                            if keyword.arg == "n_neighbors" and isinstance(keyword.value, ast.Constant):
                                if keyword.value.value == 10:
                                    failures.append(f"{path.relative_to(root)}：n_neighbors关键字")
                    if isinstance(node, ast.FunctionDef):
                        args = node.args.args[-len(node.args.defaults):] if node.args.defaults else []
                        for arg, default in zip(args, node.args.defaults):
                            if (arg.arg == "n_neighbors" or
                                    (node.name == "source_neighborhoods" and arg.arg == "k")):
                                if isinstance(default, ast.Constant) and default.value == 10:
                                    failures.append(f"{path.relative_to(root)}：函数默认值")
    assert not failures, "\n".join(sorted(set(failures)))

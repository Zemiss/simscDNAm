"""评估配置、统一抽样、输入核验和独立实验输出。"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from data.paths import PROJECT, SERVER, check_output, require_environment
from data.evaluation import dense_block, read_labels, region_pairs, validate_real_duplicates, validate_region_axis, sample_rows
from evaluate.plotting import plt

def load_config(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置必须是 YAML 映射")
    required = {"real_h5ad", "synthetic_h5ad", "output_dir"}
    if not required.issubset(config) or not isinstance(config["synthetic_h5ad"], dict) or not config["synthetic_h5ad"]:
        raise ValueError("请填写 real_h5ad、非空 synthetic_h5ad 方法映射和 output_dir")
    defaults = {"reference_h5ad": None, "region_pairs_csv": None, "label_key": "Subclass",
        "duplicate_count": 1000, "seed": 42, "evaluation_cells": None, "a3_top_regions": 50000,
        "pca_components": 50, "milisi_neighbors": 30, "umap_seed": 2022, "spearman_top_regions": 2000,
        "a1_regions_csv": None, "genome_assembly": None, "read_in_memory": False}
    config = {**defaults, **config}
    if not isinstance(config["read_in_memory"], bool):
        raise ValueError("read_in_memory必须为布尔值")
    if config["label_key"] != "Subclass":
        raise ValueError("本流程统一使用真实类型 Subclass，请将真实标签字段改名为 Subclass")
    if config["duplicate_count"] != 1000:
        raise ValueError("本流程固定最后 1000 个区间为复制区间")
    for key in ("a3_top_regions", "pca_components", "milisi_neighbors", "spearman_top_regions"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 2:
            raise ValueError(f"{key} 必须为不小于 2 的整数")
    if config["evaluation_cells"] is not None and (not isinstance(config["evaluation_cells"], int)
            or isinstance(config["evaluation_cells"], bool) or config["evaluation_cells"] < 3):
        raise ValueError("evaluation_cells 必须为 null 或不小于 3 的整数")
    if any(not isinstance(name, str) or not name.strip() or name in {"真实数据", "训练参照", "CVAEKMeansAligned"}
           for name in config["synthetic_h5ad"]):
        raise ValueError("方法名须为非空字符串，且不能使用保留名称或暂不参与的 CVAEKMeansAligned")
    return config


def run_evaluation(config_path, experiment, evaluator):
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    def resolve_path(value):
        path = Path(value).expanduser()
        return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    plt.rcParams.update({"figure.dpi": 110, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    methods = list(config["synthetic_h5ad"])
    paths = {"真实数据": resolve_path(config["real_h5ad"]),
             **{name: resolve_path(value) for name, value in config["synthetic_h5ad"].items()}}
    reference_path = resolve_path(config["reference_h5ad"]) if config["reference_h5ad"] else paths["真实数据"]
    pair_path = resolve_path(config["region_pairs_csv"]) if config["region_pairs_csv"] else None
    region_path = resolve_path(config["a1_regions_csv"]) if config.get("a1_regions_csv") else None
    output = resolve_path(config["output_dir"]) / experiment
    check_output(output)
    all_paths = set(paths.values()) | {reference_path}
    if pair_path:
        all_paths.add(pair_path)
    if experiment == "A1" and region_path:
        all_paths.add(region_path)
    if any(path.is_relative_to(output) for path in all_paths):
        raise ValueError("输出目录不能包含输入文件，以免覆盖输入；请选择独立的评估输出目录")
    for path in all_paths:
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")
    if len(set(paths.values())) != len(paths):
        raise ValueError("真实数据和各方法合成数据必须指向不同文件")
    records = [{"路径": str(path), "字节数": path.stat().st_size, "修改时间纳秒": path.stat().st_mtime_ns}
               for path in sorted(all_paths)]
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    report = {"状态": "运行中", "标签字段": "Subclass", "最后复制区间数": 1000, "方法": methods,
              "参照模式": "独立训练真实参照" if reference_path != paths["真实数据"] else "评估真实数据拟合参照，非独立留出评估",
              "配置": config, "配置路径": str(config_path), "输入文件": records, "Python解释器": sys.executable,
              "依赖版本": {name: importlib.metadata.version(name) for name in
                  ["numpy", "pandas", "scipy", "anndata", "scikit-learn", "umap-learn"]}}
    def write_report():
        (output / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report()
    try:
        with ExitStack() as stack, threadpool_limits(limits=8):
            datasets = {}
            for name, path in paths.items():
                item = ad.read_h5ad(path, backed=None if config.get("read_in_memory", False) else "r")
                stack.callback(item.file.close)
                datasets[name] = item
            real = datasets["真实数据"]
            if reference_path == paths["真实数据"]:
                datasets["训练参照"] = real
            else:
                item = ad.read_h5ad(reference_path, backed=None if config.get("read_in_memory", False) else "r")
                stack.callback(item.file.close)
                datasets["训练参照"] = item
                if len(real.obs_names.intersection(item.obs_names)):
                    raise ValueError("独立训练参照与评估真实细胞 ID 重叠，请提供分离后的数据")
            for name, item in datasets.items():
                validate_region_axis(real, item)
                if not item.obs_names.is_unique:
                    raise ValueError(f"{name} 的细胞 ID 不唯一")
            pairs = region_pairs(real, pair_path=pair_path)
            validate_real_duplicates(real, pairs)
            if datasets["训练参照"] is not real:
                validate_real_duplicates(datasets["训练参照"], pairs)
            natural_count = real.n_vars - 1000
            count = real.n_obs if config["evaluation_cells"] is None else config["evaluation_cells"]
            rows = {name: sample_rows(item, count, config["seed"]) for name, item in datasets.items() if name != "训练参照"}
            rows["训练参照"] = np.arange(datasets["训练参照"].n_obs)
            labels = {name: read_labels(item, rows[name]) for name, item in datasets.items()}
            if experiment == "A3" and 2 * count <= config["milisi_neighbors"]:
                raise ValueError("合并细胞数必须大于 miLISI 近邻数")
            pairs.to_csv(output / "region_pairs.csv", index=False)
            pd.concat([pd.DataFrame({"数据集": name, "原行索引": selected,
                "细胞ID": datasets[name].obs_names[selected], "细胞类型": labels[name]})
                for name, selected in rows.items()], ignore_index=True).to_csv(output / "selected_cells.csv", index=False)
            report.update({"评估细胞数": count, "真实参照细胞数": len(rows["训练参照"]), "原始区间数": natural_count,
                           "总区间数": real.n_vars, "A1_A3排除末尾复制列": True})
            write_report()
            print(f"输入校验通过：原始区间 {natural_count}，复制区间 1000，每种方法评估 {count} 个细胞。", flush=True)

            summary, details = evaluator(datasets, rows, methods, pairs, natural_count, output, config, region_path)
            report.update(details)
            diagnostics = []
            for name in ["真实数据", *methods]:
                for domain, start, stop in [("原始区间", 0, natural_count), ("复制区间", natural_count, real.n_vars)]:
                    valid, outside = 0, 0
                    for left in range(start, stop, 512):
                        block = dense_block(datasets[name], rows[name], slice(left, min(left + 512, stop)))
                        finite = np.isfinite(block)
                        valid += int(finite.sum())
                        outside += int((finite & ((block < 0) | (block > 1))).sum())
                    diagnostics.append({"数据集": name, "区间范围": domain, "有限观测数": valid,
                        "缺失比例": 1 - valid / (len(rows[name]) * (stop - start)),
                        "超出0到1比例": outside / valid if valid else np.nan})
            pd.DataFrame(diagnostics).to_csv(output / "input_diagnostics.csv", index=False)
            for record in records:
                path = Path(record["路径"])
                if path.stat().st_size != record["字节数"] or path.stat().st_mtime_ns != record["修改时间纳秒"]:
                    raise RuntimeError(f"评估期间输入文件发生变化：{path}")
            report.update({"状态": "完成", "输入文件大小及修改时间未变": True,
                           "输出目录": str(output), "输出文件": sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())})
            write_report()
            print(summary.to_string(index=False))
            print(experiment + " 评估完成，结果目录：", output)
            return summary
    except Exception as error:
        report.update({"状态": "失败", "错误": f"{type(error).__name__}: {error}"})
        write_report()
        raise



def evaluation_main(experiment, evaluator, argv=None):
    parser = argparse.ArgumentParser(description=f'{experiment} 独立评估与绘图')
    parser.add_argument('--config', type=Path,
        default=PROJECT / 'evaluate' / ('config.server.yaml' if SERVER else 'config.yaml'),
        help='输入方法、输出目录和评估参数 YAML；相对路径以 YAML 所在目录解析')
    args = parser.parse_args(argv)
    require_environment()
    return run_evaluation(args.config, experiment, evaluator)

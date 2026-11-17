"""A1 评估与绘图命令行入口。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from evaluate.runtime import evaluation_main
from evaluate.A1.metrics import a1_metrics
from evaluate.A1.plotting import load_plot_regions, plot_a1



def evaluate(datasets, rows, methods, pairs, natural_count, output, config, region_path=None):
    real = datasets["真实数据"]
    if region_path is None:
        raise ValueError("请通过 a1_regions_csv 提供有文献依据的展示清单；不再随机选择区域")
    plot_regions = load_plot_regions(real, natural_count, region_path, config.get("genome_assembly"))
    print("开始 A1：按真实类型加权的边缘指标。", flush=True)
    a1_rows, detail_tables = [], []
    for method in methods:
        summary, details = a1_metrics(real, datasets[method], rows["真实数据"], rows[method], natural_count)
        a1_rows.append({"方法": method, **summary})
        details.insert(0, "方法", method)
        detail_tables.append(details)
    a1 = pd.DataFrame(a1_rows)
    details = pd.concat(detail_tables, ignore_index=True)
    a1.to_csv(output / "A1_summary.csv", index=False)
    details.to_csv(output / "A1_type_details.csv", index=False)
    plot_a1(datasets, rows, methods, plot_regions, output)

    summary = a1

    return summary, {}



def main(argv=None):
    return evaluation_main('A1', evaluate, argv)


if __name__ == '__main__':
    main()

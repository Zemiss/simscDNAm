"""A2 评估与绘图命令行入口。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from data.evaluation import dense_block, natural_reference_statistics, select_features, imputed_features
from evaluate.runtime import evaluation_main
from evaluate.A2.metrics import paired_pearson, spearman_vector
from evaluate.A2.plotting import plot_duplicate_pearson



def evaluate(datasets, rows, methods, pairs, natural_count, output, config, region_path=None):
    real = datasets["真实数据"]
    print("开始 A2：读取实际复制列，计算 1000 对 Pearson。", flush=True)
    pair_tables = []
    for name in ["真实数据", *methods]:
        original = dense_block(datasets[name], rows[name], pairs["原region索引"].to_numpy())
        copied = dense_block(datasets[name], rows[name], pairs["复制region索引"].to_numpy())
        frame = pd.concat([pairs, paired_pearson(original, copied)], axis=1)
        frame.insert(0, "方法", name)
        pair_tables.append(frame)
    correlations = pd.concat(pair_tables, ignore_index=True)
    a2 = correlations.groupby("方法", sort=False).agg(配对数=("Pearson", "size"),
        有效配对数=("Pearson", "count"), Pearson均值=("Pearson", "mean"), Pearson中位数=("Pearson", "median")).reset_index()
    a2["无效配对数"] = a2["配对数"] - a2["有效配对数"]
    correlations.to_csv(output / "A2_duplicate_correlations.csv", index=False)
    a2.to_csv(output / "A2_pearson_summary.csv", index=False)
    plot_duplicate_pearson(correlations, a2, output)

    print("仅用真实参照的原始区间拟合中位数、按 Subclass 的 ANOVA。", flush=True)
    fill, scores = natural_reference_statistics(datasets["训练参照"], rows["训练参照"], natural_count)
    selected = select_features(scores, config["spearman_top_regions"])
    pd.DataFrame({"原region索引": selected, "原region": real.var_names[selected], "ANOVA": scores[selected]}).to_csv(
        output / "A2_natural_selected_regions.csv", index=False)
    spearman_rows = []
    for index, name in enumerate(["真实数据", *methods]):
        vector = spearman_vector(imputed_features(datasets[name], rows[name], selected, fill))
        filename = f"A2_Spearman_{index:02d}.npy"
        np.save(output / filename, vector)
        finite = vector[np.isfinite(vector)]
        spearman_rows.append({"方法": name, "入选原始区域数": len(selected), "区域对数": len(vector),
            "有效区域对数": len(finite), "Spearman均值": finite.mean() if len(finite) else np.nan,
            "Spearman中位数": np.median(finite) if len(finite) else np.nan, "向量文件": filename})
    pd.DataFrame(spearman_rows).to_csv(output / "A2_natural_Spearman_summary.csv", index=False)

    summary = a2

    return summary, {}



def main(argv=None):
    return evaluation_main('A2', evaluate, argv)


if __name__ == '__main__':
    main()

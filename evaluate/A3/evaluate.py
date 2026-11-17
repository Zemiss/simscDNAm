"""A3 评估与绘图命令行入口。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from data.evaluation import read_labels, natural_reference_statistics, select_features, imputed_features
from evaluate.runtime import evaluation_main
from evaluate.A3.metrics import milisi
from evaluate.A3.plotting import plot_joint_umap

def run_a3(datasets, rows, methods, fill, scores, output, config):
    from sklearn.decomposition import PCA
    import umap
    selected = select_features(scores, config["a3_top_regions"])
    n_components = min(config["pca_components"], len(rows["训练参照"]) - 1, len(selected))
    if n_components < 2 or len(rows["训练参照"]) < 4:
        raise ValueError("真实参照不足以拟合至少二维的 PCA 和 UMAP")
    pd.DataFrame({"原region索引": selected, "原region": datasets["训练参照"].var_names[selected],
                  "ANOVA": scores[selected], "填补中位数": fill[selected]}).to_csv(output / "A3_selected_regions.csv", index=False)
    np.save(output / "reference_natural_medians.npy", fill)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=config["seed"])
    representations = {"训练参照": pca.fit_transform(imputed_features(
        datasets["训练参照"], rows["训练参照"], selected, fill))}
    names = ["训练参照", "真实数据", *methods]
    labels = {name: read_labels(datasets[name], rows[name]) for name in names}
    for name in names[1:]:
        if datasets[name] is datasets["训练参照"] and np.array_equal(rows[name], rows["训练参照"]):
            representations[name] = representations["训练参照"]
        else:
            representations[name] = pca.transform(imputed_features(datasets[name], rows[name], selected, fill))
    pc_tables = []
    for name in names:
        frame = pd.DataFrame(representations[name], columns=[f"PC{i + 1}" for i in range(n_components)])
        frame.insert(0, "细胞类型", labels[name])
        frame.insert(0, "细胞ID", datasets[name].obs_names[rows[name]])
        frame.insert(0, "数据集", name)
        pc_tables.append(frame)
    pd.concat(pc_tables, ignore_index=True).to_csv(output / "A3_PCA_coordinates.csv", index=False)
    summaries, cell_tables, group_tables = [], [], []
    for method in methods:
        score, cells, groups = milisi(representations["真实数据"], labels["真实数据"],
            representations[method], labels[method], config["milisi_neighbors"])
        cells.insert(0, "细胞ID", np.r_[datasets["真实数据"].obs_names[rows["真实数据"]],
                                      datasets[method].obs_names[rows[method]]])
        cells.insert(0, "方法", method)
        groups.insert(0, "方法", method)
        cell_tables.append(cells)
        group_tables.append(groups)
        summaries.append({"方法": method, "miLISI": score, "近邻数": config["milisi_neighbors"],
                          "入选区域数": len(selected), "PCA维数": n_components})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "A3_miLISI.csv", index=False)
    pd.concat(cell_tables, ignore_index=True).to_csv(output / "A3_miLISI_cells.csv", index=False)
    pd.concat(group_tables, ignore_index=True).to_csv(output / "A3_type_details.csv", index=False)
    print("A3 的 PCA 和 miLISI 已计算，开始绘制 UMAP。", flush=True)
    parameters = dict(n_neighbors=min(config["milisi_neighbors"], len(rows["训练参照"]) - 1),
                      random_state=config["umap_seed"], transform_seed=config["umap_seed"], n_jobs=1)
    joint_tables = []
    for method in methods:
        combined = np.vstack([representations["真实数据"], representations[method]])
        joint_parameters = {**parameters, "n_neighbors": min(config["milisi_neighbors"], len(combined) - 1)}
        xy = umap.UMAP(**joint_parameters).fit_transform(combined)
        source = np.r_[np.repeat("Real", len(rows["真实数据"])), np.repeat("Synthetic", len(rows[method]))]
        joint_tables.append(pd.DataFrame({"方法": method, "来源": np.where(source == "Real", "真实", "合成"),
            "细胞ID": np.r_[datasets["真实数据"].obs_names[rows["真实数据"]], datasets[method].obs_names[rows[method]]],
            "UMAP1": xy[:, 0], "UMAP2": xy[:, 1]}))
    coordinates = pd.concat(joint_tables, ignore_index=True)
    coordinates.to_csv(output / "A3_joint_coordinates.csv", index=False)
    plot_joint_umap(coordinates, summary, methods, output, config["umap_seed"])
    return summary, {"入选区域数": len(selected), "PCA维数": n_components,
                     "解释方差比合计": float(pca.explained_variance_ratio_.sum())}

def evaluate(datasets, rows, methods, pairs, natural_count, output, config, region_path=None):
    real = datasets["真实数据"]
    print("开始 A3：真实训练参照 ANOVA、PCA、30近邻 miLISI 和 UMAP。", flush=True)
    fill, scores = natural_reference_statistics(datasets["训练参照"], rows["训练参照"], natural_count)
    a3, a3_protocol = run_a3(datasets, rows, methods, fill, scores, output, config)

    summary = a3

    return summary, {"A3": a3_protocol}



def main(argv=None):
    return evaluation_main('A3', evaluate, argv)


if __name__ == '__main__':
    main()

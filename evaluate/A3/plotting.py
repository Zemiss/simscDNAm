"""A3 对应实验图。"""
import numpy as np
from evaluate.plotting import plt, plot_name, plot_color, save_figure

def plot_joint_umap(coordinates, summary, methods, output, seed):
    """可直接用本轮正式坐标重绘；图上miLISI来自正式PCA空间得分。"""
    figure, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4.4), squeeze=False)
    figure.subplots_adjust(left=0.035, right=0.995, bottom=0.16, top=0.80, wspace=0.15)
    scores = summary.set_index("方法")["miLISI"]
    for axis, method in zip(axes.ravel(), methods):
        local = coordinates.loc[coordinates["方法"].eq(method)]
        xy = local[["UMAP1", "UMAP2"]].to_numpy()
        source = local["来源"].to_numpy()
        draw = np.random.default_rng(seed).permutation(len(local))
        for name, label, color in [("真实", "Real", "#A6A6A6"), ("合成", "Synthetic", plot_color(method))]:
            keep = draw[source[draw] == name]
            axis.scatter(xy[keep, 0], xy[keep, 1], s=8, alpha=0.65, color=color, linewidths=0, label=label, rasterized=True)
        axis.set(title=f"{plot_name(method)}\nmiLISI = {scores.loc[method]:.4f}", xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
        axis.title.set_color(plot_color(method))
        axis.title.set_fontsize(10)
        axis.legend(frameon=False, fontsize=8, markerscale=2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Real / synthetic mixing (k = {int(summary['近邻数'].iloc[0])})", fontsize=12, weight="bold", y=0.98)
    figure.text(0.035, 0.025, "Each panel has its own joint UMAP in real-reference PCA space. Coordinates are not comparable across panels.\n"
                "miLISI is computed in PCA space, not from these 2D projections.", fontsize=7, color="#555555")
    save_figure(figure, output, "A3_joint_UMAP")

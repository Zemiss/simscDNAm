"""A2 对应实验图。"""
import numpy as np
from evaluate.plotting import plt, plot_name, plot_color, style_axis, save_figure

def draw_correlation_violin(axis, vector, position, color, seed):
    finite = np.asarray(vector)[np.isfinite(vector)]
    if not finite.size:
        axis.text(position, 0, "No valid pairs", rotation=90, ha="center", fontsize=8)
        return
    median = np.median(finite)
    values = np.random.default_rng(seed).choice(finite, 200000, replace=False) if finite.size > 200000 else finite
    if values.size > 1 and np.ptp(values) > 1e-12:
        violin = axis.violinplot(values, positions=[position], widths=0.8, showextrema=False, points=200)
        for body in violin["bodies"]:
            body.set(facecolor=color, edgecolor="none", alpha=0.72)
        axis.hlines(median, position - 0.13, position + 0.13,
                    color="white" if color == "#D95F02" else "#E8793E", linewidth=1)
    else:
        axis.hlines(median, position - 0.25, position + 0.25, color=color, linewidth=1.4)


def plot_duplicate_pearson(correlations, summary, output):
    """小提琴展示有效对分布；柱高直接读取正式汇总，避免重画改变得分。"""
    methods = summary.loc[~summary["方法"].eq("真实数据"), "方法"].tolist()
    table = summary.set_index("方法").loc[methods]
    pair_count = int(summary["配对数"].max())
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.26, top=0.84, wspace=0.12)
    figure.suptitle(f"Dependency preservation after appending {pair_count:,} duplicated regions", fontsize=12, weight="bold", y=0.98)
    for position, method in enumerate(["真实数据", *methods], 1):
        vector = correlations.loc[correlations["方法"].eq(method), "Pearson"].to_numpy()
        draw_correlation_violin(axes[0], vector, position, plot_color(method), 42)
    axes[0].axhline(1, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
    axes[0].set(title="Original vs duplicated regions", ylabel="Pearson correlation", ylim=(-1.05, 1.05),
                xticks=np.arange(1, len(methods) + 2), xticklabels=[plot_name(name) for name in ["真实数据", *methods]])
    means = table["Pearson均值"].to_numpy(dtype=float)
    axes[1].bar(np.arange(len(methods)), means, color=[plot_color(name) for name in methods], width=0.8)
    finite = means[np.isfinite(means)]
    lower = min(-0.05, float(finite.min()) * 1.25) if len(finite) else -0.05
    axes[1].set(title="Mean Pearson across valid region pairs", ylabel="Mean Pearson (higher is better)",
                ylim=(lower, 1.12), xticks=np.arange(len(methods)), xticklabels=[plot_name(name) for name in methods])
    for position, value in enumerate(means):
        label = f"{value:.4f}" if np.isfinite(value) else "NA"
        y = value + (0.02 if value >= 0 else -0.02) if np.isfinite(value) else 0.02
        axes[1].text(position, y, label, ha="center", va="bottom" if not np.isfinite(value) or value >= 0 else "top", fontsize=8)
    for axis in axes:
        style_axis(axis)
        axis.title.set_fontsize(10)
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
        for label in axis.get_xticklabels():
            label.set_ha("right")
    counts = "; ".join(f"{plot_name(name)} {int(table.loc[name, '有效配对数']):,}/{int(table.loc[name, '配对数']):,}" for name in methods)
    figure.text(0.06, 0.015, "Valid pairs: " + counts, fontsize=7, color="#555555")
    save_figure(figure, output, "A2_duplicate_Pearson")

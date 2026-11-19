"""A1 对应实验图。"""
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from data.evaluation import dense_block, read_labels
from evaluate.plotting import plt, plot_name, plot_color, style_axis, save_figure

def probability_histogram(values, edges):
    """按全部有限原值归一化；越界值单独报告，不压入边缘分箱。"""
    finite = np.asarray(values)[np.isfinite(values)]
    probabilities = np.histogram(finite, bins=edges)[0].astype(float)
    outside = np.nan
    if finite.size:
        probabilities /= finite.size
        outside = np.mean((finite < edges[0]) | (finite > edges[-1]))
    return probabilities, outside, int(finite.size)


def load_plot_regions(real, natural_count, path, assembly):
    """显式读取展示证据清单；不把随机区间或复制尾部当作文献区域。"""
    table = pd.read_csv(path)
    required = {"原region", "展示名称", "基因组版本", "文献来源", "单区域展示", "展示细胞类型"}
    if not required.issubset(table) or table.empty:
        raise ValueError("展示区间清单缺少区域、名称、基因组版本、文献来源或单区域展示信息")
    if table[["原region", "展示名称", "基因组版本", "文献来源"]].isna().any().any():
        raise ValueError("展示区间及文献证据不能为空")
    if not assembly or not table["基因组版本"].eq(assembly).all():
        raise ValueError("展示区间清单的基因组版本与配置不一致")
    indices = real.var_names[:natural_count].get_indexer(table["原region"])
    if np.any(indices < 0) or table["原region"].duplicated().any():
        raise ValueError("展示区间须为互不重复且存在的原始区间，不能使用复制尾部")
    flags = pd.to_numeric(table["单区域展示"], errors="coerce")
    if not flags.isin([0, 1]).all() or flags.sum() != 1:
        raise ValueError("清单须且仅须指定一个单区域展示窗口")
    detail = table.loc[flags.eq(1)].iloc[0]
    levels = str(detail["展示细胞类型"]).split("|")
    if pd.isna(detail["展示细胞类型"]) or any(not level.strip() for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("单区域展示须提供以 | 分隔的不重复真实细胞类型")
    table["单区域展示"] = flags.astype(int)
    table.insert(0, "原region索引", indices)
    return table


def marginal_legend(figure, methods, y=0.99):
    handles = [Patch(facecolor=to_rgba(plot_color(name), 0.4), edgecolor=plot_color(name), label=plot_name(name))
               for name in ["真实数据", *methods]]
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8,
                  bbox_to_anchor=(0.5, y))


def draw_marginal(axis, real_histogram, synthetic_histogram, method, edges):
    style_axis(axis)
    for name, histogram, edge_color in [("真实数据", real_histogram, "#555555"),
                                       (method, synthetic_histogram, plot_color(method))]:
        axis.stairs(histogram[0], edges, fill=True, linewidth=0.8,
                    edgecolor=edge_color, facecolor=to_rgba(plot_color(name), 0.35))
    outside = synthetic_histogram[1]
    annotation = f"Out of [0, 1]: {outside:.1%}" if np.isfinite(outside) else "Out of [0, 1]: n/a"
    axis.text(0.98, 0.95, annotation, transform=axis.transAxes, fontsize=6.5, va="top", ha="right")
    axis.set(xlim=(0, 1), xticks=np.linspace(0, 1, 6))
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.tick_params(labelsize=7, length=3)


def plot_marginal_regions(datasets, rows, methods, regions, output):
    selected = regions["原region索引"].to_numpy(dtype=int)
    regions.to_csv(output / "A1_plot_regions.csv", index=False)
    values = {name: dense_block(datasets[name], rows[name], selected) for name in ["真实数据", *methods]}
    edges = np.linspace(0, 1, 31)
    figure, axes = plt.subplots(len(selected), len(methods), figsize=(3.15 * len(methods), 2.0 * len(selected) + 0.7),
                                sharex=True, sharey="row", squeeze=False)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.85 / figure.get_figheight(), top=1 - 0.95 / figure.get_figheight(), wspace=0.10, hspace=0.34)
    source_tables = []
    for i, row in regions.iterrows():
        histograms = {name: probability_histogram(values[name][:, i], edges) for name in ["真实数据", *methods]}
        upper = max(0.1, np.ceil(max(item[0].max() for item in histograms.values()) * 12) / 10)
        for name, (probabilities, outside, count) in histograms.items():
            source_tables.append(pd.DataFrame({"原region": row["原region"], "方法": name,
                "分箱左界": edges[:-1], "分箱右界": edges[1:], "概率": probabilities,
                "有限值数": count, "越界比例": outside}))
        for j, method in enumerate(methods):
            axis = axes[i, j]
            draw_marginal(axis, histograms["真实数据"], histograms[method], method, edges)
            axis.text(0.02, 0.95, str(row["原region"]), transform=axis.transAxes, fontsize=6, va="top", color="#555555")
            axis.set_ylim(0, upper)
            if i == 0:
                axis.set_title(plot_name(method), color=plot_color(method), fontsize=10, weight="bold", pad=7)
            if i == len(selected) - 1:
                axis.set_xlabel("Methylation level", fontsize=8)
            if j == 0:
                axis.set_ylabel(f"{row['展示名称']}\nProbability per bin", fontsize=8)
    marginal_legend(figure, methods, 0.98)
    figure.suptitle("Literature-guided Tle4 locus windows (mm10)" if regions["展示名称"].str.startswith("Tle4").all()
                     else "Literature-guided marginal distributions", fontsize=11, weight="bold", y=1.01)
    figure.text(0.075, 0.015, "All cell types pooled. Shared 30 bins; finite values only; out-of-range mass is not clipped.\n"
                  "Windows are within the published example locus.",
                fontsize=7, color="#555555")
    pd.concat(source_tables, ignore_index=True).to_csv(output / "A1_plot_histograms.csv", index=False)
    save_figure(figure, output, "A1_marginal_regions")


def plot_region_by_type(datasets, rows, methods, regions, output):
    detail = regions.loc[regions["单区域展示"].eq(1)].iloc[0]
    levels = detail["展示细胞类型"].split("|")
    names = ["真实数据", *methods]
    labels = {name: read_labels(datasets[name], rows[name]) for name in names}
    values = {name: dense_block(datasets[name], rows[name], [int(detail["原region索引"])]).ravel() for name in names}
    for name in names:
        if any(not np.any(labels[name] == level) for level in levels):
            raise ValueError(f"{name} 缺少清单指定的展示细胞类型，不静默混合或替换标签")
    edges = np.linspace(0, 1, 31)
    histograms = {(name, level): probability_histogram(values[name][labels[name] == level], edges)
                  for name in names for level in levels}
    upper = max(0.1, np.ceil(max(item[0].max() for item in histograms.values()) * 12) / 10)
    figure, axes = plt.subplots(len(methods), len(levels), figsize=(3.0 * len(levels), 1.9 * len(methods) + 1),
                                sharex=True, sharey=True, squeeze=False)
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.85 / figure.get_figheight(), top=1 - 1.35 / figure.get_figheight(), wspace=0.12, hspace=0.24)
    for i, method in enumerate(methods):
        for j, level in enumerate(levels):
            axis = axes[i, j]
            real_histogram, synthetic_histogram = histograms["真实数据", level], histograms[method, level]
            draw_marginal(axis, real_histogram, synthetic_histogram, method, edges)
            axis.text(0.02, 0.95, f"n: R={real_histogram[2]}, S={synthetic_histogram[2]}",
                      transform=axis.transAxes, fontsize=6.5, va="top", color="#555555")
            axis.set_ylim(0, upper)
            if i == 0:
                axis.set_title(level.replace(" CTX Glut", "\nCTX Glut"), fontsize=9, pad=8)
            if j == 0:
                axis.set_ylabel(f"{plot_name(method)}\nProbability per bin", color=plot_color(method), fontsize=8)
            if i == len(methods) - 1:
                axis.set_xlabel("Methylation level", fontsize=8)
    marginal_legend(figure, methods, 1 - 0.48 / figure.get_figheight())
    figure.suptitle(f"{detail['展示名称']} | {detail['原region']}\nSingle-region distributions by Subclass",
                   fontsize=11, weight="bold", y=1.01)
    figure.text(0.09, 0.015, "Rows: methods. Columns: saved cell types. R = evaluation real; S = synthetic; n counts finite observations.\n"
                "Shared 30 bins and y-axis; out-of-range values are reported separately.", fontsize=7, color="#555555")
    source_tables = [pd.DataFrame({"原region": detail["原region"], "细胞类型": level, "方法": name,
        "分箱左界": edges[:-1], "分箱右界": edges[1:], "概率": probability,
        "有限值数": count, "越界比例": outside})
        for (name, level), (probability, outside, count) in histograms.items()]
    pd.concat(source_tables, ignore_index=True).to_csv(output / "A1_type_histograms.csv", index=False)
    save_figure(figure, output, "A1_region_by_type")


def plot_a1(datasets, rows, methods, regions, output):
    plot_marginal_regions(datasets, rows, methods, regions, output)
    plot_region_by_type(datasets, rows, methods, regions, output)

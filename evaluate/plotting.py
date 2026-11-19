"""评估图的共用配色、样式和 PNG/PDF 导出。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PLOT_COLORS = {"真实数据": "#A6A6A6", "Real": "#A6A6A6",
    "simscDNAm": "#3B73B9", "CVAE_training_mask": "#3B73B9", "FittedMarginal": "#269D91",
    "simATAC": "#A487C9", "simCAS": "#D95F02", "PeakVI": "#CC79A7",
    "DiTSim": "#C49A35"}



def plot_name(name):
    return {"真实数据": "Real", "CVAE_training_mask": "CVAE (training mask)"}.get(name, name)


def plot_color(name):
    return PLOT_COLORS.get(name, "#3B73B9")


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(0.7)
    axis.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    axis.set_axisbelow(True)


def save_figure(figure, output, stem):
    with plt.rc_context({"pdf.fonttype": 42}):
        for extension in ("png", "pdf"):
            figure.savefig(output / "figures" / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)

"""Shared look for every paper figure. Change colours and fonts here, not in the figure scripts."""
import matplotlib as mpl

EVENT_COLOR = {"T": "#C0392B", "M": "#2471A3"}            # as in Paper_Plots-2
EVENT_LABEL = {"T": "T$_{2m}$ extremes", "M": "h$_{2m}$ (wet-bulb) extremes"}
DELTA_LABEL = {"delta_FT": r"$\delta_{a} = h_{ML} - h^*_{500}$",
               "delta_S": r"$\delta_{S} = h_{2m} - h_{ML}$",
               "delta_total": r"$\delta = h_{2m} - h^*_{500}$"}
DELTA_SHORT = {"delta_FT": r"$\delta_{a}$", "delta_S": r"$\delta_S$", "delta_total": r"$\delta$"}
SUBBAND_LABEL = {"all": "", "TP": "tropics", "ST": "subtropics", "ML": "midlatitudes"}
PANEL_BG = "#F8F9FA"                                        # pale axes background, as in Paper_Plots-2
SPINE = "#444444"
WINDOW_COLOR = "#0F766E"                                    # the 400–600 hPa pass band
# densities: from the panel background through peach, coral and magenta to deep indigo, so empty
# bins blend into the background instead of showing as white gaps
DENSITY_COLORS = ["#F8F9FA", "#FDE7C4", "#FCB57A", "#F0705B", "#C83A6E", "#7A1F7E", "#2D0B59"]


def _density_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("paper_density", DENSITY_COLORS, N=256)


DENSITY_CMAP = _density_cmap()


def paper_axes(ax, minor=True, grid=False):
    """The Paper_Plots-2 look: pale background, dark left/bottom spines, small outward ticks."""
    from matplotlib.ticker import AutoMinorLocator
    ax.set_facecolor(PANEL_BG)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.5, width=0.8, color=SPINE)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", length=2, width=0.6, color=SPINE)
    if grid:
        ax.grid(color="white", lw=0.9, zorder=0)
    return ax


def panel_letter(ax, k, x=0.02, y=0.98, **kw):
    ax.text(x, y, f"({'abcdefghij'[k]})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left", zorder=20, **kw)


def apply():
    mpl.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def band_text(cfg, weighted=None):
    lo, hi = cfg["lat_band"]
    f = lambda v: f"{abs(v):g}°{'S' if v < 0 else 'N'}"
    w = "" if weighted is None else (" · cos(lat) weighted" if weighted else " · unweighted")
    return f"{f(lo)}–{f(hi)}, land{w}"

"""Shared look for every paper figure. Change colours and fonts here, not in the figure scripts.

Decisions and the colorblind check are in figures/STYLE.md (2026-11-09)."""
import logging

import matplotlib as mpl
import numpy as np

EVENT_COLOR = {"T": "#B2182B", "M": "#2166AC"}            # brick / navy
EVENT_LABEL = {"T": r"$T_{2m}$ extremes", "M": r"$WBT_{2m}$ extremes"}
EVENT_SHORT = {"T": r"$T_{2m}$", "M": r"$WBT_{2m}$"}
KJ = "[kJ/kg]"                                              # units: square brackets with a slash
RH_LABEL = "surface RH [%]"
DELTA_LABEL = {"delta_FT": r"$\delta_{a} = h_{ML} - h^*_{500}$",
               "delta_S": r"$\delta_{S} = h_{2m} - h_{ML}$",
               "delta_total": r"$\delta = h_{2m} - h^*_{500}$"}
DELTA_SHORT = {"delta_FT": r"$\delta_{a}$", "delta_S": r"$\delta_S$", "delta_total": r"$\delta$"}
SUBBAND_LABEL = {"all": "", "TP": "tropics", "ST": "subtropics", "ML": "midlatitudes"}
PANEL_BG = "white"                                          # mostly white panels, no grid
SPINE = "#444444"
BAND_COLOR, BAND_ALPHA = "#D8C9A3", 0.35                    # the 400–600 hPa band: pale sand tint
WINDOW_COLOR = "#8C7346"                                    # darker sand for band edges and labels
# Entrainment rates shown in the plume figures (case study, pooled tops, schematic): light → dark
EPS_SHOWN = (0.2, 0.3, 0.4)                                 # km⁻¹
EPS_COLOR = {0.0: "#9CA3AF", 0.2: "#E0A040", 0.3: "#3A8FB7", 0.4: "#5B2A86"}   # ochre / blue-green / plum
BRACKET_COLOR = {"weak": "#A8A29E", "strong": "#57534E"}    # ε outside the plausible family, dashed
BRACKET_LS = (0, (5, 2))
DIVERGING_CMAP = "coolwarm"                                 # maps of δ_a and δ_S
REGIME_COLOR = ["#7C3AED", "#D97706", "#0F766E"]            # cascade regimes (b, c, d)


def _cmaps():
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap
    density = colormaps["Purples"].copy()
    density.set_bad("white")
    rh = LinearSegmentedColormap.from_list("rh_sand_teal", ["#E9D8A6", "#94D2BD", "#0A9396", "#005F73"], N=256)
    return density, rh


DENSITY_CMAP, RH_CMAP = _cmaps()                            # event densities; profiles binned by RH


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


def band(ax, lo=400.0, hi=600.0, edges=False, label=None, axis="y"):
    """The 400–600 hPa band: pale sand fill, optional dashed edges and a small label at the right."""
    span = ax.axhspan if axis == "y" else ax.axvspan
    span(hi, lo, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0, zorder=0)
    if edges:
        line = ax.axhline if axis == "y" else ax.axvline
        for v in (lo, hi):
            line(v, color=WINDOW_COLOR, lw=0.9, ls=(0, (5, 3)), zorder=1)
    if label and axis == "y":
        ax.text(0.98, lo + 0.15 * (hi - lo), label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7, color=WINDOW_COLOR)


def rh_percent(ax, axis="x", label=True):
    """Show an RH axis stored as a fraction (0–1) in percent."""
    from matplotlib.ticker import FuncFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    if label:
        (ax.set_xlabel if axis == "x" else ax.set_ylabel)(RH_LABEL)


def panel_letter(ax, k, x=0.02, y=0.98, **kw):
    ax.text(x, y, f"({'abcdefghij'[k]})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left", zorder=20, **kw)


def apply():
    # AMS lists Arial and Helvetica as standard figure fonts; Liberation Sans is the Arial-metric font on
    # Linux (Casper). Missing fonts fall through quietly.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.facecolor": PANEL_BG, "axes.grid": False,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def band_text(cfg, weighted=None):
    lo, hi = cfg["lat_band"]
    f = lambda v: f"{abs(v):g}°{'S' if v < 0 else 'N'}"
    w = "" if weighted is None else (" · cos(lat) weighted" if weighted else " · unweighted")
    return f"{f(lo)}–{f(hi)}, land{w}"

"""Shared look for every paper figure. Change colours and fonts here, not in the figure scripts."""
import matplotlib as mpl

EVENT_COLOR = {"T": "#C0392B", "M": "#2471A3"}            # as in Paper_Plots-2
EVENT_LABEL = {"T": "T$_{2m}$ extremes", "M": "h$_{2m}$ (wet-bulb) extremes"}
DELTA_LABEL = {"delta_FT": r"$\delta_{FT} = h_{LPL} - h^*_{500}$",
               "delta_S": r"$\delta_{S} = h_{2m} - h_{LPL}$",
               "delta_total": r"$\delta_{total} = h_{2m} - h^*_{500}$"}
DELTA_SHORT = {"delta_FT": r"$\delta_{FT}$", "delta_S": r"$\delta_S$", "delta_total": r"$\delta_{total}$"}
SUBBAND_LABEL = {"all": "", "TP": "tropics", "ST": "subtropics", "ML": "midlatitudes"}


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

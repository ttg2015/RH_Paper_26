# Figure style guide (decided 2026-11-09)

One reference for every paper figure. `style.py` should hold these as constants. The option
sheets that led to them are in `draft-merged/storyline/style_options/`, including the colorblind check.

## Colors

| Use | Choice | Values |
|---|---|---|
| Event types | brick / navy | temperature `#B2182B`, wet-bulb `#2166AC` |
| 2D event densities (Figs 5, 9, 11a, 12) | Purples | matplotlib `Purples`, zeros masked |
| Profiles binned by surface RH (Figs 10, 11) | sand → teal | `#E9D8A6`, `#94D2BD`, `#0A9396`, `#005F73` (linear) |
| Entrainment rates ε = 0.2 / 0.3 / 0.4 | ochre / blue-green / plum, light → dark | `#E0A040`, `#3A8FB7`, `#5B2A86` |
| Undilute plume (ε = 0) | gray, dotted | `#9CA3AF` |
| Bracketing ε (too weak / too strong) | light / dark gray, dashed | `#A8A29E`, `#57534E` |
| 400–600 hPa band | pale sand tint | `#D8C9A3`, low alpha (about 0.25) |
| Diverging maps (Fig 4) | coolwarm | matplotlib `coolwarm`, symmetric limits |
| Regime boxes (Fig 11) | violet / orange / teal | `#7C3AED`, `#D97706`, `#0F766E` |

Colorblind check: under simulated deuteranopia, protanopia and tritanopia, the event pair, the
ε trio and the regime boxes all keep a color difference ΔE of at least about 28. The sequential maps
change lightness steadily, so their order survives. The first ε choice (sea-green 0.3) turned gray
under red-green deficiency and collided with the bracket grays, so it was replaced.

## Type and layout

- **Font:** Helvetica, then Arial, then Liberation Sans (the Arial-metric font on Linux, e.g. Casper).
  AMS lists Arial and Helvetica as standard figure fonts and asks for one font across all figures.
  Avoid glyphs Helvetica lacks (e.g. →); use mathtext or words.
- **Units:** square brackets with a slash, e.g. `MSE [kJ/kg]`, `δ_a [kJ/kg]`, `Pressure [hPa]`.
- **RH:** in percent on axes and colorbars (`surface RH [%]`). Note: the text gives RH0 as a fraction
  (0.316); change it to percent (31.6 %) or keep the two consistent.
- **Event names:** `T₂ₘ` and `WBT₂ₘ` (mathtext `T_{2m}`, `WBT_{2m}`) for temperature and wet-bulb
  extremes, where the event type matters. Pooled figures need no event label.
- **Titles:** short panel titles of a few words (e.g. "middle latitudes"); no suptitles, run notes or
  year/band/weighting strings in the figure. Captions carry those.
- **Panel letters:** bold `(a)`, inside the axes at the top left.
- **Panels:** mostly white with no grid. Use a pale background or a grid only where a figure needs it
  (decide per figure).
- **Widths:** AMS single column 19 pc, double 39 pc. Keep text readable at print size; large text makes
  AMS shrink the figure.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgba

# ── Config ─────────────────────────────────────────────────────────────────────

ACCENT   = "#2C3E6B"   # dark navy — header background
ALT_ROW  = "#F2F5FA"   # light blue-grey — alternating row shading
FONT     = "DejaVu Sans"

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv("pad_merged.csv")
df.columns = df.columns.str.strip()

# ── Column groups ──────────────────────────────────────────────────────────────

continuous_vars = {
    "mtnest":               "Mountainous terrain (%)",
    "lmtnest":              "Mountainous terrain (log)",
    "gdp_per_capita":       "GDP per capita (USD)",
    "gdp_growth":           "GDP growth (%)",
    "milper":               "Military personnel (000s)",
    "v2x_polyarchy":        "Electoral democracy index",
    "v2x_liberal":          "Liberal democracy index",
    "v2x_partip":           "Participatory democracy index",
    "v2xlg_legcon":         "Legislative constraints index",
    "v2x_jucon":            "Judicial constraints index",
    "v2x_freexp_altinf":    "Freedom of expression index",
    "v2mecenefm":           "Media censorship index",
    "v2regdur":             "Regime duration (years)",
    "n_neighbors_at_war":   "No. of neighbors at war",
    "pct_neighbors_at_war": "% neighbors at war",
    "bd_best_total":        "Battle deaths (best estimate)",
    "bd_low_total":         "Battle deaths (low estimate)",
    "bd_high_total":        "Battle deaths (high estimate)",
}

binary_vars = {
    "cease":              "Ceasefire",
    "intarmy":            "Integration of armed forces",
    "ddr":                "DDR provision",
    "withd":              "Withdrawal provision",
    "mil_prov":           "Military provisions (any)",
    "pp":                 "Power-sharing (political)",
    "intgov":             "Interim government",
    "intciv":             "Civil society integration",
    "elections":          "Elections provision",
    "interim":            "Interim arrangements",
    "natalks":            "National dialogue",
    "shagov":             "Government power-sharing",
    "pol_prov":           "Political provisions (any)",
    "aut":                "Autonomy provision",
    "fed":                "Federalism provision",
    "ind":                "Independence provision",
    "shaloc":             "Local power-sharing",
    "regdev":             "Regional development",
    "cul":                "Cultural provisions",
    "demarcation":        "Demarcation provision",
    "locgov":             "Local governance",
    "terr_prov":          "Territorial provisions (any)",
    "amn":                "Amnesty provision",
    "pris":               "Prisoner release",
    "recon":              "Reconciliation provision",
    "justice_prov":       "Justice provisions (any)",
    "reaffirm":           "Reaffirmation of prior agreements",
    "outlin":             "Outline/framework agreement",
    "pko":                "PKO provision",
    "gender":             "Gender provisions",
    "co_impl":            "Joint implementation body",
    "inclusive":          "Inclusive process",
    "ended":              "Agreement collapsed",
    "neighbor_at_war":    "Neighbor at war (binary)",
    "regime_change":      "Regime change",
    "government_victory": "Government victory",
    "rebel_victory":      "Rebel victory",
}

# ── Build dataframes ───────────────────────────────────────────────────────────

cont_rows = []
for col, label in continuous_vars.items():
    if col not in df.columns:
        continue
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    cont_rows.append({
        "Variable": label,
        "N":        f"{len(s):,}",
        "Mean":     f"{s.mean():.3f}",
        "SD":       f"{s.std():.3f}",
        "Min":      f"{s.min():.3f}",
        "Median":   f"{s.median():.3f}",
        "Max":      f"{s.max():.3f}",
        "Missing":  str(df[col].isna().sum()),
    })
cont_table = pd.DataFrame(cont_rows)

bin_rows = []
for col, label in binary_vars.items():
    if col not in df.columns:
        continue
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    bin_rows.append({
        "Variable":   label,
        "N":          f"{len(s):,}",
        "N (= 1)":    f"{int(s.sum()):,}",
        "Proportion": f"{s.mean():.3f}",
        "Missing":    str(df[col].isna().sum()),
    })
bin_table = pd.DataFrame(bin_rows)

# ── Render function ────────────────────────────────────────────────────────────

def render_table(df_in, filename, note=None):
    cols   = list(df_in.columns)
    n_rows = len(df_in)
    n_cols = len(cols)

    row_h  = 0.38
    head_h = 0.55
    note_h = 0.35 if note else 0
    fig_h  = head_h + row_h * n_rows + note_h + 0.2

    col_widths = [3.2] + [0.95] * (n_cols - 1)
    fig_w = sum(col_widths) + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    margin_l = 0.02
    margin_r = 0.02
    table_top = 1.0  # no title bar — table starts at the very top

    total_w = sum(col_widths)
    cum_w   = 0
    x_positions = []
    for w in col_widths:
        x_positions.append(margin_l + (cum_w / total_w) * (1 - margin_l - margin_r))
        cum_w += w
    x_positions.append(1 - margin_r)

    cell_h       = row_h / fig_h
    header_h_frac = head_h / fig_h
    data_top     = table_top - header_h_frac

    # ── Header ─────────────────────────────────────────────────────────────────
    for ci, col in enumerate(cols):
        x0 = x_positions[ci]
        x1 = x_positions[ci + 1]
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (x0, data_top), x1 - x0, header_h_frac,
            transform=fig.transFigure, figure=fig,
            boxstyle="square,pad=0",
            facecolor=ACCENT, edgecolor="white", linewidth=0.5,
            clip_on=False,
        ))
        ha    = "left" if ci == 0 else "center"
        xtext = x0 + 0.008 if ci == 0 else (x0 + x1) / 2
        fig.text(
            xtext, data_top + header_h_frac / 2,
            col, ha=ha, va="center",
            fontsize=8.5, fontweight="bold", color="white",
            fontfamily=FONT,
        )

    # ── Data rows ──────────────────────────────────────────────────────────────
    for ri, row in enumerate(df_in.itertuples(index=False)):
        y_top = data_top - ri * cell_h
        bg    = ALT_ROW if ri % 2 == 0 else "white"
        for ci, val in enumerate(row):
            x0 = x_positions[ci]
            x1 = x_positions[ci + 1]
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (x0, y_top - cell_h), x1 - x0, cell_h,
                transform=fig.transFigure, figure=fig,
                boxstyle="square,pad=0",
                facecolor=bg, edgecolor="#D0D7E5", linewidth=0.4,
                clip_on=False,
            ))
            ha    = "left" if ci == 0 else "center"
            xtext = x0 + 0.008 if ci == 0 else (x0 + x1) / 2
            fig.text(
                xtext, y_top - cell_h / 2,
                str(val), ha=ha, va="center",
                fontsize=8, color="#1A1A2E",
                fontfamily=FONT,
            )

    # ── Bottom rule ────────────────────────────────────────────────────────────
    bottom_y = data_top - n_rows * cell_h
    fig.add_artist(mpl.lines.Line2D(
        [margin_l, 1 - margin_r], [bottom_y, bottom_y],
        transform=fig.transFigure, color=ACCENT, linewidth=1.2,
    ))

    # ── Note ───────────────────────────────────────────────────────────────────
    if note:
        fig.text(
            margin_l, bottom_y - 0.01,
            note, ha="left", va="top",
            fontsize=7.5, color="#555555", style="italic",
            fontfamily=FONT,
        )

    plt.savefig(filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {filename}")


# ── Split binary table in half ─────────────────────────────────────────────────

mid = len(bin_table) // 2
bin_table_a = bin_table.iloc[:mid].reset_index(drop=True)
bin_table_b = bin_table.iloc[mid:].reset_index(drop=True)

# ── Export ─────────────────────────────────────────────────────────────────────

CONT_NOTE = "Note: Statistics computed on non-missing observations. 'Missing' reports the raw count of missing values before listwise deletion."
BIN_NOTE  = "Note: Proportion = share of non-missing observations coded 1."

render_table(cont_table,  filename="desc_stats_continuous.png",  note=CONT_NOTE)
render_table(bin_table_a, filename="desc_stats_binary_1.png",    note=None)
render_table(bin_table_b, filename="desc_stats_binary_2.png",    note=BIN_NOTE)
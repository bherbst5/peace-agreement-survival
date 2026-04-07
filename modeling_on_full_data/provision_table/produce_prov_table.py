import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgba

# ── Config ─────────────────────────────────────────────────────────────────────

ACCENT   = "#2C3E6B"   # dark navy — header background
ALT_ROW  = "#F2F5FA"   # light blue-grey — alternating row shading
FONT     = "DejaVu Sans"

# ── Data ───────────────────────────────────────────────────────────────────────

data = [
    {
        "Domain": "Military", 
        "Specific Provisions Included": "Ceasefire\nIntegration of rebels into armed forces\nDisarmament (DDR)\nWithdrawal of foreign forces"
    },
    {
        "Domain": "Political", 
        "Specific Provisions Included": "Government power-sharing\nIntegration of rebels into government\nLocal government power-sharing\nElections or electoral reforms\nIntegration of rebels into interim government\nNational talks"
    },
    {
        "Domain": "Territorial", 
        "Specific Provisions Included": "Autonomy granted to disputed region\nFederal state solution\nIndependence granted to disputed region\nReferendum on future status of disputed region\nDisputed region granted\nBorder demarcation\nLocal governance granted to disputed region"
    },
    {
        "Domain": "Justice", 
        "Specific Provisions Included": "Amnesty\nRelease of prisoners\nNational reconciliation\nReturn of refugees\nExtension of cultural freedoms"
    },
    {
        "Domain": "Miscellaneous",
        "Specific Provisions Included": "Deployment of peacekeeping operation\nReaffirmation of previous agreement\nOutline of negotiating agenda\nInclusion of women or gender\nOversight committee"
    }
]

df = pd.DataFrame(data)

# ── Render function ────────────────────────────────────────────────────────────

def render_multi_line_table(df_in, filename, note=None):
    cols   = list(df_in.columns)
    n_rows = len(df_in)
    n_cols = len(cols)

    # Calculate dynamic row heights based on line breaks
    row_heights_inches = []
    for row in df_in.itertuples(index=False):
        max_lines = max(str(val).count('\n') + 1 for val in row)
        row_heights_inches.append(max_lines * 0.25 + 0.15) 

    head_h = 0.55
    note_h = 0.35 if note else 0
    fig_h  = head_h + sum(row_heights_inches) + note_h + 0.4 # Added slight bottom buffer

    # Column widths (3/5 of original 5.6 is 3.36)
    col_widths = [2.2, 3.36] 
    table_width_inches = sum(col_widths)
    
    # Define a figure width slightly wider than the table to allow for centering
    fig_w = table_width_inches + 1.0 

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # --- CENTERING LOGIC ---
    # Calculate how much space is left over to split into margins
    # We work in "figure fraction" (0 to 1)
    table_w_frac = table_width_inches / fig_w
    centering_offset = (1.0 - table_w_frac) / 2

    x_positions = []
    cum_w = 0
    for w in col_widths:
        # Map table coordinates into the centered figure space
        x_positions.append(centering_offset + (cum_w / table_width_inches) * table_w_frac)
        cum_w += w
    x_positions.append(centering_offset + table_w_frac)

    table_top = 0.95 # Slight top margin
    header_h_frac = head_h / fig_h
    data_top      = table_top - header_h_frac

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
        
        fig.text(
            x0 + 0.015, data_top + header_h_frac / 2,
            col, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color="white",
            fontfamily=FONT,
        )

    # ── Data rows ──────────────────────────────────────────────────────────────
    current_y = data_top
    for ri, row in enumerate(df_in.itertuples(index=False)):
        cell_h_frac = row_heights_inches[ri] / fig_h
        bg    = ALT_ROW if ri % 2 == 0 else "white"
        
        for ci, val in enumerate(row):
            x0 = x_positions[ci]
            x1 = x_positions[ci + 1]
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (x0, current_y - cell_h_frac), x1 - x0, cell_h_frac,
                transform=fig.transFigure, figure=fig,
                boxstyle="square,pad=0",
                facecolor=bg, edgecolor="#D0D7E5", linewidth=0.4,
                clip_on=False,
            ))
            
            weight = "bold" if ci == 0 else "normal"
            
            fig.text(
                x0 + 0.015, current_y - cell_h_frac / 2,
                str(val), ha="left", va="center",
                fontsize=8.5, color="#1A1A2E",
                fontfamily=FONT, fontweight=weight, linespacing=1.6
            )
            
        current_y -= cell_h_frac

    # ── Bottom rule ────────────────────────────────────────────────────────────
    fig.add_artist(mpl.lines.Line2D(
        [x_positions[0], x_positions[-1]], [current_y, current_y],
        transform=fig.transFigure, color=ACCENT, linewidth=1.2,
    ))

    # ── Note ───────────────────────────────────────────────────────────────────
    if note:
        fig.text(
            x_positions[0], current_y - 0.02,
            note, ha="left", va="top",
            fontsize=7.5, color="#555555", style="italic",
            fontfamily=FONT, linespacing=1.4,
            wrap=True # Wrap the note if it's wider than the table
        )

    # Use a fixed dpi and save; the table is now centered relative to fig_w
    plt.savefig(filename, dpi=180, facecolor="white")
    plt.close()
    print(f"Saved: {filename}")

# ── Export ─────────────────────────────────────────────────────────────────────

TABLE_NOTE = (
    "Note: Provisions lacking a direct classification into the four primary domains in the PAD Codebook "
    "have been grouped under 'Miscellaneous'."
)

render_multi_line_table(df, filename="pad_provisions_table.png", note=TABLE_NOTE)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, text, xy, boxsize=(2.8, 0.8)):
    """Draw a box with centered text."""
    x, y = xy
    width, height = boxsize
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", edgecolor="black", facecolor="#D0E1F9")
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9)

def draw_side_arrow(ax, start_xy, end_xy, label=""):
    """Draw a side arrow left of boxes, label to the right of arrow."""
    x0, y0 = start_xy
    x1, y1 = end_xy
    x_offset = 3.6  # Pfeil direkt am linken Rand der Box
    ax.annotate("",
                xy=(x_offset, y1), xycoords='data',
                xytext=(x_offset, y0), textcoords='data',
                arrowprops=dict(arrowstyle="->", lw=1.5))

    # Label rechts neben dem Pfeil, mittig zwischen den Boxen
    x_text = x_offset + 0.4
    y_text = (y0 + y1) / 2
    ax.text(x_text, y_text, label, va="center", ha="left", fontsize=8)

# Setup
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis('off')

# Y-Positionen der Boxen von oben nach unten
y_positions = [7.5, 5.5, 3.5, 1.5]
labels = [
    "Benchmark-Modell",
    "not_nit-Modell",
    "not_lyser-Modell",
    "low_Input-Modell"
]

# Boxen zeichnen
for label, y in zip(labels, y_positions):
    draw_box(ax, label, (5, y))

# Pfeile mit Beschriftung zwischen den Boxen
draw_side_arrow(ax, (5, 9), (5, 8), "Alle Daten")
draw_side_arrow(ax, (5, 7), (5, 6), "Nitrat-Daten von NF & TTP entfernt")
draw_side_arrow(ax, (5, 5), (5, 4), "Lyser-Geräte entfernt\n(elc, nit, tcd, tsp, doc, toc, tur)")
draw_side_arrow(ax, (5, 3), (5, 2), "GS3 Sensor entfernt\n(ec15, stemp15, evc15)")

plt.tight_layout()
plt.show()

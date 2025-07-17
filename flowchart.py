import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, text, xy, boxsize=(2.8, 0.8), fontsize=12):
    """Draw a box with centered text."""
    x, y = xy
    width, height = boxsize
    box = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                         boxstyle="round,pad=0.1", edgecolor="black", facecolor="#D0E1F9")
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)

def draw_zigzag_arrow(ax, start_xy, end_xy, label="", label_offset=(0.3, 0.1), fontsize=11):
    """Draw diagonal arrow with a label."""
    ax.annotate("",
                xy=end_xy, xycoords='data',
                xytext=start_xy, textcoords='data',
                arrowprops=dict(arrowstyle="->", lw=1.5))
    # Position label near the middle
    mx = (start_xy[0] + end_xy[0]) / 2 + label_offset[0]
    my = (start_xy[1] + end_xy[1]) / 2 + label_offset[1]
    ax.text(mx, my, label, fontsize=fontsize, ha="left", va="center")

# Setup
fig, ax = plt.subplots(figsize=(10, 4.5))  # Angepasste Bildgröße

ax.axis('off')

# Positionen der Boxen (zickzack)
box_positions = [
    (3, 2),   # Box 1
    (6, 4),   # Box 2
    (9, 2),   # Box 3
    (12, 4),  # Box 4
]

labels = [
    "Benchmark-Modell",
    "not_nit-Modell",
    "not_lyser-Modell",
    "low_Input-Modell"
]

# Draw boxes
for label, pos in zip(labels, box_positions):
    draw_box(ax, label, pos, fontsize=12)

# Draw arrows + labels
draw_zigzag_arrow(ax, (0.5, 3.5), (2, 2.5), "Alle Daten", label_offset=(-0.3, +0.3), fontsize=11)
draw_zigzag_arrow(ax, (3.5, 2.5), (5, 3.5), "Nitrat-Daten von\nNF & TTP entfernt", label_offset=(-0.05, -0.25), fontsize=11)
draw_zigzag_arrow(ax, (6.5, 3.5), (8, 2.5), "Lyser-Geräte entfernt\n(elc, nit, tcd, tsp, doc, toc, tur)", label_offset=(-0.21, +0.3), fontsize=11)
draw_zigzag_arrow(ax, (9.5, 2.5), (11, 3.5), "GS3 Sensor entfernt\n(ec15, stemp15, evc15)", label_offset=(-0.05, -0.25), fontsize=11)

# Auto-Achsenbereich passend machen
ax.relim()
ax.autoscale_view()
plt.margins(x=0.1, y=0.05)  # Reduziert Rand
plt.tight_layout(pad=0.2)
plt.show()

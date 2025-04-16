import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from itertools import combinations
import math

# Example data (can be replaced by user input)
coordinates = [(0.4, 0.4), (0.5, 0.5), (0.8, 0.8), (0.5, 0.2)]  # disc centers
bit_mask = [1, 1, 1, 0]  # 1 = active, 0 = inactive
radius = 0.15  # common radius

# Helper to compute area of overlap between two discs
def disc_overlap_area(d, r):
    if d >= 2 * r:
        return 0.0
    elif d <= 0:
        return math.pi * r ** 2
    part1 = r ** 2 * math.acos(d / (2 * r))
    part2 = r ** 2 * math.acos(d / (2 * r))
    part3 = 0.5 * math.sqrt((-d + 2 * r) * (d + 2 * r) * d * d)
    return 2 * part1 - part3

# Compute overlap area between active discs
def compute_total_overlap(coords, mask, r):
    total_overlap = 0.0
    active_coords = [coords[i] for i in range(len(coords)) if mask[i]]
    for (x1, y1), (x2, y2) in combinations(active_coords, 2):
        d = math.hypot(x2 - x1, y2 - y1)
        overlap = max(0.0, 2 * r - d)
        total_overlap += overlap
    return total_overlap

# Plot and save the visualization
def visualize_discs(coords, mask, r, filename="disc_placement.png"):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Disc Placement Visualization")

    for (x, y), active in zip(coords, mask):
        color = 'blue' if active else 'gray'
        alpha = 0.6 if active else 0.2
        circle = Circle((x, y), r, color=color, alpha=alpha)
        ax.add_patch(circle)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(filename)
    plt.close(fig)

# Run functions
total_overlap = compute_total_overlap(coordinates, bit_mask, radius)
visualize_discs(coordinates, bit_mask, radius)

print(total_overlap)

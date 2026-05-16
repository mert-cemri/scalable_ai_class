# EVOLVE-BLOCK-START
"""Hexagonal circle packing for n=26 circles in unit square"""
import numpy as np


def construct_packing():
    """
    Hexagonal packing of 26 circles using staggered rows for density.
    Pattern: 6+5+6+5+4 = 26 circles with alternating offsets.
    """
    centers = []
    
    # Row configuration: (count, x_start, x_end) for each row
    row_configs = [
        (6, 0.04, 0.96),  # Row 0: 6 circles, full width
        (5, 0.10, 0.90),  # Row 1: 5 circles, offset inward
        (6, 0.04, 0.96),  # Row 2: 6 circles, full width
        (5, 0.10, 0.90),  # Row 3: 5 circles, offset inward
        (4, 0.15, 0.85),  # Row 4: 4 circles, more centered
    ]
    
    # Vertical spacing optimized for square container
    y_positions = [0.12, 0.32, 0.50, 0.68, 0.88]
    
    for row_idx, (count, x_start, x_end) in enumerate(row_configs):
        y = y_positions[row_idx]
        x_step = (x_end - x_start) / (count - 1)
        for i in range(count):
            x = x_start + i * x_step
            centers.append([x, y])
    
    centers = np.array(centers)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute max radii using iterative relaxation to resolve overlaps.
    """
    n = centers.shape[0]
    # Initialize radii based on distance to square boundaries
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Iteratively resolve overlaps with multiple passes
    for iteration in range(100):
        changed = False
        max_change = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if radii[i] + radii[j] > dist + 1e-10:
                    excess = radii[i] + radii[j] - dist
                    total_r = radii[i] + radii[j]
                    if total_r > 0:
                        scale_i = radii[i] / total_r
                        scale_j = radii[j] / total_r
                        reduction_i = excess * scale_i * 0.5
                        reduction_j = excess * scale_j * 0.5
                        new_r_i = max(0.0001, radii[i] - reduction_i)
                        new_r_j = max(0.0001, radii[j] - reduction_j)
                        max_change = max(max_change, abs(radii[i] - new_r_i), abs(radii[j] - new_r_j))
                        radii[i] = new_r_i
                        radii[j] = new_r_j
                        changed = True
        
        if not changed or max_change < 1e-8:
            break
    
    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)

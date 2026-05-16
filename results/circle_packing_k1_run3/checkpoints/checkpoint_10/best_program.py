# EVOLVE-BLOCK-START
"""Hexagonal grid circle packing for n=26 circles in unit square"""
import numpy as np


def construct_packing():
    """
    Hexagonal staggered grid packing of 26 circles in unit square.
    Uses offset rows for better density, similar to optimal hexagonal packing.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = []
    
    # Hexagonal grid parameters optimized for n=26 in unit square
    # Row height ~sqrt(3)/2 * 2r for hexagonal packing
    row_h = 0.162
    col_w = 0.148
    x_base = 0.076
    x_offset = 0.074
    y_base = 0.076
    
    y = y_base
    row = 0
    while len(centers) < n and y < 1 - y_base:
        # Staggered rows: even rows start at x_base, odd rows offset
        x_start = x_base if row % 2 == 0 else x_base + x_offset
        for i in range(6):
            x = x_start + i * col_w
            if len(centers) < n:
                centers.append([x, y])
        y += row_h
        row += 1
    
    centers = np.array(centers[:n])
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute maximum radii satisfying boundary and overlap constraints.
    Uses iterative constraint satisfaction for convergence.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
    
    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    # Initialize radii by boundary distance
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    # Iteratively satisfy circle-circle constraints until convergence
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    # Adjust both radii proportionally to touch but not overlap
                    ratio = radii[i] / (radii[i] + radii[j])
                    radii[i] = dist * ratio
                    radii[j] = dist * (1 - ratio)
                    changed = True
        if not changed:
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

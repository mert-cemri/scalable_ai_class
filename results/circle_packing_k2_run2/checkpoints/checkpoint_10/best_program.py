# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Hexagonal grid packing for n=26 circles in unit square.
    Uses hexagonal spacing (optimal density) adapted to square container.
    Places circles in staggered rows with corner optimization.
    """
    n = 26
    centers = []
    
    # Hexagonal grid: 5 rows with staggered columns for dense packing
    # Row spacing = sqrt(3)/2 * base_spacing, col spacing = base_spacing
    base_spacing = 0.19
    for row in range(5):
        y = 0.1 + row * base_spacing * np.sqrt(3) / 2
        offset = base_spacing / 2 if row % 2 == 1 else 0  # Stagger odd rows
        for col in range(6):
            if len(centers) >= n:
                break
            x = offset + col * base_spacing
            # Ensure within bounds with margin
            if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98:
                centers.append([x, y])
    
    # Fill remaining positions with corner optimization
    corners = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
    for corner in corners:
        if len(centers) >= n:
            break
        if not any(np.linalg.norm(np.array(c) - np.array(corner)) < 0.15 for c in centers):
            centers.append(corner)
    
    centers = np.array(centers[:n])
    
    # Compute radii: max distance to boundaries
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Iteratively resolve overlaps by scaling down proportionally
    for _ in range(200):
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    s = d / (radii[i] + radii[j])
                    radii[i] *= s
                    radii[j] *= s
    
    return centers, radii, np.sum(radii)


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

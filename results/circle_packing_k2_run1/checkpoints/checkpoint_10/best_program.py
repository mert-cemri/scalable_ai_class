# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using staggered grid"""
import numpy as np


def construct_packing():
    """
    Optimized hexagonal packing for n=26 circles.
    Uses 6 rows with pattern 5-4-5-4-5-3 = 26 circles for better square utilization.
    Includes radius growth phase to maximize sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Optimized row positions with hexagonal spacing for square container
    # Pattern 5-4-5-4-5-3 fills corners better than 4-5-4-5-4-4
    rows_config = [
        (0.08, 5, [0.05, 0.22, 0.40, 0.58, 0.75]),  # Row 0: 5 circles
        (0.26, 4, [0.15, 0.38, 0.62, 0.85]),        # Row 1: 4 circles (staggered)
        (0.44, 5, [0.05, 0.22, 0.40, 0.58, 0.75]),  # Row 2: 5 circles
        (0.62, 4, [0.15, 0.38, 0.62, 0.85]),        # Row 3: 4 circles (staggered)
        (0.80, 5, [0.05, 0.22, 0.40, 0.58, 0.75]),  # Row 4: 5 circles
        (0.95, 3, [0.25, 0.50, 0.75]),              # Row 5: 3 circles (top gap)
    ]
    
    idx = 0
    for y, count, cols in rows_config:
        for x in cols:
            centers[idx] = [x, y]
            radii[idx] = min(x, y, 1-x, 1-y)
            idx += 1
    
    # Phase 1: Resolve overlaps with proportional scaling
    for _ in range(50):
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d > 0 and radii[i] + radii[j] > d:
                    total = radii[i] + radii[j]
                    scale = d / total
                    radii[i] *= scale
                    radii[j] *= scale
    
    # Phase 2: Grow radii where possible (key improvement)
    for _ in range(30):
        for i in range(n):
            # Calculate max possible radius for circle i
            max_r = min(centers[i][0], centers[i][1], 
                       1-centers[i][0], 1-centers[i][1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d - radii[j])
            if max_r > radii[i]:
                radii[i] = max_r
    
    # Phase 3: Final overlap check with conservative scaling
    for _ in range(10):
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d > 0 and radii[i] + radii[j] > d:
                    total = radii[i] + radii[j]
                    scale = d / total
                    radii[i] *= scale
                    radii[j] *= scale
    
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

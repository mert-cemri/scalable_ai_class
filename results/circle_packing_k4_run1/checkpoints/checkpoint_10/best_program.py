# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 using Delaunay triangulation"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def construct_packing():
    """
    Optimize circle packing using Delaunay triangulation to identify
    critical neighbor constraints. Maximizes sum of radii by optimizing
    center positions with sparse constraint evaluation.
    
    The Delaunay triangulation identifies which circles are geometric neighbors
    and should potentially touch. This reduces complexity from O(n^2) to O(n).
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Generate initial hexagonal configuration
    centers = generate_initial_configuration(n)
    
    # Optimize using Delaunay-based constraints with multiple restarts
    best_result = None
    best_sum = 0
    
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        x0 = centers.flatten() + np.random.uniform(-0.02, 0.02, size=2*n)
        x0 = np.clip(x0, 0.05, 0.95)
        
        result = minimize(
            negative_sum_radii_delaunay,
            x0,
            method='SLSQP',
            bounds=[(0, 1)] * (2 * n),
            options={'ftol': 1e-10, 'maxiter': 500, 'disp': False}
        )
        
        sum_radii = -result.fun
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_result = result
    
    optimized_centers = best_result.x.reshape(n, 2)
    radii = compute_radii_delaunay(optimized_centers)
    sum_radii = np.sum(radii)
    
    return optimized_centers, radii, sum_radii


def generate_initial_configuration(n):
    """Generate initial hexagonal lattice configuration for optimization."""
    centers = []
    spacing = 0.185
    y_step = spacing * np.sqrt(3) / 2
    row_counts = [5, 4, 5, 4, 5, 3]
    
    for row, count in enumerate(row_counts):
        offset = spacing / 2 if row % 2 == 1 else 0
        row_width = (count - 1) * spacing
        x_start = (1 - row_width) / 2 + offset
        
        for col in range(count):
            x = x_start + col * spacing
            y = 0.11 + row * y_step
            centers.append([x, y])
    
    return np.array(centers[:n])


def negative_sum_radii_delaunay(x):
    """Objective function: negative sum of radii (for minimization)."""
    n = 26
    centers = x.reshape(n, 2)
    radii = compute_radii_delaunay(centers)
    return -np.sum(radii)


def compute_radii_delaunay(centers):
    """
    Compute maximum radii using Delaunay triangulation.
    Each radius is constrained by boundary distance and half distance to Delaunay neighbors.
    
    The Delaunay triangulation identifies the sparse set of critical constraints
    (circles that should potentially touch), reducing computational complexity.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
    
    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    
    # Initialize with boundary constraints (distance to nearest edge)
    radii = np.array([
        min(c[0], c[1], 1 - c[0], 1 - c[1]) for c in centers
    ])
    
    # Build Delaunay triangulation to find neighbor relationships
    try:
        tri = Delaunay(centers)
    except Exception:
        # Fallback to boundary-only if triangulation fails (collinear points)
        return radii
    
    # For each circle, check Delaunay neighbors (sparse constraint evaluation)
    for i in range(n):
        for simplex in tri.simplices:
            if i in simplex:
                for j in simplex:
                    if i != j:
                        dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                        radii[i] = min(radii[i], dist / 2)
    
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

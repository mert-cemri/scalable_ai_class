# EVOLVE-BLOCK-START
"""Joint coordinate-radius optimization for n=26 circle packing using SLSQP"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint optimization of circle centers and radii using SLSQP.
    Optimizes 78 variables (26 circles × 3: x, y, r) with explicit constraints.
    
    Constraints:
    - Boundary: x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
    - Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Generate initial configuration
    x0, r0 = generate_initial_configuration()
    x_init = np.concatenate([x0.flatten(), r0])
    
    def objective(x):
        """Objective: negative sum of radii (for minimization)."""
        return -np.sum(x[2*n:])
    
    def non_overlap_constraint(x):
        """Compute all pairwise non-overlap constraints.
        Returns constraint values >= 0 for valid configurations.
        Constraint: dist_ij^2 - (r_i + r_j)^2 >= 0
        """
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx**2 + dy**2
                min_dist_sq = (radii[i] + radii[j])**2
                constraints.append(dist_sq - min_dist_sq)
        
        return np.array(constraints)
    
    def boundary_constraint(x):
        """Compute boundary constraints.
        Returns constraint values >= 0 for valid configurations.
        Constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
        """
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        
        constraints = []
        for i in range(n):
            constraints.append(centers[i, 0] - radii[i])  # x_i - r_i >= 0
            constraints.append(1 - centers[i, 0] - radii[i])  # 1 - x_i - r_i >= 0
            constraints.append(centers[i, 1] - radii[i])  # y_i - r_i >= 0
            constraints.append(1 - centers[i, 1] - radii[i])  # 1 - y_i - r_i >= 0
        
        return np.array(constraints)
    
    # Bounds for each variable
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    # Nonlinear constraints (inequality: g(x) >= 0)
    non_overlap = {'type': 'ineq', 'fun': non_overlap_constraint}
    boundary = {'type': 'ineq', 'fun': boundary_constraint}
    
    # Run optimization with multiple restarts
    best_result = None
    best_sum = -np.inf
    
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        x0_perturbed = x_init + np.random.uniform(-0.02, 0.02, size=3*n)
        x0_perturbed = np.clip(x0_perturbed, 0.01, 0.99)
        # Ensure radius bounds
        x0_perturbed[2*n:] = np.clip(x0_perturbed[2*n:], 0.01, 0.4)
        
        result = minimize(
            objective,
            x0_perturbed,
            method='SLSQP',
            bounds=bounds,
            constraints=[non_overlap, boundary],
            options={
                'ftol': 1e-10,
                'maxiter': 800,
                'disp': False,
                'eps': 1e-8
            }
        )
        
        current_sum = -result.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = result
    
    x_opt = best_result.x
    centers = x_opt[:2*n].reshape(n, 2)
    radii = x_opt[2*n:]
    
    return centers, radii, np.sum(radii)


def generate_initial_configuration():
    """
    Generate initial hexagonal lattice configuration with reasonable radii.
    Row pattern: 5-4-5-4-5-3 (26 total) with hexagonal spacing.
    
    Returns:
        Tuple of (centers, radii)
    """
    centers = []
    spacing = 0.192
    y_step = spacing * np.sqrt(3) / 2
    row_counts = [5, 4, 5, 4, 5, 3]
    
    for row, count in enumerate(row_counts):
        offset = spacing / 2 if row % 2 == 1 else 0
        row_width = (count - 1) * spacing
        x_start = (1 - row_width) / 2 + offset
        
        for col in range(count):
            x = x_start + col * spacing
            y = 0.12 + row * y_step
            centers.append([x, y])
    
    centers = np.array(centers[:26])
    
    # Compute initial radii based on boundary and pairwise distances
    n = 26
    radii = np.array([
        min(c[0], c[1], 1 - c[0], 1 - c[1]) for c in centers
    ])
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            half_dist = dist / 2
            radii[i] = min(radii[i], half_dist)
            radii[j] = min(radii[j], half_dist)
    
    return centers, radii


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

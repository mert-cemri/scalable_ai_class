# EVOLVE-BLOCK-START
"""Joint optimization with multi-start for n=26 circles in unit square"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Multi-start joint optimization of all 78 variables (x, y, r for 26 circles)
    using SLSQP. Runs from multiple perturbed hexagonal lattice configurations
    to escape local optima and find better global solutions.
    
    Optimization problem:
    - Variables: 78 (x_i, y_i, r_i for i=0..25)
    - Objective: maximize sum of radii (minimize negative sum)
    - Constraints: 
      1. Boundary: r_i <= min(x_i, y_i, 1-x_i, 1-y_i) for all i
      2. Non-overlap: dist(i,j) >= r_i + r_j for all pairs
    """
    n = 26
    
    # Generate base hexagonal lattice configuration
    row_sizes = [6, 5, 5, 5, 5]
    r_est = 0.098
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    total_width = (max(row_sizes) - 1) * dx
    total_height = (len(row_sizes) - 1) * dy
    x_start = (1 - total_width) / 2
    y_start = (1 - total_height) / 2
    
    base_centers = []
    for row_idx, row_n in enumerate(row_sizes):
        y = y_start + row_idx * dy
        x_offset = dx / 2 if row_idx % 2 == 1 else 0
        for col_idx in range(row_n):
            x = x_start + col_idx * dx + x_offset
            base_centers.append([x, y])
    
    base_centers = np.array(base_centers[:26])
    initial_radii = np.ones(n) * 0.098
    
    # Bounds: x, y in [0.01, 0.99], r in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
        bounds.append((0.001, 0.5))
    
    # Precompute constraint indices for efficiency
    constraint_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_result = None
    best_sum = -1
    
    # Multi-start optimization with perturbed initial configurations
    np.random.seed(42)
    for start_idx in range(5):
        # Perturb the base configuration slightly
        if start_idx == 0:
            perturbation = 0.0
        else:
            perturbation = 0.02 * start_idx
        
        perturbed_centers = base_centers + np.random.uniform(
            -perturbation, perturbation, base_centers.shape
        )
        perturbed_centers = np.clip(perturbed_centers, 0.05, 0.95)
        
        # Initialize variable vector
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = perturbed_centers[i, 0]
            x0[3*i+1] = perturbed_centers[i, 1]
            x0[3*i+2] = initial_radii[i]
        
        # Objective: minimize negative sum of radii
        def objective(x):
            return -np.sum(x[2::3])
        
        # Build constraints efficiently
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            idx = i
            def constraint(x, i=idx):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                return min(x_i, y_i, 1-x_i, 1-y_i) - r_i
            constraints.append({'type': 'ineq', 'fun': constraint})
        
        # Non-overlap constraints
        for idx_i, idx_j in constraint_pairs:
            i, j = idx_i, idx_j
            def constraint(x, i=i, j=j):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                return np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2) - r_i - r_j
            constraints.append({'type': 'ineq', 'fun': constraint})
        
        # Run optimization with tuned parameters
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-12, 'eps': 1e-12}
        )
        
        # Extract sum of radii
        current_sum = np.sum(result.x[2::3])
        
        # Keep best result
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = result
    
    # Extract final configuration from best result
    final_x = best_result.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i, 0] = final_x[3*i]
        final_centers[i, 1] = final_x[3*i+1]
        final_radii[i] = final_x[3*i+2]
    
    return final_centers, final_radii, np.sum(final_radii)


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

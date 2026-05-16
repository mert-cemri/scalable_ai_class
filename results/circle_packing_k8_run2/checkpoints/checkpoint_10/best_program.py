# EVOLVE-BLOCK-START
"""Joint position-radius optimization for n=26 circle packing using SLSQP"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint position and radius optimization for 26 circles using SLSQP.
    
    Uses scipy.optimize.minimize with SLSQP to simultaneously optimize
    all 52 variables (26 x-coords, 26 y-coords, 26 radii). Maximizes
    sum(radii) subject to non-overlap and boundary constraints.
    Uses multiple random initializations to escape local minima.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum_radii = 0.0
    best_solution = None
    
    # Multiple restarts to escape local minima in non-convex problem
    for seed in range(15):
        np.random.seed(seed)
        
        # Initialize with hexagonal-like pattern + small noise
        centers = _generate_hexagonal_init(n)
        centers += np.random.uniform(-0.03, 0.03, size=centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        
        # Initial radii based on boundary distances
        radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
        
        # Pack all variables: [x0, y0, x1, y1, ..., x25, y25, r0, r1, ..., r25]
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Define bounds for each variable
        bounds = []
        for i in range(n):
            bounds.append((0.01, 0.99))  # x_i
            bounds.append((0.01, 0.99))  # y_i
        for i in range(n):
            bounds.append((0.0, 1.0))    # r_i
        
        # Objective: minimize negative sum of radii
        def objective(x):
            radii = x[2*n:]
            return -np.sum(radii)
        
        # Constraint: non-overlap for all pairs (dist >= r_i + r_j)
        def non_overlap_constraint(x):
            centers = x[:2*n].reshape(n, 2)
            radii = x[2*n:]
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    constraints.append(dist - radii[i] - radii[j])
            return np.array(constraints)
        
        # Constraint: boundary (r_i <= distance to nearest edge)
        def boundary_constraint(x):
            centers = x[:2*n].reshape(n, 2)
            radii = x[2*n:]
            constraints = []
            for i in range(n):
                min_edge = min(centers[i][0], centers[i][1], 
                              1-centers[i][0], 1-centers[i][1])
                constraints.append(min_edge - radii[i])
            return np.array(constraints)
        
        # Run SLSQP optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': non_overlap_constraint},
                {'type': 'ineq', 'fun': boundary_constraint}
            ],
            options={'ftol': 1e-10, 'maxiter': 1000}
        )
        
        if result.success and -result.fun > best_sum_radii:
            best_sum_radii = -result.fun
            best_solution = result.x
    
    # Extract final solution
    centers = best_solution[:2*n].reshape(n, 2)
    radii = best_solution[2*n:]
    
    return centers, radii, best_sum_radii


def _generate_hexagonal_init(n):
    """
    Generate initial hexagonal packing positions for n circles.
    
    Uses staggered rows with hexagonal spacing as starting point
    for the joint optimization.
    
    Args:
        n: Number of circles to place
    
    Returns:
        np.array of shape (n, 2) with (x, y) coordinates
    """
    centers = []
    
    # 5 rows with [5, 5, 5, 5, 6] = 26 circles total
    row_counts = [5, 5, 5, 5, 6]
    h_spacing = 0.178  # Horizontal spacing
    v_spacing = 0.173  # Vertical spacing (sqrt(3)/2 * h_spacing)
    y_start = 0.108    # Bottom offset
    
    for row_idx, n_circles in enumerate(row_counts):
        # Stagger odd rows for hexagonal packing
        x_offset = 0.108 if row_idx % 2 == 0 else 0.197
        
        for col_idx in range(n_circles):
            x = x_offset + col_idx * h_spacing
            y = y_start + row_idx * v_spacing
            
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append([x, y])
    
    return np.array(centers[:n])


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

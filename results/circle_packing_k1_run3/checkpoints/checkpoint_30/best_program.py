# EVOLVE-BLOCK-START
"""Joint position-radius optimization for n=26 circles using SLSQP with improvements"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint optimization of positions and radii using scipy.optimize.minimize with SLSQP.
    Optimizes all 78 variables (26 positions × 2 + 26 radii) simultaneously.
    Uses multiple restarts with different hexagonal initializations for better global optima.
    Maximizes sum of radii subject to:
    - Circle containment in unit square (x±r, y±r within [0,1])
    - Pairwise non-overlap (distance >= r_i + r_j)
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    from scipy.optimize import minimize
    
    n = 26
    TOLERANCE = 1e-10  # Numerical tolerance for constraints
    
    # Objective: minimize negative sum of radii (to maximize sum of radii)
    def objective(X):
        return -np.sum(X[2::3])
    
    # Boundary constraints for each circle with numerical tolerance
    def make_boundary_constraints():
        constraints = []
        for i in range(n):
            # x_i - r_i >= tolerance
            def boundary_x_min(X, i=i): return X[3*i] - X[3*i + 2] - TOLERANCE
            # 1 - x_i - r_i >= tolerance
            def boundary_x_max(X, i=i): return 1.0 - X[3*i] - X[3*i + 2] - TOLERANCE
            # y_i - r_i >= tolerance
            def boundary_y_min(X, i=i): return X[3*i + 1] - X[3*i + 2] - TOLERANCE
            # 1 - y_i - r_i >= tolerance
            def boundary_y_max(X, i=i): return 1.0 - X[3*i + 1] - X[3*i + 2] - TOLERANCE
            constraints.extend([
                {'type': 'ineq', 'fun': boundary_x_min},
                {'type': 'ineq', 'fun': boundary_x_max},
                {'type': 'ineq', 'fun': boundary_y_min},
                {'type': 'ineq', 'fun': boundary_y_max}
            ])
        return constraints
    
    # Pairwise distance constraints: dist(i,j) >= r_i + r_j
    def make_distance_constraints():
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                def dist_constraint(X, i=i, j=j):
                    xi, yi, ri = X[3*i], X[3*i + 1], X[3*i + 2]
                    xj, yj, rj = X[3*j], X[3*j + 1], X[3*j + 2]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return dist - ri - rj - TOLERANCE
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        return constraints
    
    # Generate hexagonal initialization with given parameters
    def generate_hexagonal_init(seed, row_h, col_w, x_base, y_base, x_offset):
        np.random.seed(seed)
        init_x = []
        y = y_base
        row = 0
        while len(init_x) < n * 3 and y < 1 - y_base:
            x_start = x_base if row % 2 == 0 else x_base + x_offset
            col = 0
            while len(init_x) < n * 3 and x_start + col * col_w < 1 - x_base:
                x = x_start + col * col_w
                r = min(x, y, 1-x, 1-y) * 0.95
                init_x.extend([x, y, r])
                col += 1
            y += row_h
            row += 1
        
        # Pad if needed
        while len(init_x) < n * 3:
            cx, cy = 0.5 + 0.1 * np.random.randn(), 0.5 + 0.1 * np.random.randn()
            cx, cy = max(0.1, min(0.9, cx)), max(0.1, min(0.9, cy))
            r = min(cx, cy, 1-cx, 1-cy) * 0.5
            init_x.extend([cx, cy, r])
        
        return init_x[:n * 3]
    
    # Bounds for each variable (x, y, r for each circle)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    # Combine all constraints
    constraints = make_boundary_constraints() + make_distance_constraints()
    
    # Multiple restarts with different hexagonal parameters
    best_result = None
    best_sum = -np.inf
    
    # Try multiple initialization strategies
    init_configs = [
        (42, 0.165, 0.150, 0.080, 0.080, 0.075),
        (123, 0.170, 0.145, 0.075, 0.075, 0.070),
        (456, 0.160, 0.155, 0.085, 0.085, 0.080),
        (789, 0.168, 0.148, 0.078, 0.078, 0.073),
        (321, 0.162, 0.152, 0.082, 0.082, 0.077),
    ]
    
    for seed, row_h, col_w, x_base, y_base, x_offset in init_configs:
        init_x = generate_hexagonal_init(seed, row_h, col_w, x_base, y_base, x_offset)
        
        result = minimize(
            objective,
            init_x,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 8000, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success or result.nit > 100:
            sum_radii = -result.fun
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = result
    
    # Fallback if all optimizations failed
    if best_result is None:
        init_x = generate_hexagonal_init(42, 0.165, 0.150, 0.080, 0.080, 0.075)
        best_result = minimize(
            objective, init_x, method='SLSQP', bounds=bounds,
            constraints=constraints, options={'maxiter': 8000, 'ftol': 1e-10, 'disp': False}
        )
    
    # Extract results
    X_opt = best_result.x
    centers = np.array([[X_opt[3*i], X_opt[3*i + 1]] for i in range(n)])
    radii = np.array([X_opt[3*i + 2] for i in range(n)])
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii


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

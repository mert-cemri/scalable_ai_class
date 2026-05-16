# EVOLVE-BLOCK-START
"""Delaunay-constrained circle packing optimization for n=26 circles"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def construct_packing():
    """
    Circle packing using Delaunay triangulation for neighbor-constrained optimization.
    Builds a Delaunay triangulation of circle centers to identify critical constraint pairs,
    then optimizes only those neighbor constraints plus boundary constraints.
    This reduces constraints from 325 to ~156, enabling faster convergence.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    TOLERANCE = 1e-10
    
    # Generate hexagonal-like initial configuration
    def generate_initial_config(seed):
        np.random.seed(seed)
        centers = []
        # Staggered rows for hexagonal pattern
        for row in range(6):
            y = 0.1 + row * 0.15
            x_start = 0.08 if row % 2 == 0 else 0.12
            for col in range(5):
                if len(centers) < n:
                    x = x_start + col * 0.16
                    centers.append([x, y])
        # Add remaining circles if needed
        while len(centers) < n:
            cx, cy = 0.5 + 0.2 * np.random.randn(), 0.5 + 0.2 * np.random.randn()
            cx, cy = max(0.1, min(0.9, cx)), max(0.1, min(0.9, cy))
            centers.append([cx, cy])
        return np.array(centers[:n])
    
    # Extract neighbor pairs from Delaunay triangulation
    def get_delaunay_neighbors(centers):
        tri = Delaunay(centers)
        neighbors = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = sorted([simplex[i], simplex[j]])
                    neighbors.add((a, b))
        return list(neighbors)
    
    # Objective: maximize sum of radii
    def objective(X):
        return -np.sum(X[2::3])
    
    # Boundary constraints
    def make_boundary_constraints():
        constraints = []
        for i in range(n):
            constraints.extend([
                {'type': 'ineq', 'fun': lambda X, i=i: X[3*i] - X[3*i + 2] - TOLERANCE},
                {'type': 'ineq', 'fun': lambda X, i=i: 1.0 - X[3*i] - X[3*i + 2] - TOLERANCE},
                {'type': 'ineq', 'fun': lambda X, i=i: X[3*i + 1] - X[3*i + 2] - TOLERANCE},
                {'type': 'ineq', 'fun': lambda X, i=i: 1.0 - X[3*i + 1] - X[3*i + 2] - TOLERANCE}
            ])
        return constraints
    
    # Neighbor distance constraints from Delaunay
    def make_neighbor_constraints(neighbor_pairs):
        constraints = []
        for i, j in neighbor_pairs:
            def dist_constraint(X, i=i, j=j):
                xi, yi, ri = X[3*i], X[3*i + 1], X[3*i + 2]
                xj, yj, rj = X[3*j], X[3*j + 1], X[3*j + 2]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                return dist - ri - rj - TOLERANCE
            constraints.append({'type': 'ineq', 'fun': dist_constraint})
        return constraints
    
    # Variable bounds
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    # Multiple restarts with different seeds
    best_result = None
    best_sum = -np.inf
    
    for seed in [42, 123, 456, 789, 321, 654, 987]:
        centers = generate_initial_config(seed)
        X0 = []
        for x, y in centers:
            r = min(x, y, 1-x, 1-y) * 0.9
            X0.extend([x, y, r])
        
        # Use Delaunay to find neighbor constraints
        neighbors = get_delaunay_neighbors(centers)
        boundary_constraints = make_boundary_constraints()
        neighbor_constraints = make_neighbor_constraints(neighbors)
        constraints = boundary_constraints + neighbor_constraints
        
        result = minimize(
            objective,
            X0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 6000, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success or result.nit > 50:
            sum_radii = -result.fun
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = result
    
    # Fallback if all optimizations failed
    if best_result is None:
        centers = generate_initial_config(42)
        X0 = []
        for x, y in centers:
            r = min(x, y, 1-x, 1-y) * 0.9
            X0.extend([x, y, r])
        neighbors = get_delaunay_neighbors(centers)
        constraints = make_boundary_constraints() + make_neighbor_constraints(neighbors)
        best_result = minimize(objective, X0, method='SLSQP', bounds=bounds, constraints=constraints)
    
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

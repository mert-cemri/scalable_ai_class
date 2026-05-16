"""Voronoi-based gap detection and SLSQP refinement for n=26 circles in unit square"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi


def construct_packing():
    """
    Optimize 26 circles using Voronoi-based gap detection with SLSQP refinement.
    
    Approach:
    1. Start with hexagonal packing configuration
    2. Compute Voronoi diagram to identify spatial gaps
    3. Use Voronoi vertices to determine expansion potential for each circle
    4. Apply SLSQP optimization with better initial radii from Voronoi analysis
    5. Multiple restarts to escape local minima
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Build hexagonal initial guess
    row_counts = [5, 4, 5, 4, 5, 3]
    r_init = 1 / (2 + 5 * np.sqrt(3))
    
    centers_init = []
    y_spacing = r_init * np.sqrt(3)
    
    for row_idx, count in enumerate(row_counts):
        y = r_init + row_idx * y_spacing
        x_start = r_init if row_idx % 2 == 0 else 2 * r_init
        
        for col_idx in range(count):
            x = x_start + col_idx * 2 * r_init
            centers_init.append([x, y])
    
    centers = np.array(centers_init)
    
    # Voronoi-based radius initialization
    # Compute Voronoi diagram to identify gaps where circles can expand
    try:
        vor = Voronoi(centers)
    except:
        vor = None
    
    radii = np.zeros(n)
    if vor is not None:
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region:
                # Infinite region (boundary circles) - use distance to neighbors
                min_dist = np.inf
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(centers[i] - centers[j])
                        min_dist = min(min_dist, dist)
                radii[i] = min_dist / 2.5
            else:
                # Finite region - use distance to closest Voronoi vertex
                vertices = vor.vertices[region]
                if len(vertices) > 0:
                    min_dist = np.min(np.linalg.norm(vertices - centers[i], axis=1))
                    radii[i] = min_dist * 0.95
                else:
                    radii[i] = r_init
    else:
        radii = np.full(n, r_init)
    
    # Scale to ensure initial feasibility
    min_sep = np.inf
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_sep = min(min_sep, dist / (radii[i] + radii[j]))
    
    if min_sep < 1.0:
        radii = radii * min_sep * 0.92
    
    radii_init = radii.copy()
    
    # Combine into optimization variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.concatenate([centers.flatten(), radii_init])
    
    # Objective: maximize sum of radii (minimize negative)
    def objective(x):
        return -np.sum(x[2*n:])
    
    # Overlap constraints: distance >= r1 + r2
    def overlap_constraints(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    def boundary_constraints(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        constraints = []
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            constraints.extend([x - r, 1 - r - x, y - r, 1 - r - y])
        return np.array(constraints)
    
    # Non-negative radii
    def radius_constraints(x):
        return x[2*n:]
    
    # Set up constraints
    constraints = [
        {'type': 'ineq', 'fun': overlap_constraints},
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': radius_constraints}
    ]
    
    # Variable bounds
    bounds = [(0, 1) for _ in range(2*n)] + [(0, None) for _ in range(n)]
    
    # Multiple restarts with varied perturbations for better exploration
    best_result = None
    best_sum = 0
    
    for seed in range(10):
        np.random.seed(seed)
        # Use smaller perturbations for more precise optimization
        perturbation = 0.003 * np.random.randn(3*n)
        x0_perturbed = x0 + perturbation
        
        try:
            result = minimize(
                objective, x0_perturbed, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success and np.isfinite(result.fun):
                sum_radii = -result.fun
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = result
        except:
            pass
    
    # Fallback to hexagonal if optimization fails
    if best_result is None:
        centers = centers
        radii = radii_init
        sum_radii = np.sum(radii)
    else:
        centers = best_result.x[:2*n].reshape(n, 2)
        radii = best_result.x[2*n:]
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

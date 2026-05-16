"""Basinhopping optimization for n=26 circles in unit square"""
import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial import Voronoi


def construct_packing():
    """
    Basinhopping optimization for n=26 circles using SLSQP local solver.
    
    Approach:
    1. Start with hexagonal packing configuration
    2. Use Voronoi-based radius initialization
    3. Apply basinhopping with SLSQP local solver for global exploration
    4. Two-phase optimization: large steps (0.1) then small steps (0.01)
    5. Final SLSQP refinement with tight tolerances
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Build hexagonal initial guess: 5-4-5-4-5-3 = 26 circles
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
    
    # Voronoi-based radius initialization for better starting radii
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
                min_dist = np.inf
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(centers[i] - centers[j])
                        min_dist = min(min_dist, dist)
                radii[i] = min_dist / 2.5
            else:
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
            xi, yi = centers[i]
            ri = radii[i]
            constraints.extend([xi - ri, 1 - ri - xi, yi - ri, 1 - ri - yi])
        return np.array(constraints)
    
    # Non-negative radii
    def radius_constraints(x):
        return x[2*n:]
    
    # Set up constraints for SLSQP
    constraints = [
        {'type': 'ineq', 'fun': overlap_constraints},
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': radius_constraints}
    ]
    
    # Variable bounds
    bounds = [(0, 1) for _ in range(2*n)] + [(0, None) for _ in range(n)]
    
    # Phase 1: Large step basinhopping for global exploration
    best_result = None
    best_sum = 0
    
    for seed in range(3):
        np.random.seed(seed)
        x0_perturbed = x0 + 0.01 * np.random.randn(3*n)
        
        try:
            # Phase 1: Large steps (0.1) for exploration
            result1 = basinhopping(
                objective, x0_perturbed,
                minimizer_kwargs={'method': 'SLSQP', 
                                  'constraints': constraints,
                                  'bounds': bounds,
                                  'options': {'maxiter': 200, 'ftol': 1e-10}},
                stepsize=0.1,
                niter=150,
                niter_success=30,
                T=0.1,
                seed=seed
            )
            
            if result1.success and np.isfinite(result1.fun):
                x0_refine = result1.x
                
                # Phase 2: Small steps (0.01) for refinement
                result2 = basinhopping(
                    objective, x0_refine,
                    minimizer_kwargs={'method': 'SLSQP', 
                                      'constraints': constraints,
                                      'bounds': bounds,
                                      'options': {'maxiter': 300, 'ftol': 1e-12}},
                    stepsize=0.01,
                    niter=100,
                    niter_success=50,
                    T=0.05,
                    seed=seed
                )
                
                if result2.success and np.isfinite(result2.fun):
                    sum_radii = -result2.fun
                    if sum_radii > best_sum:
                        best_sum = sum_radii
                        best_result = result2
        except Exception:
            pass
    
    # Final SLSQP refinement with tight tolerances
    if best_result is not None:
        try:
            result = minimize(
                objective, best_result.x, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            if result.success and np.isfinite(result.fun):
                sum_radii = -result.fun
                if sum_radii > best_sum:
                    best_result = result
        except Exception:
            pass
    
    # Fallback to hexagonal if optimization fails
    if best_result is None:
        return centers, radii_init, np.sum(radii_init)
    else:
        centers = best_result.x[:2*n].reshape(n, 2)
        radii = best_result.x[2*n:]
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

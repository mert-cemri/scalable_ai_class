# EVOLVE-BLOCK-START
"""Differential evolution + SLSQP for n=26 circles in unit square"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi


def construct_packing():
    """
    Differential evolution + SLSQP optimization for n=26 circles.
    
    Approach:
    1. Initialize with hexagonal grid pattern (5+5+6+5+5 = 26 circles)
    2. Use differential evolution to globally optimize radii with fixed relative positions
    3. Apply local SLSQP refinement for final optimization
    4. This explores non-convex constraint landscape better than gradient methods
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Initialize with hexagonal grid pattern: 5+5+6+5+5 = 26 circles
    spacing = 0.102
    row_spacing = spacing * np.sqrt(3)
    total_height = 4 * row_spacing
    base_y = (1 - total_height) / 2
    
    centers = np.zeros((n, 2))
    idx = 0
    
    row1_y = base_y
    row1_x_start = 0.051
    for i in range(5):
        centers[idx] = [row1_x_start + i * spacing * 2, row1_y]
        idx += 1
    
    row2_y = base_y + row_spacing
    row2_x_start = row1_x_start + spacing
    for i in range(5):
        centers[idx] = [row2_x_start + i * spacing * 2, row2_y]
        idx += 1
    
    row3_y = base_y + 2 * row_spacing
    row3_x_start = 0.026
    for i in range(6):
        centers[idx] = [row3_x_start + i * spacing * 2, row3_y]
        idx += 1
    
    row4_y = base_y + 3 * row_spacing
    row4_x_start = row2_x_start
    for i in range(5):
        centers[idx] = [row4_x_start + i * spacing * 2, row4_y]
        idx += 1
    
    row5_y = base_y + 4 * row_spacing
    row5_x_start = row1_x_start
    for i in range(5):
        centers[idx] = [row5_x_start + i * spacing * 2, row5_y]
        idx += 1
    
    # Differential evolution: globally optimize radii for fixed positions
    radii = differential_evolution_optimize(centers)
    
    # Local SLSQP refinement for final polishing
    centers, radii = slsqp_refine(centers, radii)
    
    return centers, radii, np.sum(radii)


def differential_evolution_optimize(centers, maxiter=800):
    """
    Use differential evolution to find optimal radii for fixed center positions.
    
    For each candidate solution (26 radii values), check validity by computing 
    minimum distances; if valid, return negative sum_radii as fitness.
    Invalid solutions receive large penalty to guide search toward feasible region.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        maxiter: maximum number of differential evolution iterations
    
    Returns:
        np.array of shape (n) with optimized radii
    """
    n = centers.shape[0]
    
    def objective(radii):
        """
        Objective function for differential evolution.
        Returns negative sum of radii for valid packings, or large penalty for invalid.
        """
        # Check boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if r < 0 or r > 0.15:
                return 1e10
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return 1e10
        
        # Check pairwise distance constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(centers[i, 0] - centers[j, 0], 
                               centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > dist + 1e-10:
                    return 1e10
        
        # Valid solution - return negative sum (to maximize)
        return -np.sum(radii)
    
    # Bounds for each radius
    bounds = [(0, 0.15)] * n
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            seed=42,
            popsize=20,
            mutation=(0.5, 1.5),
            recombination=0.7,
            tol=1e-10,
            disp=False
        )
        return result.x
    except Exception:
        # Fallback to initial radii computation
        return compute_max_radii(centers)


# Voronoi optimization removed - differential evolution provides better global search


def slsqp_refine(centers, initial_radii=None):
    """
    SLSQP refinement of differential evolution-optimized positions.
    
    Optimizes both circle positions and radii to maximize sum of radii.
    Uses gradient-based optimization for fine-tuning after global search.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        initial_radii: np.array of shape (n) with initial radii values
    
    Returns:
        Tuple of (centers, radii)
    """
    n = centers.shape[0]
    
    # Use provided radii or compute initial
    if initial_radii is None:
        initial_radii = compute_max_radii(centers)
    
    # Prepare initial guess for optimizer
    initial_params = np.concatenate([centers.flatten(), initial_radii])
    
    # Bounds for each variable
    bounds = [(0, 1)] * (2 * n)  # coordinates
    bounds.extend([(0, 0.5)] * n)  # radii
    
    # Objective function: minimize negative sum of radii
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    # Constraint function for SLSQP
    def constraint_func(params):
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            constraints.append(x - r)
            constraints.append(1 - x - r)
            constraints.append(y - r)
            constraints.append(1 - y - r)
        
        # Pairwise distance constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(centers[i, 0] - centers[j, 0], 
                               centers[i, 1] - centers[j, 1])
                constraints.append(dist - radii[i] - radii[j])
        
        return np.array(constraints)
    
    # Run SLSQP for local refinement
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        centers = optimized_params[:2*n].reshape(n, 2)
        radii = optimized_params[2*n:]
        radii = np.maximum(radii, 0)
        
    except Exception:
        # Fallback to initial values
        pass
    
    return centers, radii


def compute_max_radii(centers):
    """
    Compute radii using iterative constraint relaxation.
    
    Each circle's radius is limited by:
    - Distance to square boundaries
    - Half distance to any other circle's center
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
    
    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with distance to borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iterative refinement with relaxation (optimized with early stopping)
    prev_sum = np.sum(radii)
    for _ in range(50):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(centers[i, 0] - centers[j, 0], 
                               centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > dist:
                    avg = dist / 2
                    radii[i] = min(radii[i], avg)
                    radii[j] = min(radii[j], avg)
        
        # Early stopping if converged
        if abs(np.sum(radii) - prev_sum) < 1e-12:
            break
        prev_sum = np.sum(radii)
    
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

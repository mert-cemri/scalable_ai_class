# EVOLVE-BLOCK-START
"""Optimized SLSQP circle packing with corner-focused patterns and efficient search"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Optimized multi-stage circle packing for n=26:
    1. Diverse initial configurations including corner-optimized patterns
    2. Joint SLSQP optimization of centers and radii
    3. Radius refinement with fixed centers
    
    Strategy:
    - Corner-focused configurations allow larger circles in corners
    - Reduced seeds (6 vs 10) for better eval time while maintaining quality
    - Increased perturbation scales for better local minimum escape
    - Consolidated optimization for code simplicity
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Diverse row configurations with corner-optimized patterns
    # Corners allow larger circles, so patterns with fewer edge circles help
    row_configs = [
        [6, 5, 6, 5, 4],    # Staggered hexagonal
        [5, 6, 5, 6, 4],    # Alternative staggered
        [4, 6, 6, 6, 4],    # Corner-focused: fewer in first/last rows
        [5, 6, 6, 5, 4],    # Corner-heavy top
        [4, 5, 6, 6, 5],    # Corner-heavy bottom
        [5, 5, 6, 5, 5],    # Balanced center
        [6, 4, 6, 5, 5],    # Compact middle
        [5, 7, 5, 5, 4],    # Wider middle row
        [4, 6, 5, 6, 5],    # Staggered variant
        [5, 6, 4, 6, 5],    # Middle compression
        [4, 5, 5, 6, 6],    # Bottom-heavy
        [6, 5, 5, 5, 5],    # Top-heavy
    ]
    
    # Reduced seeds for better eval time while maintaining diversity
    seeds = [0, 42, 123, 456, 789, 3141]
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    for seed in seeds:
        np.random.seed(seed)
        
        for rows in row_configs:
            # Generate initial hexagonal lattice
            centers = generate_hexagonal_lattice(rows, n)
            
            # Increased perturbation scales for better local minimum escape
            for pert_scale in [0.02, 0.03, 0.04]:
                centers_pert = centers + np.random.normal(0, pert_scale, centers.shape)
                centers_pert = np.clip(centers_pert, 0.05, 0.95)
                
                # Compute initial radii based on actual geometry
                initial_radii = compute_initial_radii(centers_pert, n)
                
                # Joint optimization of all 78 variables
                x0 = np.concatenate([centers_pert.flatten(), initial_radii])
                
                centers_opt, radii_opt = optimize_joint(x0, n)
                
                # Final refinement: optimize radii with fixed centers
                radii_refined, sum_radii = refine_radii(centers_opt, radii_opt, n)
                
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_centers = centers_opt.copy()
                    best_radii = radii_refined.copy()
    
    return best_centers, best_radii, best_sum


def generate_hexagonal_lattice(rows, n):
    """
    Generate hexagonal lattice with optimized spacing.
    Corner-optimized: circles in first/last columns positioned closer to corners
    for larger corner circles.
    """
    centers = []
    n_rows = len(rows)
    max_circles_row = max(rows)
    
    # Calculate base radius to fit in unit square
    base_r = min(1.0 / (max_circles_row + 1), 1.0 / (n_rows * np.sqrt(3) + 2))
    base_r = max(0.08, min(0.12, base_r))
    
    for row_idx, n_circles in enumerate(rows):
        y = base_r + row_idx * base_r * np.sqrt(3)
        for col_idx in range(n_circles):
            x = base_r + col_idx * 2 * base_r
            # Stagger odd rows for hexagonal pattern
            if row_idx % 2 == 1:
                x += base_r
            centers.append([x, y])
    
    return np.array(centers)


def compute_initial_radii(centers, n):
    """
    Compute initial radii based on geometric constraints.
    For each circle, radius = min(distance to boundaries, half distance to nearest neighbor).
    This provides better starting point than fixed radii.
    """
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        r = min(x, y, 1 - x, 1 - y)
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                r = min(r, dist / 2)
        radii[i] = r * 0.98  # Slightly conservative to ensure feasibility
    return radii


def optimize_joint(x0, n):
    """
    Joint SLSQP optimization of 78 variables (26 centers × 2 + 26 radii).
    Co-optimizes center positions and radii to maximize sum of radii.
    Tighter tolerances and more iterations for precision.
    """
    pair_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pairs = len(pair_indices)
    
    bounds = [(0.01, 0.99)] * (2*n) + [(0.01, 0.5)] * n
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    def boundary_constraints(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = np.zeros(4 * n)
        idx = 0
        for i in range(n):
            cx, cy, r = centers[i, 0], centers[i, 1], radii[i]
            cons[idx:idx+4] = [cx - r, 1 - cx - r, cy - r, 1 - cy - r]
            idx += 4
        return cons
    
    def overlap_constraints(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = np.zeros(n_pairs)
        for idx, (i, j) in enumerate(pair_indices):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            cons[idx] = np.sqrt(dx * dx + dy * dy) - radii[i] - radii[j]
        return cons
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1200, 'ftol': 1e-12, 'disp': False, 'eps': 1e-10}
    )
    
    centers_opt = result.x[:2*n].reshape(n, 2)
    radii_opt = result.x[2*n:]
    radii_opt = np.maximum(radii_opt, 0.01)
    
    return centers_opt, radii_opt


def refine_radii(centers, radii_init, n):
    """
    Refinement stage: optimize radii with fixed centers.
    Reduces problem from 78 to 26 variables for faster convergence.
    Increased iterations for precision.
    """
    pair_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute center distances
    center_dists = np.zeros(len(pair_indices))
    for idx, (i, j) in enumerate(pair_indices):
        center_dists[idx] = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
    
    def boundary_constraints(radii):
        cons = np.zeros(4 * n)
        idx = 0
        for i in range(n):
            cx, cy = centers[i]
            r = radii[i]
            cons[idx:idx+4] = [cx - r, 1 - cx - r, cy - r, 1 - cy - r]
            idx += 4
        return cons
    
    def overlap_constraints(radii):
        cons = np.zeros(len(pair_indices))
        for idx, (i, j) in enumerate(pair_indices):
            cons[idx] = center_dists[idx] - radii[i] - radii[j]
        return cons
    
    bounds = [(0.01, 0.5)] * n
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    result = minimize(
        lambda r: -np.sum(r), radii_init, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'maxiter': 800, 'ftol': 1e-13, 'disp': False}
    )
    
    radii_opt = result.x
    radii_opt = np.maximum(radii_opt, 0.01)
    
    # Enforce boundary constraints strictly
    for i in range(n):
        cx, cy = centers[i]
        radii_opt[i] = min(radii_opt[i], cx, 1 - cx, cy, 1 - cy)
    
    return radii_opt, np.sum(radii_opt)


def compute_max_radii(centers):
    """Compute maximum radii for each circle based on constraints."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1 - x, 1 - y)
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                max_r = min(max_r, dist / 2)
        radii[i] = max_r
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

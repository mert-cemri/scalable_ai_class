# EVOLVE-BLOCK-START
"""Joint SLSQP optimization of all 78 variables (centers and radii) simultaneously"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint SLSQP optimization of all 78 variables (centers and radii) simultaneously.
    
    This breakthrough approach removes the artificial constraint of two-stage
    decomposition, allowing centers and radii to co-optimize. The optimizer
    explores the full 78-dimensional solution space where center positions
    and radii are jointly adjusted to maximize sum of radii while satisfying
    all constraints (104 boundary + 325 overlap = 429 total constraints).
    
    Uses multiple initial configurations with random perturbations to escape
    local minima and explore diverse regions of the solution space.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Try multiple row configurations with different perturbations
    row_configs = [
        [6, 5, 6, 5, 4],  # Original staggered hexagonal
        [5, 6, 5, 6, 4],  # Alternative staggered
        [6, 6, 5, 5, 4],  # More circles in top rows
        [4, 6, 6, 5, 5],  # More circles in bottom rows
        [5, 5, 6, 5, 5],  # Balanced configuration
        [6, 5, 5, 5, 5],  # Another variant
    ]
    
    # Multiple random seeds for perturbation diversity
    seeds = [0, 42, 123, 456, 789, 1000]
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    for seed in seeds:
        np.random.seed(seed)
        
        for rows in row_configs:
            # Generate initial hexagonal lattice
            centers = generate_hexagonal_lattice(rows, n)
            
            # Add small random perturbation to escape local minima
            centers_pert = centers + np.random.normal(0, 0.02, centers.shape)
            centers_pert = np.clip(centers_pert, 0.05, 0.95)
            
            # Initial radii estimate (slightly conservative)
            initial_radii = np.ones(n) * 0.095
            
            # Joint optimization of all 78 variables
            x0 = np.concatenate([centers_pert.flatten(), initial_radii])
            
            centers_opt, radii_opt, sum_radii = optimize_joint(x0, n)
            
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_centers = centers_opt.copy()
                best_radii = radii_opt.copy()
    
    return best_centers, best_radii, best_sum


def generate_hexagonal_lattice(rows, n):
    """
    Generate hexagonal lattice configuration with optimized spacing.
    Staggered rows with sqrt(3)/2 vertical spacing for hexagonal packing.
    
    Args:
        rows: List of circle counts per row
        n: Total number of circles (should match sum(rows))
    
    Returns:
        np.array of shape (n, 2) with center coordinates
    """
    centers = []
    
    # Calculate optimal base radius to fit in unit square
    n_rows = len(rows)
    max_circles_row = max(rows)
    
    # Estimate spacing to fit in unit square
    base_r = min(1.0 / (max_circles_row + 1), 
                 1.0 / (n_rows * np.sqrt(3) + 2))
    base_r = max(0.08, min(0.12, base_r))  # Clamp to reasonable range
    
    for row_idx, n_circles in enumerate(rows):
        # Vertical position with hexagonal spacing
        y = base_r + row_idx * base_r * np.sqrt(3)
        
        # Horizontal position
        for col_idx in range(n_circles):
            x = base_r + col_idx * 2 * base_r
            
            # Stagger odd rows for hexagonal pattern
            if row_idx % 2 == 1:
                x += base_r
            
            centers.append([x, y])
    
    return np.array(centers)


def optimize_joint(x0, n):
    """
    Joint SLSQP optimization of all 78 variables (26 centers × 2 + 26 radii).
    
    This is the breakthrough approach that allows centers and radii to co-evolve
    in a single optimization, removing the artificial constraint of two-stage
    decomposition. The optimizer can now find better local optima where center
    positions and radii are jointly optimized.
    
    Constraints:
    - Boundary: 4 per circle (left, right, bottom, top) = 104 constraints
    - Overlap: C(26,2) = 325 constraints
    - Total: 429 inequality constraints
    
    Args:
        x0: Initial values for all 78 variables [x1,y1,x2,y2,...,r1,r2,...]
        n: Number of circles (should be 26)
    
    Returns:
        Tuple of (optimized_centers, optimized_radii, sum_radii)
    """
    # Precompute pair indices for overlap constraints
    pair_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pairs = len(pair_indices)
    
    # Bounds: centers in [0.01, 0.99], radii in [0.01, 0.5]
    bounds = [(0.01, 0.99)] * (2*n) + [(0.01, 0.5)] * n
    
    # Objective: maximize sum of radii (negate for minimization)
    def objective(x):
        return -np.sum(x[2*n:])
    
    # Boundary constraints: each circle must stay in unit square
    # cx - r >= 0, 1 - cx - r >= 0, cy - r >= 0, 1 - cy - r >= 0
    def boundary_constraints(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = np.zeros(4 * n)
        idx = 0
        for i in range(n):
            cx, cy = centers[i]
            r = radii[i]
            cons[idx] = cx - r          # left
            cons[idx + 1] = 1 - cx - r  # right
            cons[idx + 2] = cy - r      # bottom
            cons[idx + 3] = 1 - cy - r  # top
            idx += 4
        return cons
    
    # Overlap constraints: distance >= r_i + r_j
    def overlap_constraints(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = np.zeros(n_pairs)
        for idx, (i, j) in enumerate(pair_indices):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            cons[idx] = dist - radii[i] - radii[j]
        return cons
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    # Run SLSQP optimization with tight tolerances
    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-8}
    )
    
    # Extract results
    centers_opt = result.x[:2*n].reshape(n, 2)
    radii_opt = result.x[2*n:]
    
    # Ensure positive radii
    radii_opt = np.maximum(radii_opt, 0.01)
    
    # Post-process: fix any constraint violations
    for i in range(n):
        cx, cy = centers_opt[i]
        r = radii_opt[i]
        if cx - r < 0:
            radii_opt[i] = cx
        if cx + r > 1:
            radii_opt[i] = 1 - cx
        if cy - r < 0:
            radii_opt[i] = cy
        if cy + r > 1:
            radii_opt[i] = 1 - cy
    
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii


def compute_max_radii(centers):
    """
    Compute maximum radii for each circle based on constraints.
    Used for visualization and validation.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
    
    Returns:
        np.array of shape (n) with maximum valid radius for each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    for i in range(n):
        x, y = centers[i]
        # Distance to square boundaries
        max_r = min(x, y, 1 - x, 1 - y)
        
        # Distance to other circle centers
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

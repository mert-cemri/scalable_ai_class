# EVOLVE-BLOCK-START
"""Direct constructor for n=26 circles maximizing sum of radii in unit square"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Direct constructor using optimized hexagonal pattern with SLSQP refinement.
    
    Strategy:
    1. Initialize with carefully tuned hexagonal grid (5+5+6+5+5 pattern)
    2. Use vectorized constraint computation for efficiency
    3. Apply SLSQP optimization with aggressive multi-start strategy
    4. Use tighter tolerances and more iterations for better convergence
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    best_centers = None
    best_radii = None
    best_sum = 0
    
    # Extended parameter search with finer granularity
    param_sets = [
        (0.1030, 0.0472, 0.0222),
        (0.1032, 0.0475, 0.0225),
        (0.1035, 0.0480, 0.0230),
        (0.1038, 0.0485, 0.0235),
        (0.1040, 0.0490, 0.0240),
        (0.1028, 0.0468, 0.0218),
        (0.1025, 0.0462, 0.0212),
        (0.1033, 0.0477, 0.0227),
        (0.1036, 0.0482, 0.0232),
        (0.1029, 0.0469, 0.0219),
    ]
    
    for spacing, row1_x_start, row3_x_start in param_sets:
        centers = create_hexagonal_grid(n, spacing, row1_x_start, row3_x_start)
        radii = compute_radii_vectorized(centers)
        
        # Apply SLSQP refinement with multiple perturbations
        centers, radii = refine_with_slsqp(centers, radii, n)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum


def create_hexagonal_grid(n, spacing, row1_x_start, row3_x_start):
    """
    Create hexagonal grid pattern with 5+5+6+5+5 circle distribution.
    
    Args:
        n: Number of circles (26)
        spacing: Horizontal spacing parameter
        row1_x_start: X offset for rows 1, 5
        row3_x_start: X offset for row 3 (center row with 6 circles)
    
    Returns:
        np.array of shape (n, 2) with circle centers
    """
    row_spacing = spacing * np.sqrt(3)
    total_height = 4 * row_spacing
    base_y = (1 - total_height) / 2
    
    centers = np.zeros((n, 2))
    idx = 0
    
    # Row 1: 5 circles
    row1_y = base_y
    for i in range(5):
        centers[idx] = [row1_x_start + i * spacing * 2, row1_y]
        idx += 1
    
    # Row 2: 5 circles (staggered)
    row2_y = base_y + row_spacing
    row2_x_start = row1_x_start + spacing
    for i in range(5):
        centers[idx] = [row2_x_start + i * spacing * 2, row2_y]
        idx += 1
    
    # Row 3: 6 circles (center)
    row3_y = base_y + 2 * row_spacing
    for i in range(6):
        centers[idx] = [row3_x_start + i * spacing * 2, row3_y]
        idx += 1
    
    # Row 4: 5 circles (staggered)
    row4_y = base_y + 3 * row_spacing
    for i in range(5):
        centers[idx] = [row2_x_start + i * spacing * 2, row4_y]
        idx += 1
    
    # Row 5: 5 circles
    row5_y = base_y + 4 * row_spacing
    for i in range(5):
        centers[idx] = [row1_x_start + i * spacing * 2, row5_y]
        idx += 1
    
    return centers


def compute_radii_vectorized(centers):
    """
    Compute radii using vectorized operations for efficiency.
    
    Each radius is limited by boundary distances and half-distance to neighbors.
    Uses iterative relaxation until convergence.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
    
    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.minimum(
        np.minimum(centers[:, 0], centers[:, 1]),
        np.minimum(1 - centers[:, 0], 1 - centers[:, 1])
    )
    
    # Pre-compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    
    # Iterative relaxation with early stopping
    for _ in range(100):
        prev_sum = np.sum(radii)
        
        # Vectorized constraint update
        for i in range(n):
            for j in range(i + 1, n):
                dist = distances[i, j]
                avg = dist / 2
                if radii[i] + radii[j] > dist:
                    radii[i] = min(radii[i], avg)
                    radii[j] = min(radii[j], avg)
        
        if abs(np.sum(radii) - prev_sum) < 1e-15:
            break
    
    return radii


def refine_with_slsqp(centers, radii, n, max_restarts=5):
    """
    SLSQP refinement with multi-restart strategy to escape local optima.
    
    Args:
        centers: np.array of shape (n, 2) with initial centers
        radii: np.array of shape (n) with initial radii
        n: Number of circles
        max_restarts: Number of perturbation restarts
    
    Returns:
        Tuple of (optimized_centers, optimized_radii)
    """
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    for seed in range(max_restarts):
        np.random.seed(seed)
        
        # Perturb centers to explore different local optima
        perturbation_scale = 0.004
        perturbed = centers + np.random.uniform(
            -perturbation_scale, perturbation_scale, (n, 2)
        )
        perturbed = np.clip(perturbed, 0.02, 0.98)
        
        # Compute radii for perturbed configuration
        perturbed_radii = compute_radii_vectorized(perturbed)
        
        # Prepare optimization parameters
        params = np.concatenate([perturbed.flatten(), perturbed_radii])
        
        # Bounds: centers in [0,1], radii in [0, 0.5]
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        # SLSQP optimization
        try:
            result = minimize(
                lambda p: -np.sum(p[2*n:]),
                params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda p: constraint_check(p, n)},
                options={'maxiter': 1200, 'ftol': 1e-15, 'eps': 1e-10}
            )
            
            opt_params = result.x
            opt_centers = opt_params[:2*n].reshape(n, 2)
            opt_radii = np.maximum(opt_params[2*n:], 0)
            
            opt_sum = np.sum(opt_radii)
            if opt_sum > best_sum:
                best_sum = opt_sum
                best_centers = opt_centers.copy()
                best_radii = opt_radii.copy()
        
        except Exception:
            continue
    
    return best_centers, best_radii


def constraint_check(params, n):
    """
    Compute all constraint values for SLSQP (must be >= 0).
    
    Constraints:
    1. Circle must be inside square: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    2. Circles must not overlap: dist(i,j) - r[i] - r[j] >= 0
    
    Args:
        params: Concatenated [centers, radii] array
        n: Number of circles
    
    Returns:
        np.array of constraint values (all must be >= 0)
    """
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        constraints.extend([x - r, 1 - x - r, y - r, 1 - y - r])
    
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], 
                           centers[i, 1] - centers[j, 1])
            constraints.append(dist - radii[i] - radii[j])
    
    return np.array(constraints)


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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)

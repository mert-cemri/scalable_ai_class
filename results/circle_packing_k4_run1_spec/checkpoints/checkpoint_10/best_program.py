"""Joint optimization of circle positions and radii using SLSQP"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Jointly optimize 26 circle positions and radii using SLSQP with efficiency improvements.
    
    Optimizes all 78 variables (26 centers + 26 radii) simultaneously
    to maximize sum of radii subject to:
    - Boundary constraints: circles stay within unit square
    - Non-overlap constraints: distance(i,j) >= r_i + r_j + epsilon
    
    Uses multiple initialization patterns and optimized constraint evaluation.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    epsilon = 1e-8  # Small epsilon for numerical stability
    
    # Pre-compute constraint indices for efficiency
    overlap_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Bounds: x in [0,1], y in [0,1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    
    # Objective: maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])
    
    # Run optimization with multiple restarts and initialization patterns
    best_result = None
    best_sum_radii = -np.inf
    
    # Try different initialization patterns
    init_patterns = [
        initialize_hexagonal_grid,
        initialize_hexagonal_grid_tighter,
        initialize_hexagonal_grid_staggered,
    ]
    
    for pattern_idx, init_func in enumerate(init_patterns):
        for seed in range(2):
            np.random.seed(seed + pattern_idx * 100)
            
            centers_init = init_func(n)
            x0 = np.zeros(3 * n)
            for i in range(n):
                x0[3*i] = centers_init[i, 0]
                x0[3*i+1] = centers_init[i, 1]
                x0[3*i+2] = 0.065  # Slightly larger initial radius guess
            
            # Add small random perturbation to initial positions
            x0_trial = x0.copy()
            x0_trial[:2*n] += np.random.uniform(-0.015, 0.015, 2*n)
            
            # Build constraints (only once per trial to save overhead)
            constraints = []
            
            # Boundary constraints for each circle (4 per circle)
            for i in range(n):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]})
            
            # Non-overlap constraints with epsilon for numerical stability
            for i, j in overlap_pairs:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - x[3*i+2] - x[3*j+2] - epsilon
                })
            
            try:
                result = minimize(
                    objective,
                    x0_trial,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1200, 'ftol': 1e-9, 'eps': 1e-8, 'disp': False}
                )
                
                if result.success or best_result is None:
                    current_sum = -result.fun
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_result = result
            except Exception:
                continue
    
    if best_result is None:
        # Fallback to original hexagonal approach
        centers = initialize_hexagonal_grid(n)
        radii = compute_max_radii(centers)
        sum_radii = np.sum(radii)
    else:
        # Extract final solution
        x_opt = best_result.x
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = x_opt[3*i]
            centers[i, 1] = x_opt[3*i+1]
            radii[i] = max(1e-6, x_opt[3*i+2])
        sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii


def initialize_hexagonal_grid_tighter(n):
    """Initialize with tighter hexagonal grid for denser packing"""
    centers = []
    spacing = 0.185  # Slightly tighter spacing
    row_height = spacing * np.sqrt(3) / 2
    offset = spacing / 2
    
    row_configs = [
        (5, 0.052, 0.052),
        (5, 0.052 + offset, 0.052 + row_height),
        (5, 0.052, 0.052 + 2*row_height),
        (5, 0.052 + offset, 0.052 + 3*row_height),
        (6, 0.052, 0.052 + 4*row_height),
    ]
    
    idx = 0
    for num_circles, x_start, y_start in row_configs:
        for i in range(num_circles):
            x = x_start + i * spacing
            centers.append([x, y_start])
            idx += 1
    
    return np.array(centers[:n])


def initialize_hexagonal_grid_staggered(n):
    """Initialize with staggered pattern starting from corners"""
    centers = []
    spacing = 0.188
    row_height = spacing * np.sqrt(3) / 2
    offset = spacing / 2
    
    # Start from corners for better edge utilization
    row_configs = [
        (5, 0.048, 0.048),
        (5, 0.048 + offset, 0.048 + row_height),
        (5, 0.048, 0.048 + 2*row_height),
        (5, 0.048 + offset, 0.048 + 3*row_height),
        (6, 0.048, 0.048 + 4*row_height),
    ]
    
    idx = 0
    for num_circles, x_start, y_start in row_configs:
        for i in range(num_circles):
            x = x_start + i * spacing
            centers.append([x, y_start])
            idx += 1
    
    return np.array(centers[:n])


def initialize_hexagonal_grid(n):
    """Initialize circles in hexagonal grid pattern for good starting point"""
    centers = []
    spacing = 0.19
    row_height = spacing * np.sqrt(3) / 2
    offset = spacing / 2
    
    row_configs = [
        (5, 0.05, 0.05),
        (5, 0.05 + offset, 0.05 + row_height),
        (5, 0.05, 0.05 + 2*row_height),
        (5, 0.05 + offset, 0.05 + 3*row_height),
        (6, 0.05, 0.05 + 4*row_height),
    ]
    
    idx = 0
    for num_circles, x_start, y_start in row_configs:
        for i in range(num_circles):
            x = x_start + i * spacing
            centers.append([x, y_start])
            idx += 1
    
    return np.array(centers[:n])


def compute_max_radii(centers):
    """
    Compute maximum valid radii using iterative refinement.
    Fallback method if optimization fails.

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
    
    # Iteratively adjust radii to avoid overlaps
    for _ in range(50):
        max_violation = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist and dist > 0:
                    excess = radii[i] + radii[j] - dist
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    max_violation = max(max_violation, excess)
        
        if max_violation < 1e-6:
            break
    
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

# EVOLVE-BLOCK-START
"""Direct circle position optimization using SLSQP for n=26 circles"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Direct circle position optimization using SLSQP for n=26 circles.
    
    Strategy:
    1. Try multiple diverse initial configurations (hexagonal, random, grid patterns)
    2. Optimize all 52 coordinates (x,y for each circle) simultaneously using SLSQP
    3. Compute radii based on min distance to walls and other circles
    4. Use warm-starting: continue optimizing from best found solution
    5. Increase iterations for better convergence
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Try diverse initial configurations
    row_patterns = [
        [5, 5, 5, 5, 4, 2], [6, 5, 5, 5, 5], [5, 5, 5, 4, 5, 2],
        [5, 6, 5, 5, 5], [4, 6, 5, 5, 6], [5, 5, 4, 6, 5, 1],
        [6, 4, 6, 6, 4], [5, 5, 5, 5, 3, 3], [7, 5, 5, 5, 4]
    ]
    
    # First pass: try different patterns
    for seed in range(12):
        np.random.seed(seed)
        
        # Try different row patterns
        for pattern_idx in range(min(3, len(row_patterns))):
            row_counts = row_patterns[(pattern_idx + seed) % len(row_patterns)]
            if sum(row_counts) != n:
                continue
                
            centers = generate_initial_config(n, row_counts)
            
            # Flatten centers for optimization
            x0 = centers.flatten()
            
            # Bounds: all coordinates must be in [0, 1]
            bounds = [(0.0, 1.0)] * (2 * n)
            
            # Optimize using SLSQP with more iterations
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            # Accept result if converged or enough iterations completed
            if result.success or result.nit >= 200:
                optimized_centers = result.x.reshape(-1, 2)
                radii = compute_radii(optimized_centers)
                current_sum = np.sum(radii)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = optimized_centers.copy()
                    best_radii = radii.copy()
    
    # Second pass: refine best solution with warm-start
    if best_centers is not None:
        for seed in range(6):
            np.random.seed(seed + 100)
            x0 = best_centers.flatten() + np.random.normal(0, 0.005, 2*n)
            x0 = np.clip(x0, 0.01, 0.99)
            
            bounds = [(0.0, 1.0)] * (2 * n)
            
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success or result.nit >= 150:
                optimized_centers = result.x.reshape(-1, 2)
                radii = compute_radii(optimized_centers)
                current_sum = np.sum(radii)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = optimized_centers.copy()
                    best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum


def objective_function(params):
    """
    Objective function: negative sum of radii (to minimize).
    
    Args:
        params: flattened array of 2*n coordinates (x0, y0, x1, y1, ..., x25, y25)
    
    Returns:
        Negative sum of radii (negative because we minimize)
    """
    n = 26
    centers = params.reshape(-1, 2)
    radii = compute_radii(centers)
    return -np.sum(radii)


def compute_radii(centers):
    """
    Compute radii for each circle based on min distance to walls and other circles.
    Uses vectorized operations for efficiency.
    
    Args:
        centers: array of shape (n, 2) with (x, y) coordinates
    
    Returns:
        array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Precompute distance matrix for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    
    for i in range(n):
        x, y = centers[i]
        # Distance to walls
        r = min(x, 1 - x, y, 1 - y)
        
        # Distance to other circles (vectorized)
        other_dists = dist_matrix[i, :]
        other_dists[i] = np.inf  # Ignore self-distance
        r = min(r, np.min(other_dists) / 2)
        
        radii[i] = r
    
    return radii


def generate_initial_config(n, row_counts=None):
    """
    Generate initial hexagonal configuration with small random perturbations.
    
    Args:
        n: number of circles
        row_counts: optional list of circles per row
    
    Returns:
        array of shape (n, 2) with initial (x, y) coordinates
    """
    # Use provided row counts or default
    if row_counts is None:
        row_counts = [5, 5, 5, 5, 4, 2]
    
    centers = np.zeros((n, 2))
    idx = 0
    
    # Calculate optimal spacing based on row counts
    num_rows = len(row_counts)
    max_cols = max(row_counts)
    
    # Use spacing that fits well in unit square
    base_spacing = 0.185
    row_height = 0.16
    
    for row_idx, count in enumerate(row_counts):
        # Alternate offset for hexagonal packing
        x_offset = base_spacing / 2 if row_idx % 2 == 1 else 0
        y = base_spacing / 2 + row_idx * row_height
        for col in range(count):
            if idx < n:
                x = base_spacing / 2 + col * base_spacing + x_offset
                centers[idx] = [x, y]
                idx += 1
    
    # Add small random perturbations to escape local minima
    centers += np.random.normal(0, 0.015, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    return centers


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

# EVOLVE-BLOCK-START
"""Explicit radius optimization with nonlinear constraints for n=26 circles"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Circle packing optimization with explicit radius variables.
    
    Strategy:
    1. Use 78 optimization variables: [x0,y0,r0, x1,y1,r1, ..., x25,y25,r25]
    2. Add explicit nonlinear constraints for wall and circle-circle boundaries
    3. This creates smoother optimization landscape (radii are not piecewise functions)
    4. Try diverse initial configurations with hexagonal and random patterns
    5. Use warm-starting to refine best solutions
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Diverse row patterns for hexagonal configurations
    row_patterns = [
        [5, 5, 5, 5, 4, 2], [6, 5, 5, 5, 5], [5, 5, 5, 4, 5, 2],
        [5, 6, 5, 5, 5], [4, 6, 5, 5, 6], [5, 5, 4, 6, 5, 1],
        [6, 4, 6, 6, 4], [5, 5, 5, 5, 3, 3], [7, 5, 5, 5, 4]
    ]
    
    # First pass: try different patterns with explicit radius optimization
    for seed in range(12):
        np.random.seed(seed)
        
        # Try different row patterns
        for pattern_idx in range(min(2, len(row_patterns))):
            row_counts = row_patterns[(pattern_idx + seed) % len(row_patterns)]
            if sum(row_counts) != n:
                continue
            
            # Try multiple spacing values for hexagonal pattern
            for spacing in [0.192, 0.195, 0.188]:
                # Generate initial configuration
                centers = generate_initial_config(n, row_counts, base_spacing=spacing)
                initial_radii = np.full(n, 0.15)  # Start with reasonable uniform radii
                
                # 78 variables: [x0,y0,r0, x1,y1,r1, ..., x25,y25,r25]
                x0 = np.concatenate([centers[:, 0], centers[:, 1], initial_radii])
                
                # Bounds: x,y in [0.01, 0.99], r in [0.01, 0.4]
                bounds = [(0.01, 0.99)] * n + [(0.01, 0.99)] * n + [(0.01, 0.4)] * n
                
                # Optimize with explicit radius variables
                result = minimize(
                    objective_explicit_radius,
                    x0,
                    args=(n,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': lambda p, nn=n: constraint_explicit_radius(p, nn)},
                    options={'maxiter': 400, 'ftol': 1e-10, 'disp': False}
                )
                
                if result.success or result.nit >= 80:
                    optimized = result.x
                    opt_centers = np.column_stack([optimized[:n], optimized[n:2*n]])
                    opt_radii = optimized[2*n:]
                    current_sum = np.sum(opt_radii)
                    
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = opt_centers.copy()
                        best_radii = opt_radii.copy()
        
        # Add random configurations for diversity
        if seed % 4 == 0:
            centers = np.random.uniform(0.15, 0.85, (n, 2))
            initial_radii = np.full(n, 0.12)
            x0 = np.concatenate([centers[:, 0], centers[:, 1], initial_radii])
            bounds = [(0.01, 0.99)] * n + [(0.01, 0.99)] * n + [(0.01, 0.4)] * n
            
            result = minimize(
                objective_explicit_radius,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda p, nn=n: constraint_explicit_radius(p, nn)},
                options={'maxiter': 400, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or result.nit >= 80:
                optimized = result.x
                opt_centers = np.column_stack([optimized[:n], optimized[n:2*n]])
                opt_radii = optimized[2*n:]
                current_sum = np.sum(opt_radii)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = opt_centers.copy()
                    best_radii = opt_radii.copy()
    
    # Second pass: refine best solution with adaptive perturbations
    if best_centers is not None:
        for seed in range(12):
            np.random.seed(seed + 200)
            # Use varying perturbation scales for diversity
            perturbation_scale = 0.005 + 0.01 * seed / 12
            perturbation = np.random.normal(0, perturbation_scale, 3 * n)
            x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_radii]) + perturbation
            x0 = np.clip(x0, 0.01, 0.99)
            x0[2*n:] = np.clip(x0[2*n:], 0.01, 0.4)  # Ensure radii are positive
            
            bounds = [(0.01, 0.99)] * n + [(0.01, 0.99)] * n + [(0.01, 0.4)] * n
            
            result = minimize(
                objective_explicit_radius,
                x0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda p, nn=n: constraint_explicit_radius(p, nn)},
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or result.nit >= 100:
                optimized = result.x
                opt_centers = np.column_stack([optimized[:n], optimized[n:2*n]])
                opt_radii = optimized[2*n:]
                current_sum = np.sum(opt_radii)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = opt_centers.copy()
                    best_radii = opt_radii.copy()
    
    return best_centers, best_radii, best_sum


def objective_explicit_radius(params, n):
    """
    Objective function: negative sum of radii (to minimize).
    Radii are explicit optimization variables, not computed from geometry.
    
    Args:
        params: array of 3*n values [x0,y0,r0, x1,y1,r1, ..., xn-1,yn-1,rn-1]
        n: number of circles
    
    Returns:
        Negative sum of radii
    """
    radii = params[2*n:]
    return -np.sum(radii)


def constraint_explicit_radius(params, n):
    """
    Nonlinear inequality constraints (all must be >= 0):
    1. Wall constraints: x_i - r_i >= 0, 1-x_i - r_i >= 0, y_i - r_i >= 0, 1-y_i - r_i >= 0
    2. Circle-circle constraints: dist(i,j) - r_i - r_j >= 0
    
    Uses vectorized computation for efficiency.
    
    Args:
        params: array of 3*n values [x0,y0,r0, x1,y1,r1, ..., xn-1,yn-1,rn-1]
        n: number of circles
    
    Returns:
        Array of constraint values (all should be >= 0)
    """
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    constraints = []
    
    # Wall constraints (4 per circle)
    constraints.extend(x - r)  # x_i - r_i >= 0
    constraints.extend(1 - x - r)  # 1-x_i - r_i >= 0
    constraints.extend(y - r)  # y_i - r_i >= 0
    constraints.extend(1 - y - r)  # 1-y_i - r_i >= 0
    
    # Circle-circle constraints (n*(n-1)/2 pairs) - vectorized
    # Compute all pairwise distances using broadcasting
    centers = np.column_stack([x, y])
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Extract upper triangular (i < j) distances
    upper_tri_idx = np.triu_indices(n, k=1)
    pairwise_dists = dist_matrix[upper_tri_idx]
    
    # Compute dist(i,j) - r_i - r_j for all pairs
    r_sum = r[upper_tri_idx[0]] + r[upper_tri_idx[1]]
    constraints.extend(pairwise_dists - r_sum)
    
    return np.array(constraints)


def generate_initial_config(n, row_counts=None, base_spacing=0.192):
    """
    Generate initial hexagonal configuration with optimized spacing.
    
    Uses sqrt(3)/2 ratio for proper hexagonal packing density.
    Allows different spacing values to be tested for better optimization.
    
    Args:
        n: number of circles
        row_counts: optional list of circles per row
        base_spacing: spacing between adjacent circles (default 0.192)
    
    Returns:
        array of shape (n, 2) with initial (x, y) coordinates
    """
    if row_counts is None:
        row_counts = [5, 5, 5, 5, 4, 2]
    
    centers = np.zeros((n, 2))
    idx = 0
    
    row_height = np.sqrt(3) / 2 * base_spacing
    
    for row_idx, count in enumerate(row_counts):
        x_offset = base_spacing / 2 if row_idx % 2 == 1 else 0
        y = base_spacing / 2 + row_idx * row_height
        for col in range(count):
            if idx < n:
                x = base_spacing / 2 + col * base_spacing + x_offset
                centers[idx] = [x, y]
                idx += 1
    
    # Small random perturbations to escape local minima
    centers += np.random.normal(0, 0.012, centers.shape)
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

# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using scipy.optimize.SLSQP with explicit constraints"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint optimization using scipy.optimize.SLSQP with explicit constraint formulation.
    
    Approach:
    1. Initialize with hexagonal pattern for good starting configuration
    2. Use SLSQP with explicit inequality constraints (not penalties)
    3. Constraints: ||c_i - c_j|| >= r_i + r_j (non-overlap), x >= r, etc. (boundary)
    4. Objective: maximize sum of radii (negative for minimization)
    5. Use multiple restarts with different patterns to escape local minima
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    def objective(params):
        """Objective: maximize sum of radii (negative for minimization)"""
        radii = params[2*n:]
        return -np.sum(radii)
    
    def make_circle_circle_constraint(i, j):
        """Create constraint: ||c_i - c_j|| >= r_i + r_j (returns >= 0 when satisfied)"""
        def constraint(params):
            centers = params[:2*n].reshape(n, 2)
            radii = params[2*n:]
            d = np.linalg.norm(centers[i] - centers[j])
            return d - radii[i] - radii[j]
        return constraint
    
    def make_boundary_constraint(i, axis, side):
        """Create boundary constraint: c[i, axis] >= r[i] or 1 - c[i, axis] >= r[i]"""
        def constraint(params):
            centers = params[:2*n].reshape(n, 2)
            radii = params[2*n:]
            if side == 0:  # left/bottom: center >= radius
                return centers[i, axis] - radii[i]
            else:  # right/top: 1 - center >= radius
                return 1 - centers[i, axis] - radii[i]
        return constraint
    
    # Define bounds for optimization
    # Centers: [0, 1] for x and y
    # Radii: [0.001, 0.5] for each circle
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1)])  # x, y bounds
    for _ in range(n):
        bounds.append((0.001, 0.5))  # radius bounds
    
    best_sum = 0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    
    # Base hexagonal patterns with better spacing
    base_patterns = [
        # Pattern 1: 5-4-5-4-5-3 (hexagonal, staggered)
        [
            (0.095, [0.065, 0.235, 0.415, 0.595, 0.775]),
            (0.275, [0.165, 0.395, 0.625, 0.855]),
            (0.455, [0.065, 0.235, 0.415, 0.595, 0.775]),
            (0.635, [0.165, 0.395, 0.625, 0.855]),
            (0.815, [0.065, 0.235, 0.415, 0.595, 0.775]),
            (0.935, [0.285, 0.525, 0.765]),
        ],
        # Pattern 2: 5-5-5-5-4-2 (better top fill)
        [
            (0.08, [0.07, 0.24, 0.41, 0.58, 0.75]),
            (0.245, [0.155, 0.325, 0.495, 0.665, 0.835]),
            (0.41, [0.07, 0.24, 0.41, 0.58, 0.75]),
            (0.575, [0.155, 0.325, 0.495, 0.665, 0.835]),
            (0.74, [0.14, 0.38, 0.57, 0.82]),
            (0.88, [0.30, 0.70]),
        ],
        # Pattern 3: 4-5-4-5-4-4 (balanced)
        [
            (0.11, [0.11, 0.36, 0.64, 0.89]),
            (0.27, [0.05, 0.25, 0.50, 0.75, 0.95]),
            (0.44, [0.11, 0.36, 0.64, 0.89]),
            (0.60, [0.05, 0.25, 0.50, 0.75, 0.95]),
            (0.77, [0.11, 0.36, 0.64, 0.89]),
            (0.89, [0.11, 0.36, 0.64, 0.89]),
        ],
    ]
    
    # Try multiple seeds with perturbations for better exploration
    for seed in range(8):
        np.random.seed(seed)
        
        # Select a base pattern (rotate through them)
        rows_config = base_patterns[seed % len(base_patterns)]
        
        # Initialize centers with random perturbations
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        idx = 0
        for y, cols in rows_config:
            for x in cols:
                # Add random perturbation to escape local minima
                centers[idx] = [x + np.random.uniform(-0.02, 0.02),
                               y + np.random.uniform(-0.02, 0.02)]
                # Better initial radius based on nearest neighbor distance
                radii[idx] = min(x, y, 1-x, 1-y) * 0.65
                idx += 1
        
        # Refine initial radii based on actual neighbor distances
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    min_dist = min(min_dist, d)
            radii[i] = min(radii[i], min_dist * 0.35)
        
        # Create initial parameter vector
        params = np.concatenate([centers.flatten(), radii])
        
        # Build constraint list for this optimization run
        constraints = []
        
        # Circle-circle constraints (only for nearby pairs to reduce constraint count)
        for i in range(n):
            for j in range(i+1, n):
                # Only enforce constraints for pairs that might overlap
                # (initial distance < 0.4, covering most potential overlaps)
                if np.linalg.norm(centers[i] - centers[j]) < 0.4:
                    constraints.append({'type': 'ineq', 'fun': make_circle_circle_constraint(i, j)})
        
        # Boundary constraints for all circles
        for i in range(n):
            for axis in range(2):  # x and y
                constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, axis, 0)})
                constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, axis, 1)})
        
        try:
            # Run SLSQP optimization with explicit constraints
            result = minimize(
                objective,
                params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            # Extract optimized parameters
            opt_centers = result.x[:2*n].reshape(n, 2)
            opt_radii = result.x[2*n:]
            
            # Light repair to handle numerical precision issues
            for i in range(n):
                # Clamp radii to positive values
                opt_radii[i] = max(0.001, opt_radii[i])
                # Clamp centers to valid range
                opt_centers[i, 0] = max(opt_radii[i], min(1 - opt_radii[i], opt_centers[i, 0]))
                opt_centers[i, 1] = max(opt_radii[i], min(1 - opt_radii[i], opt_centers[i, 1]))
            
            # Minor overlap fix with small scaling
            for _ in range(10):
                max_overlap = 0
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(opt_centers[i] - opt_centers[j])
                        if d > 0 and opt_radii[i] + opt_radii[j] > d:
                            overlap = opt_radii[i] + opt_radii[j] - d
                            max_overlap = max(max_overlap, overlap)
                            scale = d / (opt_radii[i] + opt_radii[j])
                            opt_radii[i] *= scale
                            opt_radii[j] *= scale
                if max_overlap < 1e-10:
                    break
            
            # Validate packing
            valid = True
            for i in range(n):
                if opt_radii[i] <= 0:
                    valid = False
                    break
                if opt_centers[i][0] < opt_radii[i] or opt_centers[i][0] > 1 - opt_radii[i]:
                    valid = False
                    break
                if opt_centers[i][1] < opt_radii[i] or opt_centers[i][1] > 1 - opt_radii[i]:
                    valid = False
                    break
                for j in range(i+1, n):
                    d = np.linalg.norm(opt_centers[i] - opt_centers[j])
                    if d < opt_radii[i] + opt_radii[j] - 1e-8:
                        valid = False
                        break
            
            current_sum = np.sum(opt_radii)
            if valid and current_sum > best_sum:
                best_sum = current_sum
                best_centers = opt_centers.copy()
                best_radii = opt_radii.copy()
        
        except Exception as e:
            pass
    
    # If no valid result from optimization, use base pattern with simple repair
    if best_sum <= 0:
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        idx = 0
        for y, cols in base_patterns[0]:
            for x in cols:
                centers[idx] = [x, y]
                radii[idx] = min(x, y, 1-x, 1-y) * 0.7
                idx += 1
        
        # Simple repair for fallback
        for _ in range(20):
            for i in range(n):
                radii[i] = min(radii[i], centers[i][0], centers[i][1], 
                              1-centers[i][0], 1-centers[i][1])
                centers[i][0] = max(radii[i], min(1-radii[i], centers[i][0]))
                centers[i][1] = max(radii[i], min(1-radii[i], centers[i][1]))
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d > 0 and radii[i] + radii[j] > d:
                        scale = d / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale
        return centers, radii, np.sum(radii)
    
    return best_centers, best_radii, best_sum


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

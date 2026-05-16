# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using scipy.optimize.SLSQP optimization"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint optimization of circle positions and radii using scipy.optimize.SLSQP.
    
    Approach:
    1. Initialize with hexagonal pattern for good starting configuration
    2. Use SLSQP to jointly optimize all 78 variables (26 centers × 2 + 26 radii)
    3. Minimize negative sum of radii + penalty for constraint violations
    4. Use multiple restarts with different patterns to escape local minima
    5. Validate and repair any constraint violations
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    def objective(params):
        """Objective: maximize sum of radii with penalty for constraint violations"""
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        # Primary objective: maximize sum of radii (negative for minimization)
        obj = -np.sum(radii)
        
        # Penalty for overlaps (strong penalty)
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                overlap = max(0, radii[i] + radii[j] - d)
                penalty += 1000 * overlap ** 2
        
        # Penalty for boundary violations (strong penalty)
        for i in range(n):
            if radii[i] > centers[i][0]:
                penalty += 1000 * (radii[i] - centers[i][0]) ** 2
            if radii[i] > centers[i][1]:
                penalty += 1000 * (radii[i] - centers[i][1]) ** 2
            if radii[i] > 1 - centers[i][0]:
                penalty += 1000 * (radii[i] - (1 - centers[i][0])) ** 2
            if radii[i] > 1 - centers[i][1]:
                penalty += 1000 * (radii[i] - (1 - centers[i][1])) ** 2
        
        return obj + penalty
    
    def repair_packing(centers, radii):
        """Repair any constraint violations to ensure validity"""
        n = len(radii)
        
        # Fix boundary violations first
        for i in range(n):
            radii[i] = min(radii[i], centers[i][0], centers[i][1], 
                          1-centers[i][0], 1-centers[i][1])
            centers[i][0] = max(radii[i], min(1-radii[i], centers[i][0]))
            centers[i][1] = max(radii[i], min(1-radii[i], centers[i][1]))
        
        # Fix overlaps with proportional scaling
        for _ in range(50):
            max_overlap = 0
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d > 0 and radii[i] + radii[j] > d:
                        overlap = radii[i] + radii[j] - d
                        max_overlap = max(max_overlap, overlap)
                        total = radii[i] + radii[j]
                        scale = max(0.95, d / total)
                        radii[i] *= scale
                        radii[j] *= scale
            if max_overlap < 1e-8:
                break
        
        # Final boundary check
        for i in range(n):
            radii[i] = min(radii[i], centers[i][0], centers[i][1], 
                          1-centers[i][0], 1-centers[i][1])
            centers[i][0] = max(radii[i], min(1-radii[i], centers[i][0]))
            centers[i][1] = max(radii[i], min(1-radii[i], centers[i][1]))
        
        return centers, radii
    
    # Define bounds for optimization
    # Centers: [0, 1] for x and y
    # Radii: [0.001, 0.5] for each circle
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1)])  # x, y bounds
    for _ in range(n):
        bounds.append((0.001, 0.5))  # radius bounds
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
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
        
        try:
            # Run SLSQP optimization with more iterations
            result = minimize(
                objective,
                params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
            )
            
            # Extract optimized parameters
            opt_centers = result.x[:2*n].reshape(n, 2)
            opt_radii = result.x[2*n:]
            
            # Repair any constraint violations
            opt_centers, opt_radii = repair_packing(opt_centers, opt_radii)
            
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
    
    # If no valid result from optimization, use last pattern with repair
    if best_centers is None:
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        idx = 0
        for y, cols in patterns[0]:
            for x in cols:
                centers[idx] = [x, y]
                radii[idx] = min(x, y, 1-x, 1-y) * 0.7
                idx += 1
        centers, radii = repair_packing(centers, radii)
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

# EVOLVE-BLOCK-START
"""Circle packing for n=26 using Voronoi-based initialization + SLSQP optimization"""
import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.spatial import Voronoi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_voronoi_radii(centers):
    """
    Compute optimal radii for fixed centers using Voronoi diagram.
    Each radius = min distance from center to Voronoi vertices or square boundaries.
    This provides a geometrically optimal radius assignment for the given centers.
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # First compute distance to square boundaries for each center
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, 1-x, y, 1-y)
    
    # Compute Voronoi diagram
    if SCIPY_AVAILABLE:
        try:
            vor = Voronoi(centers)
            
            # For each region, find the minimum distance to Voronoi vertices
            for i in range(n):
                region_idx = vor.point_region[i]
                if region_idx == -1:
                    continue
                vertices = vor.regions[region_idx]
                if -1 in vertices:
                    # Unbounded region, skip Voronoi vertices
                    continue
                for v_idx in vertices:
                    if v_idx == -1:
                        continue
                    v = vor.vertices[v_idx]
                    # Clip to square bounds
                    v_clipped = np.clip(v, 0, 1)
                    dist = np.linalg.norm(centers[i] - v_clipped)
                    radii[i] = min(radii[i], dist)
        except:
            pass
    
    return radii


def optimize_centers_voronoi(centers):
    """
    Optimize center positions to maximize sum of Voronoi-based radii.
    Uses L-BFGS-B for efficient gradient-based optimization.
    """
    if not SCIPY_AVAILABLE:
        return centers
    
    def objective(x):
        centers_opt = x.reshape(-1, 2)
        radii = compute_voronoi_radii(centers_opt)
        return -np.sum(radii)  # Minimize negative sum = maximize sum
    
    result = minimize(
        objective, centers.flatten(), method='L-BFGS-B',
        bounds=[(0, 1)] * (2 * len(centers)),
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    
    optimized_centers = result.x.reshape(-1, 2)
    optimized_radii = compute_voronoi_radii(optimized_centers)
    
    return optimized_centers, optimized_radii


def construct_packing():
    """
    Use SLSQP optimization with multiple random restarts for joint center-radius optimization.
    Objective: maximize sum of radii.
    Constraints: circle containment (x±r in [0,1], y±r in [0,1]) and non-overlap (dist >= ri+rj).
    Multiple restarts from different initial configurations to escape local optima and find better solutions.
    """
    n = 26
    best_sum = -1
    best_centers = None
    best_radii = None
    
    # Try multiple restarts with different initial configurations to escape local optima
    for restart in range(8):
        # Generate initial configuration with random perturbation
        centers = []
        
        # Base hexagonal-like pattern with 6-5-6-5-4 row configuration
        row_configs = [
            (6, 0.03, 0.97),
            (5, 0.09, 0.91),
            (6, 0.03, 0.97),
            (5, 0.09, 0.91),
            (4, 0.14, 0.86),
        ]
        y_positions = [0.10, 0.28, 0.50, 0.72, 0.90]
        
        for row_idx, (count, x_start, x_end) in enumerate(row_configs):
            y = y_positions[row_idx]
            x_step = (x_end - x_start) / (count - 1)
            for i in range(count):
                x = x_start + i * x_step
                # Add random perturbation to escape local optima
                if restart > 0:
                    x += np.random.uniform(-0.03, 0.03)
                    y += np.random.uniform(-0.02, 0.02)
                centers.append([x, y])
        
        # Clamp centers to valid range
        centers = np.array(centers)
        centers = np.clip(centers, 0.01, 0.99)
        
        # Step 1: Voronoi-based center optimization for better initialization
        if SCIPY_AVAILABLE:
            centers, initial_radii = optimize_centers_voronoi(centers)
        else:
            initial_radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
        
        # Optimization variables: [x0, y0, x1, y1, ..., x25, y25, r0, r1, ..., r25]
        x0 = np.concatenate([centers.flatten(), initial_radii])
        
        # Bounds: centers in [0, 1], radii in [0, 0.5]
        bounds = [(0, 1) for _ in range(2*n)] + [(0, 0.5) for _ in range(n)]
        
        # Constraint function: all constraints must be >= 0
        def constraints_func(x):
            cons = []
            # Containment constraints for each circle
            for i in range(n):
                cons.append(x[2*i] - x[2*n + i])
                cons.append(1 - x[2*i] - x[2*n + i])
                cons.append(x[2*i+1] - x[2*n + i])
                cons.append(1 - x[2*i+1] - x[2*n + i])
            # Non-overlap constraints for all pairs
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[2*i] - x[2*j]
                    dy = x[2*i+1] - x[2*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cons.append(dist - x[2*n+i] - x[2*n+j])
            return np.array(cons)
        
        # Objective: minimize negative sum of radii (equivalent to maximizing sum)
        def objective(x):
            return -np.sum(x[2*n:])
        
        # Run SLSQP optimization with increased iterations for guaranteed feasibility
        if SCIPY_AVAILABLE:
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints_func},
                options={'maxiter': 400, 'ftol': 1e-14, 'disp': False}
            )
            optimized_x = result.x
        else:
            optimized_x = x0
        
        # Extract optimized centers and radii
        optimized_centers = optimized_x[:2*n].reshape(n, 2)
        optimized_radii = optimized_x[2*n:]
        current_sum = np.sum(optimized_radii)
        
        # Keep best solution across all restarts
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = optimized_centers
            best_radii = optimized_radii
    
    return best_centers, best_radii, best_sum


def compute_max_radii(centers):
    """
    Fallback: Compute max radii using iterative relaxation to resolve overlaps.
    """
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    for iteration in range(100):
        changed = False
        max_change = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if radii[i] + radii[j] > dist + 1e-10:
                    excess = radii[i] + radii[j] - dist
                    total_r = radii[i] + radii[j]
                    if total_r > 0:
                        scale_i = radii[i] / total_r
                        scale_j = radii[j] / total_r
                        reduction_i = excess * scale_i * 0.5
                        reduction_j = excess * scale_j * 0.5
                        new_r_i = max(0.0001, radii[i] - reduction_i)
                        new_r_j = max(0.0001, radii[j] - reduction_j)
                        max_change = max(max_change, abs(radii[i] - new_r_i), abs(radii[j] - new_r_j))
                        radii[i] = new_r_i
                        radii[j] = new_r_j
                        changed = True
        
        if not changed or max_change < 1e-8:
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

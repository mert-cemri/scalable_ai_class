# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 using Voronoi initialization with SLSQP refinement"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi


def construct_packing():
    """
    Voronoi-initialized circle packing with local SLSQP refinement.
    
    Approach:
    1. Generate random point configurations with Voronoi diagram
    2. Set initial radii based on Voronoi vertex distances (natural spacing)
    3. Use SLSQP to jointly optimize all 78 variables (26 centers × 2 + 26 radii)
    4. Multiple random seeds to explore configuration space
    5. Validate and repair any constraint violations
    
    The Voronoi structure provides geometry-aware initialization with built-in
    spacing information, which helps escape local minima better than fixed patterns.
    
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
    
    def voronoi_init(seed):
        """
        Generate initial configuration using Voronoi diagram for natural spacing.
        
        The Voronoi diagram partitions space into regions around each point.
        The distance to Voronoi vertices indicates natural spacing between neighbors.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (centers, radii)
        """
        np.random.seed(seed)
        
        # Generate random points in unit square with margin from edges
        centers = np.random.uniform(0.1, 0.9, (n, 2))
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(centers)
        except Exception:
            # Fallback if Voronoi fails
            radii = np.zeros(n)
            for i in range(n):
                radii[i] = min(centers[i][0], centers[i][1], 
                              1-centers[i][0], 1-centers[i][1]) * 0.3
            return centers, radii
        
        # Set radii based on Voronoi vertex distances
        radii = np.zeros(n)
        for i in range(n):
            region_idx = vor.point_region[i]
            vertices_idx = vor.regions[region_idx]
            
            # Handle unbounded regions (contain -1 for infinity)
            if -1 in vertices_idx or len(vertices_idx) == 0:
                # Use nearest neighbor distance for unbounded regions
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(centers[i] - centers[j])
                        min_dist = min(min_dist, dist)
                radii[i] = min_dist * 0.35
            else:
                # Calculate min distance to all vertices of this region
                min_dist = float('inf')
                for v_idx in vertices_idx:
                    dist = np.linalg.norm(centers[i] - vor.vertices[v_idx])
                    min_dist = min(min_dist, dist)
                # Use fraction of vertex distance as radius
                radii[i] = min_dist * 0.45
        
        # Enforce boundary constraints
        for i in range(n):
            max_r = min(centers[i][0], centers[i][1], 
                       1-centers[i][0], 1-centers[i][1])
            radii[i] = min(radii[i], max_r)
        
        # Scale down to avoid initial overlaps
        radii *= 0.6
        
        return centers, radii
    
    # Define bounds for optimization
    bounds = []
    for _ in range(n):
        bounds.extend([(0.05, 0.95), (0.05, 0.95)])  # x, y bounds with margin
    for _ in range(n):
        bounds.append((0.001, 0.5))  # radius bounds
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Try multiple Voronoi initializations with different seeds
    for seed in range(10):
        # Generate Voronoi-initialized configuration
        centers, radii = voronoi_init(seed)
        
        # Create initial parameter vector
        params = np.concatenate([centers.flatten(), radii])
        
        try:
            # Run SLSQP optimization
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
        
        except Exception:
            pass
    
    # Fallback: use hexagonal pattern if Voronoi didn't produce valid result
    if best_centers is None:
        hex_pattern = [
            (0.10, [0.10, 0.30, 0.50, 0.70]),
            (0.25, [0.20, 0.40, 0.60, 0.80]),
            (0.40, [0.08, 0.28, 0.48, 0.68, 0.88]),
            (0.55, [0.18, 0.38, 0.58, 0.78, 0.92]),
            (0.70, [0.10, 0.30, 0.50, 0.70]),
            (0.85, [0.20, 0.40, 0.60, 0.80]),
        ]
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        idx = 0
        for y, cols in hex_pattern:
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

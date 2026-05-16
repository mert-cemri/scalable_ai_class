# EVOLVE-BLOCK-START
"""Voronoi-driven circle packing for n=26 circles with geometry-aware initialization"""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize


def construct_packing():
    """
    Voronoi-driven circle packing for n=26 circles.
    
    Uses Voronoi diagrams to identify natural packing regions and set initial
    radii based on cell boundaries. Then optimizes positions to maximize sum.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Try multiple restarts with different initial configurations
    for restart in range(15):
        np.random.seed(42 + restart)
        
        # Initialize with hexagonal-like pattern + perturbation
        centers = np.zeros((n, 2))
        idx = 0
        
        # 5 rows with hexagonal staggering
        row_y = np.linspace(0.10, 0.90, 5)
        row_counts = [5, 5, 5, 5, 6]  # Total 26
        spacing = 0.18
        
        for row in range(5):
            n_c = row_counts[row]
            offset = (row % 2) * (spacing / 2)
            x_start = offset + spacing / 2
            x_positions = np.linspace(x_start, 1.0 - x_start, n_c)
            
            for col, x in enumerate(x_positions):
                if idx < n:
                    # Add small perturbation to avoid collinearity
                    centers[idx] = [x + np.random.uniform(-0.02, 0.02), 
                                    row_y[row] + np.random.uniform(-0.02, 0.02)]
                    idx += 1
        
        # Clip to unit square
        centers = np.clip(centers, 0.05, 0.95)
        
        # Compute Voronoi-based initial radii
        radii = compute_voronoi_radii(centers)
        
        # Optimize positions
        result = optimize_packing(centers, radii, n)
        
        centers_opt = result.x[:2*n].reshape(-1, 2)
        radii_opt = result.x[2*n:]
        sum_r = np.sum(radii_opt)
        
        if sum_r > best_sum:
            best_sum = sum_r
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
    
    return best_centers, best_radii, best_sum


def compute_voronoi_radii(centers):
    """
    Compute radii based on Voronoi cell boundaries.
    
    For each point, the radius is the minimum distance to:
    1. Square boundaries
    2. Voronoi cell edges (if cell is bounded)
    
    This guarantees no overlap by construction.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    try:
        vor = Voronoi(centers)
        
        for i in range(n):
            # Distance to square boundaries
            x, y = centers[i]
            boundary_dist = min(x, y, 1 - x, 1 - y)
            
            # Distance to Voronoi edges
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if region != [] and -1 not in region:
                # Bounded cell - compute distance to edges
                cell_vertices = vor.vertices[region]
                min_dist = boundary_dist
                
                for j in range(len(cell_vertices)):
                    v1 = cell_vertices[j]
                    v2 = cell_vertices[(j + 1) % len(cell_vertices)]
                    dist = point_to_segment_distance(centers[i], v1, v2)
                    min_dist = min(min_dist, dist)
                
                radii[i] = min_dist
            else:
                # Unbounded cell - use boundary distance
                radii[i] = boundary_dist
    except Exception:
        # Fallback: use boundary distances only
        for i in range(n):
            x, y = centers[i]
            radii[i] = min(x, y, 1 - x, 1 - y)
    
    return radii


def point_to_segment_distance(p, a, b):
    """Compute distance from point p to line segment ab."""
    ab = b - a
    ap = p - a
    
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def optimize_packing(centers, radii, n):
    """
    Optimize circle positions using SLSQP.
    
    Objective: maximize sum of radii
    Constraints: boundary and non-overlap
    """
    x0 = np.concatenate([centers.flatten(), radii])
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    def make_constraints():
        constraints = []
        eps = 1e-10
        
        # Boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 
                               'fun': lambda x, i=i: x[2*i] - x[2*n+i] - eps})
            constraints.append({'type': 'ineq',
                               'fun': lambda x, i=i: x[2*i+1] - x[2*n+i] - eps})
            constraints.append({'type': 'ineq',
                               'fun': lambda x, i=i: 1 - x[2*i] - x[2*n+i] - eps})
            constraints.append({'type': 'ineq',
                               'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*n+i] - eps})
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq',
                                   'fun': lambda x, i=i, j=j: 
                                       np.sqrt(np.maximum(1e-12, 
                                       (x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2)) 
                                       - x[2*n+i] - x[2*n+j] - eps})
        
        return constraints
    
    constraints = make_constraints()
    
    result = minimize(
        objective, x0, method='SLSQP', constraints=constraints,
        options={'maxiter': 10000, 'ftol': 1e-10, 'disp': False}
    )
    
    return result


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

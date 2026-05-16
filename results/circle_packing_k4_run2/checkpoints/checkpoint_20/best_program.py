# EVOLVE-BLOCK-START
"""Multi-restart circle packing with Voronoi initialization for n=26 circles"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi


def construct_packing():
    """
    Multi-restart circle packing with Voronoi-based initialization.
    
    Uses multiple restarts with different initial configurations,
    Voronoi diagrams for radius estimation, and SLSQP optimization.
    Local refinement fine-tunes results after optimization.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum_radii = 0
    best_centers = None
    best_radii = None
    
    # Multiple restarts with different initial configurations
    for restart in range(5):
        # Generate initial centers with perturbation
        centers = generate_initial_centers(n, seed=restart)
        
        # Voronoi-based radius initialization (geometry-aware)
        cell_areas = compute_voronoi_areas(centers)
        radii = np.sqrt(cell_areas / np.pi) * 0.88
        
        # Clip radii to boundaries
        for i in range(n):
            max_r = min(centers[i, 0], centers[i, 1], 
                       1 - centers[i, 0], 1 - centers[i, 1])
            radii[i] = min(radii[i], max_r * 0.95)
        radii = np.maximum(radii, 0.012)
        
        # Prepare for optimization
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3 * i] = centers[i, 0]
            x0[3 * i + 1] = centers[i, 1]
            x0[3 * i + 2] = radii[i]
        
        # Bounds and constraints
        bounds = [(0, 1), (0, 1), (0.001, 0.5)] * n
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3 * i] - x[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3 * i] - x[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3 * i + 1] - x[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3 * i + 1] - x[3 * i + 2]})
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt(
                        (x[3 * i] - x[3 * j]) ** 2 + 
                        (x[3 * i + 1] - x[3 * j + 1]) ** 2
                    ) - x[3 * i + 2] - x[3 * j + 2]
                })
        
        # Optimize
        result = minimize(
            lambda x: -sum(x[3 * i + 2] for i in range(n)),
            x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        # Extract and refine
        optimized_centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            optimized_centers[i, 0] = result.x[3 * i]
            optimized_centers[i, 1] = result.x[3 * i + 1]
            radii[i] = result.x[3 * i + 2]
        
        radii = local_refinement(optimized_centers, radii)
        sum_radii = np.sum(radii)
        
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_centers = optimized_centers.copy()
            best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum_radii


def generate_initial_centers(n, seed=0):
    """Generate hexagonal pattern with random perturbation."""
    np.random.seed(seed)
    
    centers = []
    # Different row configurations for exploration
    row_counts = [5, 5, 6, 5, 5]
    
    for row_idx, row_n in enumerate(row_counts):
        y = 0.12 + row_idx * 0.18
        offset = 0.05 if row_idx % 2 == 1 else 0.05
        spacing = 0.17
        
        for col_idx in range(row_n):
            x = offset + col_idx * spacing
            # Add perturbation
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.015, 0.015)
            centers.append([np.clip(x, 0.08, 0.92), np.clip(y, 0.08, 0.92)])
    
    return np.array(centers)


def compute_voronoi_areas(centers):
    """Compute Voronoi cell areas for radius initialization."""
    n = len(centers)
    cell_areas = np.ones(n) / n
    
    try:
        voronoi = Voronoi(centers)
        for i, region_idx in enumerate(voronoi.point_region):
            region = voronoi.regions[region_idx]
            if -1 in region:
                cx, cy = centers[i]
                cell_areas[i] = min(cx, 1-cx, cy, 1-cy) ** 2 * 2
            elif region:
                vertices = voronoi.vertices[region]
                if len(vertices) >= 3:
                    x, y = vertices[:, 0], vertices[:, 1]
                    cell_areas[i] = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    except:
        pass
    
    total = np.sum(cell_areas)
    if total > 0:
        cell_areas /= total
    return cell_areas


def local_refinement(centers, radii):
    """Local search to refine radii after SLSQP optimization."""
    n = len(radii)
    step = 0.002
    improved = True
    
    while improved:
        improved = False
        for i in range(n):
            best_r, best_c = radii[i], centers[i].copy()
            
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                new_c = np.clip(centers[i] + np.array([dx, dy]), 0.02, 0.98)
                valid = all(np.sqrt(np.sum((new_c - centers[j])**2)) >= radii[i] + radii[j] 
                           for j in range(n) if j != i)
                
                if valid and min(new_c) > radii[i] and max(new_c) < 1 - radii[i]:
                    new_r = min(min(new_c), 1 - max(new_c))
                    for j in range(n):
                        if j != i:
                            dist = np.sqrt(np.sum((new_c - centers[j])**2))
                            new_r = min(new_r, dist - radii[j])
                    
                    if new_r > best_r:
                        best_r, best_c = new_r, new_c
            
            if best_r > radii[i] * 1.001:
                centers[i], radii[i] = best_c, best_r
                improved = True
    
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

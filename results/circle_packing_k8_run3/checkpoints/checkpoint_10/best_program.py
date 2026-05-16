# EVOLVE-BLOCK-START
"""Circle packing using Voronoi-based geometric construction with SLSQP refinement"""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize


def construct_packing():
    """
    Construct circle packing using Voronoi-based geometric construction.
    
    Strategy:
    1. Initialize with hexagonal lattice pattern
    2. Iteratively refine centers using Voronoi cell centroids
    3. Compute radii based on Voronoi geometry (half min distance to neighbors)
    4. Apply SLSQP optimization for fine-tuning
    
    Voronoi approach naturally ensures non-overlapping circles since each circle's
    radius is set to half the distance to its nearest neighbor.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Initialize with hexagonal lattice pattern
    centers = initialize_hexagonal_lattice(n).copy()
    
    # Voronoi-based iterative refinement of centers
    centers = voronoi_refine_centers(centers, n, iterations=50)
    
    # Compute radii based on Voronoi geometry and boundaries
    radii = compute_voronoi_radii(centers, n)
    
    # Final SLSQP refinement for better optimization
    centers, radii = slsqp_refine(centers, radii, n)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii


def voronoi_refine_centers(centers, n, iterations=50):
    """
    Refine center positions using Voronoi cell centroids.
    
    Moves each center toward the centroid of its Voronoi cell, which naturally
    creates more uniform spacing and better packing efficiency.
    
    Args:
        centers: np.array of shape (n, 2) with initial center coordinates
        n: Number of circles
        iterations: Number of refinement iterations
        
    Returns:
        np.array of shape (n, 2) with refined center coordinates
    """
    centers = centers.copy()
    
    for iteration in range(iterations):
        # Compute Voronoi diagram
        vor = Voronoi(centers)
        
        # Move each center toward its cell centroid
        step = 0.3 * (1.0 - iteration / iterations)  # Decreasing step size
        
        for i in range(n):
            # Get Voronoi region vertices for this center
            region_idx = vor.point_region[i]
            vertices_idx = vor.regions[region_idx]
            
            # Filter out infinite vertices (-1)
            finite_vertices_idx = [v for v in vertices_idx if v >= 0]
            
            if len(finite_vertices_idx) > 0:
                # Compute centroid of Voronoi cell
                centroid = np.mean(vor.vertices[finite_vertices_idx], axis=0)
                # Move center toward centroid
                centers[i] = centers[i] + step * (centroid - centers[i])
        
        # Clip to valid range
        centers = np.clip(centers, 0.02, 0.98)
    
    return centers


def compute_voronoi_radii(centers, n):
    """
    Compute radii based on Voronoi geometry.
    
    Sets each radius to half the minimum distance to nearest neighbor or boundary,
    ensuring non-overlapping circles by construction.
    
    Args:
        centers: np.array of shape (n, 2) with center coordinates
        n: Number of circles
        
    Returns:
        np.array of shape (n) with radius of each circle
    """
    radii = np.zeros(n)
    
    for i in range(n):
        # Distance to nearest other circle center
        min_dist = np.inf
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, dist)
        
        # Distance to boundary
        x, y = centers[i]
        min_dist = min(min_dist, x, y, 1 - x, 1 - y)
        
        # Set radius to half the minimum distance (ensures no overlap)
        radii[i] = 0.5 * min_dist
    
    return radii


def slsqp_refine(centers, radii, n, maxiter=500):
    """
    Refine packing using SLSQP optimization.
    
    Maximizes sum of radii subject to non-overlapping and boundary constraints.
    Uses SLSQP for efficient constraint handling.
    
    Args:
        centers: np.array of shape (n, 2) with center coordinates
        radii: np.array of shape (n) with initial radii
        n: Number of circles
        maxiter: Maximum optimization iterations
        
    Returns:
        Tuple of (centers, radii) with optimized values
    """
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Objective: minimize negative sum of radii
    def objective(x):
        return -np.sum(x[2*n:])
    
    # Build constraints
    constraints = []
    
    # Pairwise non-overlapping: distance >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            def constraint(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                ri = x[2*n+i]
                rj = x[2*n+j]
                return np.linalg.norm(ci - cj) - ri - rj
            constraints.append({'type': 'ineq', 'fun': constraint})
    
    # Boundary constraints for each circle
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*n+i]})
    
    # Bounds for variables
    bounds = [(0, 1)] * 2*n + [(0, 0.5)] * n
    
    # Run SLSQP optimization
    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False}
    )
    
    centers = result.x[:2*n].reshape(-1, 2)
    radii = result.x[2*n:]
    radii = np.maximum(radii, 0.001)
    
    return centers, radii


def initialize_hexagonal_lattice(n, spacing=0.175):
    """
    Initialize circle centers using hexagonal lattice pattern.
    
    Hexagonal packing achieves pi/(2*sqrt(3)) ≈ 0.9069 density.
    Uses staggered rows for maximum packing efficiency.
    
    Args:
        n: Number of circles to place
        spacing: Distance between circle centers in the lattice
        
    Returns:
        np.array of shape (n, 2) with initial center coordinates
    """
    centers = []
    
    # Create hexagonal rows with staggered pattern
    for row in range(6):
        y = 0.04 + row * spacing * np.sqrt(3) / 2
        if y > 0.96:
            break
        offset = spacing / 2 if row % 2 == 1 else 0
        x = 0.04 + offset
        while x < 0.96 and len(centers) < n:
            centers.append([x, y])
            x += spacing
    
    # Fill remaining circles in gaps
    while len(centers) < n:
        for row in range(6):
            y = 0.04 + row * spacing * np.sqrt(3) / 2
            if len(centers) >= n:
                break
            offset = spacing / 4 if row % 2 == 0 else spacing * 0.75
            x = 0.04 + offset
            while x < 0.96 and len(centers) < n:
                centers.append([x, y])
                x += spacing / 2
    
    return np.array(centers[:n])


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

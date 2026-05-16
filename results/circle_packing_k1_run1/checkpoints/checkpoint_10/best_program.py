"""Circle packing for n=26 circles in unit square targeting sum ~2.635"""
import numpy as np


def construct_packing():
    """
    Construct optimized hexagonal lattice packing with iterative radius growth.
    Uses multiple row configurations and grows circles proportionally.
    Row configuration [6,5,5,5,5] = 26 circles with optimized spacing.
    """
    centers = []
    row_sizes = [6, 5, 5, 5, 5]
    
    # Calculate optimal spacing for hexagonal packing in unit square
    # Target: fit 6 circles across, 5 rows tall with hexagonal spacing
    r_est = 0.095  # Increased from 0.085 for denser packing
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    # Calculate bounds and center the packing
    total_width = (max(row_sizes) - 1) * dx
    total_height = (len(row_sizes) - 1) * dy
    x_start = (1 - total_width) / 2
    y_start = (1 - total_height) / 2
    
    for row_idx, row_n in enumerate(row_sizes):
        y = y_start + row_idx * dy
        x_offset = dx / 2 if row_idx % 2 == 1 else 0
        for col_idx in range(row_n):
            x = x_start + col_idx * dx + x_offset
            # Keep within bounds with small margin
            centers.append([max(0.005, min(0.995, x)), max(0.005, min(0.995, y))])
    
    centers = np.array(centers[:26])
    radii = compute_optimal_radii(centers)
    return centers, radii, np.sum(radii)


def compute_optimal_radii(centers):
    """
    Iteratively compute maximum radii using proportional growth.
    Each circle radius limited by: distance to border, distance to neighbors/2
    Uses iterative relaxation with proportional scaling for better convergence.
    """
    n = len(centers)
    radii = np.ones(n) * 0.05  # Start smaller to allow growth
    
    # Precompute distances between all circle pairs
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dist_matrix[i, j] = dist_matrix[j, i] = d
    
    # Iterative optimization with proportional growth
    for iteration in range(200):
        new_radii = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            # Distance to nearest border
            r_border = min(x, y, 1 - x, 1 - y)
            
            # Distance to nearest neighbor constraint
            r_neighbor = float('inf')
            for j in range(n):
                if i != j and dist_matrix[i, j] > 0:
                    r_neighbor = min(r_neighbor, dist_matrix[i, j] / 2)
            
            # Take minimum of all constraints
            new_radii[i] = max(0.001, min(r_border, r_neighbor))
        
        # Check for convergence
        diff = np.max(np.abs(new_radii - radii))
        radii = new_radii
        
        if diff < 1e-9:
            break
    
    return radii


def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii
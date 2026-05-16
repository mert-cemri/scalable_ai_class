# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 with improved constraint solver"""
import numpy as np


def construct_packing():
    """
    Hexagonal lattice with corner-optimized placement for n=26 circles.
    Uses staggered rows with adjusted spacing to better utilize corner space.
    Row configuration [6,5,5,5,5] = 26 circles with optimized geometry.
    """
    centers = []
    row_sizes = [6, 5, 5, 5, 5]
    
    # Improved spacing estimates based on target sum ~2.635
    # Average radius would be ~0.101 for 26 circles
    r_est = 0.098
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    # Center the packing but allow more corner utilization
    total_width = (max(row_sizes) - 1) * dx
    total_height = (len(row_sizes) - 1) * dy
    x_start = (1 - total_width) / 2
    y_start = (1 - total_height) / 2
    
    for row_idx, row_n in enumerate(row_sizes):
        y = y_start + row_idx * dy
        x_offset = dx / 2 if row_idx % 2 == 1 else 0
        for col_idx in range(row_n):
            x = x_start + col_idx * dx + x_offset
            # Keep within bounds with minimal margin
            centers.append([max(0.001, min(0.999, x)), max(0.001, min(0.999, y))])
    
    centers = np.array(centers[:26])
    radii = compute_optimal_radii(centers)
    return centers, radii, np.sum(radii)


def compute_optimal_radii(centers):
    """
    Compute maximum valid radii using proper constraint satisfaction.
    Solves r_i + r_j <= dist(i,j) iteratively with relaxation.
    Also respects boundary constraints: r_i <= min(x, y, 1-x, 1-y)
    """
    n = len(centers)
    radii = np.ones(n) * 0.10  # Better initial estimate
    
    # Precompute pairwise distances for efficiency
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dist_matrix[i, j] = dist_matrix[j, i] = d
    
    # Iterative relaxation - solve r_i + r_j <= d properly
    for iteration in range(300):
        new_radii = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            # Distance to borders
            r_border = min(x, y, 1 - x, 1 - y)
            
            # Constraint from each neighbor: r_i <= d(i,j) - r_j
            r_min = r_border
            for j in range(n):
                if i != j:
                    d = dist_matrix[i, j]
                    r_min = min(r_min, d - radii[j])
            
            # Apply damping for stability
            new_radii[i] = max(0.001, 0.8 * r_min + 0.2 * radii[i])
        
        # Check convergence
        if np.max(np.abs(new_radii - radii)) < 1e-10:
            break
        radii = new_radii
    
    # Final enforcement pass to ensure all constraints satisfied
    for _ in range(100):
        changed = False
        for i in range(n):
            x, y = centers[i]
            r_border = min(x, y, 1 - x, 1 - y)
            
            for j in range(n):
                if i != j:
                    d = dist_matrix[i, j]
                    if radii[i] + radii[j] > d + 1e-12:
                        # Reduce the larger radius proportionally
                        excess = radii[i] + radii[j] - d
                        if radii[i] > radii[j]:
                            radii[i] -= excess * 0.5
                        else:
                            radii[j] -= excess * 0.5
                        changed = True
            
            radii[i] = min(radii[i], r_border)
        
        if not changed:
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

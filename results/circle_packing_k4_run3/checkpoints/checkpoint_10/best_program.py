# EVOLVE-BLOCK-START
"""Joint optimization of 26 circle positions and radii using SLSQP with multiple restarts"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Jointly optimize 26 circle positions and radii using SLSQP optimization.
    
    Strategy:
    - Use scipy.optimize.minimize with SLSQP method
    - 78 variables: 26 centers (x,y) + 26 radii
    - Objective: minimize negative sum of radii (equivalent to maximizing sum)
    - Constraints: non-overlap between all pairs and boundary conditions
    - Multiple restarts from different initializations to avoid local optima
    - Post-optimization refinement to push circles tighter
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Try multiple different initializations and keep the best
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Different row patterns to try for hexagonal packing
    row_patterns = [
        [5, 5, 5, 5, 6],
        [5, 5, 6, 5, 5],
        [6, 5, 5, 5, 5],
        [5, 6, 5, 5, 5],
        [4, 6, 6, 5, 5],
        [5, 5, 5, 6, 5],
    ]
    
    for pattern in row_patterns:
        centers_init = np.zeros((n, 2))
        horizontal_spacing = 0.22
        vertical_spacing = horizontal_spacing * np.sqrt(3) / 2
        
        idx = 0
        for row in range(len(pattern)):
            offset = horizontal_spacing / 2 if row % 2 == 1 else 0
            row_y = 0.1 + row * vertical_spacing
            for col in range(pattern[row]):
                col_x = offset + col * horizontal_spacing
                centers_init[idx] = [col_x, row_y]
                idx += 1
        
        # Scale to fit in unit square
        max_x = np.max(centers_init[:, 0])
        max_y = np.max(centers_init[:, 1])
        scale_factor = min(0.95 / max_x, 0.95 / max_y)
        centers_init *= scale_factor
        centers_init[:, 0] += (1 - np.max(centers_init[:, 0])) / 2
        centers_init[:, 1] += (1 - np.max(centers_init[:, 1])) / 2
        
        # Initialize radii from distance to borders
        radii_init = np.zeros(n)
        for i in range(n):
            x, y = centers_init[i]
            radii_init[i] = min(x, y, 1 - x, 1 - y) * 0.90
        
        # Pack all variables into single vector
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers_init[i, 0]
            x0[3*i + 1] = centers_init[i, 1]
            x0[3*i + 2] = radii_init[i]
        
        # Define objective function
        def objective(x):
            radii = x[2::3]
            return -np.sum(radii)
        
        # Define non-overlap constraints
        non_overlap_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(x, ci=i, cj=j):
                    c1 = np.array([x[3*ci], x[3*ci + 1]])
                    c2 = np.array([x[3*cj], x[3*cj + 1]])
                    r1 = x[3*ci + 2]
                    r2 = x[3*cj + 2]
                    dist = np.sqrt(np.sum((c1 - c2) ** 2))
                    return dist - r1 - r2
                non_overlap_constraints.append({'type': 'ineq', 'fun': constraint_func})
        
        # Define boundary constraints
        boundary_constraints = []
        for i in range(n):
            boundary_constraints.append({'type': 'ineq', 'fun': lambda x, ci=i: x[3*ci] - x[3*ci + 2]})
            boundary_constraints.append({'type': 'ineq', 'fun': lambda x, ci=i: x[3*ci + 1] - x[3*ci + 2]})
            boundary_constraints.append({'type': 'ineq', 'fun': lambda x, ci=i: 1 - x[3*ci] - x[3*ci + 2]})
            boundary_constraints.append({'type': 'ineq', 'fun': lambda x, ci=i: 1 - x[3*ci + 1] - x[3*ci + 2]})
        
        # Define variable bounds
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 0.5))
        
        # Optimize using SLSQP
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=non_overlap_constraints + boundary_constraints,
            bounds=bounds,
            options={'maxiter': 1200, 'ftol': 1e-12, 'disp': False}
        )
        
        # Extract optimized values
        optimized_x = result.x
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = optimized_x[3*i]
            centers[i, 1] = optimized_x[3*i + 1]
            radii[i] = optimized_x[3*i + 2]
        
        sum_radii = np.sum(radii)
        
        # Keep best result
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    # Post-optimization refinement: iteratively push circles tighter
    best_centers, best_radii = refine_packing(best_centers, best_radii, n)
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum


def refine_packing(centers, radii, n, max_iter=200, tol=1e-10):
    """
    Post-optimization refinement to push circles tighter together.
    
    Strategy:
    - Iteratively adjust positions to maximize radii while maintaining constraints
    - For each circle, try small movements in 8 directions
    - Accept movements that allow radius increase
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
        n: number of circles
        max_iter: Maximum iterations for refinement
        tol: Convergence tolerance
    
    Returns:
        Tuple of (refined_centers, refined_radii)
    """
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(max_iter):
        improved = False
        max_improvement = 0.0
        
        # Try to improve each circle
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Try 8 directions + stay
            directions = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            
            best_dir = (0, 0)
            best_r = r
            
            for dx, dy in directions:
                step = 0.005
                new_x = x + dx * step
                new_y = y + dy * step
                
                # Check boundary constraints
                if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
                    continue
                
                # Compute max valid radius at new position
                max_r = min(new_x, new_y, 1 - new_x, 1 - new_y)
                
                # Check non-overlap constraints
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((new_x - centers[j, 0])**2 + (new_y - centers[j, 1])**2)
                        max_r = min(max_r, dist - radii[j])
                
                max_r = max(0, max_r)
                
                if max_r > best_r:
                    best_r = max_r
                    best_dir = (dx, dy)
            
            # Apply improvement if found
            if best_r > r + tol:
                centers[i, 0] += best_dir[0] * 0.005
                centers[i, 1] += best_dir[1] * 0.005
                radii[i] = best_r
                improved = True
                max_improvement = max(max_improvement, best_r - r)
        
        # Check convergence
        if not improved or max_improvement < tol:
            break
    
    # Final pass to enforce all constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1 - x, 1 - y)
        
        for j in range(i + 1, n):
            dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    
    return centers, radii


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

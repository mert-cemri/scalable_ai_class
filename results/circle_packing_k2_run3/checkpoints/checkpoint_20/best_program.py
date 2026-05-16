# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using SLSQP optimization"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Construct optimal circle packing using scipy.optimize.minimize with SLSQP.
    Jointly optimizes 52 variables (26 centers + 26 radii) with constraints:
    - Circle-circle non-overlap: ||c_i - c_j|| >= r_i + r_j
    - Circle-boundary: r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_result = None
    best_sum = 0
    
    # Try multiple initializations
    for init_type in ['hexagonal', 'grid', 'clustered']:
        x0 = get_initial_guess(n, init_type)
        
        # Bounds: (x_min, x_max) for each variable
        # For centers: [0, 1], for radii: [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0)])  # x, y
            bounds.extend([(0.0, 0.5)])  # r
        
        # Define constraints
        constraints = []
        
        # Circle-boundary constraints: r_i <= x_i, r_i <= y_i, x_i <= 1-r_i, y_i <= 1-r_i
        for i in range(n):
            # x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            # y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            # 1 - x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            # 1 - y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
            # r_i >= 0 (handled by bounds, but add for safety)
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2]})
        
        # Circle-circle non-overlap constraints: ||c_i - c_j|| - (r_i + r_j) >= 0
        for i in range(n):
            for j in range(i+1, n):
                def circle_constraint(x, i=i, j=j):
                    dx = x[3*i] - x[3*j]
                    dy = x[3*i+1] - x[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - x[3*i+2] - x[3*j+2]
                constraints.append({'type': 'ineq', 'fun': circle_constraint})
        
        # Objective: minimize negative sum of radii
        def objective(x):
            return -np.sum(x[2::3])  # Sum of all radii
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-10, 'eps': 1e-8}
            )
            
            if result.success or result.fun < 0:
                sum_radii = -result.fun
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = result.x
        except Exception as e:
            continue
    
    # If optimization failed, fall back to hexagonal with greedy radii
    if best_result is None:
        centers = get_hexagonal_centers(n)
        radii = compute_max_radii(centers)
        return centers, radii, np.sum(radii)
    
    # Extract centers and radii from optimized result
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = [best_result[3*i], best_result[3*i+1]]
        radii[i] = best_result[3*i+2]
    
    return centers, radii, np.sum(radii)


def get_initial_guess(n, init_type='hexagonal'):
    """
    Generate initial guess for optimization based on pattern type.
    
    Args:
        n: number of circles
        init_type: 'hexagonal', 'grid', or 'clustered'
    
    Returns:
        np.array of shape (3*n,) with [x0, y0, r0, x1, y1, r1, ...]
    """
    result = np.zeros(3 * n)
    
    if init_type == 'hexagonal':
        # Hexagonal lattice with staggered rows
        row_counts = [6, 6, 6, 4, 4]
        spacing = 0.17
        row_height = spacing * np.sqrt(3) / 2
        start_x, start_y = 0.15, 0.15
        
        idx = 0
        for row_idx, count in enumerate(row_counts):
            y = start_y + row_idx * row_height
            x_offset = 0 if row_idx % 2 == 0 else spacing / 2
            for col in range(count):
                x = start_x + x_offset + col * spacing
                result[3*idx] = x
                result[3*idx+1] = y
                result[3*idx+2] = 0.08  # Initial radius estimate
                idx += 1
    
    elif init_type == 'grid':
        # Grid pattern
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                result[3*idx] = x
                result[3*idx+1] = y
                result[3*idx+2] = 0.08
                idx += 1
                if idx >= n:
                    break
    
    elif init_type == 'clustered':
        # Clustered in corners with larger circles
        idx = 0
        # Four corner clusters with 6 circles each, 2 in center
        corner_positions = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for cx, cy in corner_positions:
            for i in range(6):
                angle = 2 * np.pi * i / 6
                x = cx + 0.08 * np.cos(angle)
                y = cy + 0.08 * np.sin(angle)
                result[3*idx] = x
                result[3*idx+1] = y
                result[3*idx+2] = 0.06
                idx += 1
        # Two center circles
        result[3*idx] = 0.5
        result[3*idx+1] = 0.45
        result[3*idx+2] = 0.10
        idx += 1
        result[3*idx] = 0.5
        result[3*idx+1] = 0.55
        result[3*idx+2] = 0.10
    
    return result


def get_hexagonal_centers(n):
    """
    Generate hexagonal lattice centers as fallback.
    
    Args:
        n: number of circles
    
    Returns:
        np.array of shape (n, 2) with (x, y) coordinates
    """
    centers = []
    row_counts = [6, 6, 6, 4, 4]
    spacing = 0.17
    row_height = spacing * np.sqrt(3) / 2
    start_x, start_y = 0.15, 0.15
    
    for row_idx, count in enumerate(row_counts):
        y = start_y + row_idx * row_height
        x_offset = 0 if row_idx % 2 == 0 else spacing / 2
        for col in range(count):
            x = start_x + x_offset + col * spacing
            centers.append([x, y])
    
    return np.array(centers)


def compute_max_radii(centers):
    """
    Compute maximum radii using greedy reduction for overlaps.
    Fallback when optimization fails.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
    
    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with max possible radius (distance to nearest border)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Greedy reduction for overlapping circles
    for _ in range(100):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    avg = dist / 2
                    diff = (radii[i] - radii[j]) / 2
                    radii[i] = max(avg + diff, avg)
                    radii[j] = max(dist - radii[i], 0)
    
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

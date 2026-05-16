# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using SLSQP optimization"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Optimize circle packing using scipy.optimize.minimize with SLSQP.
    
    Strategy:
    - Use 78 variables (26 x-coords, 26 y-coords, 26 radii)
    - Maximize sum of radii via constrained optimization
    - Multiple diverse initializations to escape local optima
    - Tighter convergence criteria for better solutions
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum_radii = 0
    best_solution = None
    
    # Try 25 diverse initializations to escape local optima
    for seed in range(25):
        np.random.seed(seed)
        
        # Generate diverse initial positions
        if seed < 5:
            initial_positions = generate_hexagonal_pattern(n)
        elif seed < 15:
            initial_positions = generate_random_pattern(n)
        else:
            initial_positions = generate_layered_pattern(n)
        
        # Start with reasonable radii based on expected packing density
        initial_radii = np.full(n, 0.09 + 0.01 * np.random.randn(n))
        initial_radii = np.clip(initial_radii, 0.05, 0.15)
        
        # Flatten to single array: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = initial_positions[i, 0]
            x0[3*i+1] = initial_positions[i, 1]
            x0[3*i+2] = initial_radii[i]
        
        # Define objective: minimize negative sum of radii
        def objective(x):
            radii = x[2::3]
            return -np.sum(radii)
        
        # Build constraints more efficiently
        constraints = []
        
        # Boundary constraints for each circle
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2]})
        
        # Non-overlap constraints for all pairs
        for i in range(n):
            for j in range(i+1, n):
                def non_overlap(x, i=i, j=j):
                    dx = x[3*i] - x[3*j]
                    dy = x[3*i+1] - x[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - x[3*i+2] - x[3*j+2]
                constraints.append({'type': 'ineq', 'fun': non_overlap})
        
        # Bounds for all variables
        bounds = [(0, 1) for _ in range(2*n)] + [(0, 0.5) for _ in range(n)]
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success:
                sum_radii = -result.fun
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_solution = result.x.copy()
        except Exception:
            continue
    
    # Extract final solution
    if best_solution is not None:
        centers = np.array([[best_solution[3*i], best_solution[3*i+1]] for i in range(n)])
        radii = best_solution[2::3]
    else:
        # Fallback to hexagonal pattern if optimization fails
        centers = generate_hexagonal_pattern(n)
        radii = np.full(n, 0.09)
    
    return centers, radii, best_sum_radii


def generate_hexagonal_pattern(n):
    """
    Generate hexagonal grid pattern with optimized spacing for 26 circles.
    Uses staggered rows to maximize packing density.
    """
    centers = []
    base_spacing = 0.14  # Optimized for 26 circles in unit square
    
    for row in range(6):
        y = 0.06 + row * base_spacing * np.sqrt(3) / 2
        offset = base_spacing / 2 if row % 2 == 1 else 0
        for col in range(6):
            if len(centers) >= n:
                break
            x = offset + col * base_spacing + 0.04
            if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98:
                centers.append([x, y])
    
    centers = np.array(centers[:n])
    return centers


def generate_random_pattern(n):
    """
    Generate random initial positions with minimum separation.
    Helps escape local optima by exploring different regions.
    """
    centers = []
    attempts = 0
    max_attempts = 1000
    
    while len(centers) < n and attempts < max_attempts:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        min_dist = 0.12  # Minimum separation
        
        valid = True
        for cx, cy in centers:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                valid = False
                break
        
        if valid:
            centers.append([x, y])
        attempts += 1
    
    centers = np.array(centers[:n])
    return centers


def generate_layered_pattern(n):
    """
    Generate layered pattern with circles arranged in concentric shells.
    This pattern often leads to better local optima for constrained problems.
    """
    centers = []
    
    # Central circle
    centers.append([0.5, 0.5])
    
    # First shell: 6 circles around center
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.15
        centers.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    
    # Second shell: 12 circles
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 0.28
        centers.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    
    # Third shell: remaining circles
    remaining = n - len(centers)
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining + np.pi / remaining
        r = 0.40
        centers.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    
    centers = np.array(centers[:n])
    return centers


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

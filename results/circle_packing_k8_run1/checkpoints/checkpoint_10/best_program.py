# EVOLVE-BLOCK-START
"""Direct optimization with improved hexagonal initialization and more thorough search"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Directly optimize all 26 circle positions and radii using SLSQP with improved strategy.
    
    Key improvements over basic approach:
    1. Hexagonal lattice initialization (optimal infinite packing density ~0.9069)
    2. Increased iterations (2000) and restarts (15) for better exploration
    3. Warm-start from best previous solution for refinement
    4. Better initial radius estimates based on packing density
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    
    # Objective: minimize negative sum of radii (maximizes sum of radii)
    def objective(x):
        return -np.sum(x[2::3])
    
    # Boundary constraints: each circle must be inside unit square
    def make_boundary_constraints(n):
        constraints = []
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
        return constraints
    
    # Non-overlap constraints: distance between centers >= sum of radii
    def make_nonoverlap_constraints(n):
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint(x, i=i, j=j):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return dist - (ri + rj)
                constraints.append({'type': 'ineq', 'fun': constraint})
        return constraints
    
    all_constraints = make_boundary_constraints(n) + make_nonoverlap_constraints(n)
    bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n
    
    # Hexagonal lattice initialization - optimal for dense packing
    def make_hex_initial():
        x = np.zeros(3 * n)
        # Row counts for hexagonal arrangement: 5, 6, 5, 6, 4 = 26 circles
        row_counts = [5, 6, 5, 6, 4]
        idx = 0
        
        # Calculate optimal vertical spacing for hexagonal packing
        # For touching circles: vertical spacing = r * sqrt(3)
        target_sum = 2.63  # Target from AlphaEvolve
        avg_radius = target_sum / n  # ~0.101
        
        row_height = avg_radius * np.sqrt(3)  # Hexagonal row spacing
        start_y = avg_radius + 0.01  # Leave margin from bottom
        
        for row, count in enumerate(row_counts):
            y = start_y + row * row_height
            # Stagger odd rows for hexagonal pattern
            stagger = (avg_radius * np.sqrt(3) / 2) if row % 2 == 1 else 0.0
            
            for col in range(count):
                # Distribute horizontally with hexagonal spacing
                spacing = 2 * avg_radius
                x_val = stagger + col * spacing + avg_radius
                x_val = np.clip(x_val, avg_radius, 1 - avg_radius)
                
                x[3*idx] = x_val
                x[3*idx+1] = np.clip(y, avg_radius, 1 - avg_radius)
                x[3*idx+2] = avg_radius * 0.95  # Slightly smaller to allow growth
                idx += 1
        return x
    
    np.random.seed(42)
    initial_x = make_hex_initial()
    
    # First optimization run with hexagonal initialization
    best_result = minimize(objective, initial_x, method='SLSQP', bounds=bounds,
                          constraints=all_constraints, 
                          options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
    
    best_sum = -best_result.fun if best_result.success else 0.0
    best_x = best_result.x if best_result.success else initial_x
    
    # Multiple random restarts with varied strategies
    for restart in range(15):
        # Generate random initial configuration with hexagonal bias
        random_x = np.zeros(3 * n)
        
        # Mix of hexagonal and random initialization
        if restart < 5:
            # Hexagonal-like with noise
            hex_x = make_hex_initial()
            random_x = hex_x + np.random.normal(0, 0.02, 3 * n)
            random_x = np.clip(random_x, 0.02, 0.98)
            random_x[2::3] = np.clip(random_x[2::3], 0.03, 0.15)
        elif restart < 10:
            # Random with clustering
            for i in range(n):
                random_x[3*i] = np.random.uniform(0.1, 0.9)
                random_x[3*i+1] = np.random.uniform(0.1, 0.9)
                random_x[3*i+2] = np.random.uniform(0.04, 0.12)
        else:
            # Perturb best solution for local refinement
            random_x = best_x + np.random.normal(0, 0.01, 3 * n)
            random_x = np.clip(random_x, 0.02, 0.98)
            random_x[2::3] = np.clip(random_x[2::3], 0.03, 0.15)
        
        result = minimize(objective, random_x, method='SLSQP', bounds=bounds,
                         constraints=all_constraints,
                         options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False})
        
        if result.success:
            new_sum = -result.fun
            if new_sum > best_sum:
                best_sum = new_sum
                best_x = result.x
    
    # Extract centers and radii from optimized variables
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = [best_x[3*i], best_x[3*i+1]]
        radii[i] = best_x[3*i+2]
    
    # Ensure all radii are positive
    radii = np.maximum(radii, 1e-10)
    
    return centers, radii, np.sum(radii)


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

# EVOLVE-BLOCK-START
"""Joint position-radius optimization for n=26 circles with multi-restart SLSQP"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint position-radius optimization for 26 circles using multi-restart SLSQP.
    
    Formulates circle packing as constrained optimization problem:
    - Variables: 78 parameters (26 circles × 3: x, y, radius)
    - Objective: maximize sum of radii (minimize negative sum)
    - Constraints: circles in unit square, no overlaps
    
    Uses multiple random restarts with different initial configurations to
    escape local minima and find better global optima.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum_radii = 0
    best_centers = None
    best_radii = None
    
    # Try multiple row configurations
    row_configs = [
        [5, 5, 6, 5, 5],  # Standard hexagonal
        [6, 5, 5, 5, 5],  # More at bottom
        [5, 6, 5, 5, 5],  # More in second row
    ]
    
    # Multi-restart optimization
    for restart in range(8):
        # Select row configuration
        config = row_configs[restart % len(row_configs)]
        
        # Generate initial hexagonal pattern
        centers = generate_hexagonal_pattern(config)
        
        # Add random perturbation to escape local minima
        centers = add_random_perturbation(centers, seed=restart)
        
        # Scale to fit in unit square with margin
        base_r = 0.10
        max_x = centers[:, 0].max() + base_r
        max_y = centers[:, 1].max() + base_r
        scale = min(1.0 / max_x, 1.0 / max_y) * 0.97
        centers *= scale
        base_r *= scale
        
        # Initialize variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3 * i] = centers[i, 0]
            x0[3 * i + 1] = centers[i, 1]
            x0[3 * i + 2] = base_r
        
        # Bounds: x in [0, 1], y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])
        
        # Define constraints
        constraints = []
        
        # Square boundary constraints for each circle (4 per circle)
        for i in range(n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3 * i] - x[3 * i + 2]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[3 * i] - x[3 * i + 2]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3 * i + 1] - x[3 * i + 2]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[3 * i + 1] - x[3 * i + 2]
            })
        
        # Non-overlap constraints for each pair
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt(
                        (x[3 * i] - x[3 * j]) ** 2 + 
                        (x[3 * i + 1] - x[3 * j + 1]) ** 2
                    ) - x[3 * i + 2] - x[3 * j + 2]
                })
        
        # Objective: minimize negative sum of radii
        def objective(x):
            return -sum(x[3 * i + 2] for i in range(n))
        
        # Run optimization with SLSQP - increased iterations for better convergence
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-9}
        )
        
        # Extract optimized positions and radii
        optimized_centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            optimized_centers[i, 0] = result.x[3 * i]
            optimized_centers[i, 1] = result.x[3 * i + 1]
            radii[i] = result.x[3 * i + 2]
        
        sum_radii = np.sum(radii)
        
        # Keep best result
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_centers = optimized_centers.copy()
            best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum_radii


def generate_hexagonal_pattern(row_config):
    """
    Generate hexagonal circle pattern based on row configuration.
    
    Args:
        row_config: list of number of circles in each row
    
    Returns:
        np.array of shape (n, 2) with (x, y) coordinates
    """
    centers = []
    base_r = 0.10
    
    y_offset = base_r + 0.01
    row_spacing = base_r * np.sqrt(3)
    
    for row_idx, num_circles in enumerate(row_config):
        y = y_offset + row_idx * row_spacing
        
        # Calculate x spacing based on number of circles
        if num_circles == 1:
            x_spacing = 0
        else:
            x_spacing = (1.0 - 2 * base_r) / (num_circles - 1)
        
        # Offset alternating rows for hexagonal pattern
        x_offset = base_r
        if row_idx % 2 == 1:
            x_offset += x_spacing / 2
        
        for i in range(num_circles):
            x = x_offset + i * x_spacing
            centers.append([x, y])
    
    return np.array(centers)


def add_random_perturbation(centers, seed=0):
    """
    Add small random perturbations to centers to escape local minima.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        seed: random seed for reproducibility
    
    Returns:
        np.array of shape (n, 2) with perturbed (x, y) coordinates
    """
    np.random.seed(seed)
    perturbation = np.random.uniform(-0.02, 0.02, centers.shape)
    return centers + perturbation


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

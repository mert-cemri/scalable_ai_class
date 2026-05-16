# EVOLVE-BLOCK-START
"""Direct optimization of 26 circle positions and radii using SLSQP with refined hexagonal packing"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Optimize 26 circle positions and radii using SLSQP with refined hexagonal initialization.
    
    Uses 10 random restarts from perturbed hexagonal grid with optimized spacing
    to escape local minima and find better packings.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    eps = 1e-9  # Small epsilon for numerical stability
    
    # Try more random restarts to explore search space better
    for restart in range(10):
        np.random.seed(1234 + restart)
        
        # Initialize with refined hexagonal grid
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # Better hexagonal packing: 5 rows with counts [5,5,5,5,6]
        row_counts = [5, 5, 5, 5, 6]
        row_y = np.linspace(0.12, 0.88, 5)  # Slightly adjusted vertical spacing
        spacing = 0.175  # Optimized horizontal spacing
        
        idx = 0
        for row in range(5):
            n_c = row_counts[row]
            offset = (row % 2) * (spacing / 2)
            x_start = offset + spacing / 2
            x_positions = np.linspace(x_start, 1.0 - x_start, n_c)
            
            for col, x in enumerate(x_positions):
                if idx < n:
                    centers[idx] = [x, row_y[row]]
                    radii[idx] = 0.045  # Better initial radius estimate
                    idx += 1
        
        # Add larger random perturbations for better exploration
        centers += np.random.uniform(-0.03, 0.03, centers.shape)
        centers = np.clip(centers, 0.04, 0.96)
        
        def objective(x):
            return -np.sum(x[52:])  # Negative sum (minimize)
        
        def make_constraints():
            constraints = []
            
            # Boundary constraints with epsilon for numerical stability
            for i in range(n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i*2] - x[52+i] - eps
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i*2+1] - x[52+i] - eps
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: 1 - x[i*2] - x[52+i] - eps
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: 1 - x[i*2+1] - x[52+i] - eps
                })
            
            # Non-overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: np.sqrt((x[i*2] - x[j*2])**2 + (x[i*2+1] - x[j*2+1])**2) - x[52+i] - x[52+j] - eps
                    })
            
            return constraints
        
        x0 = np.concatenate([centers.flatten(), radii])
        constraints = make_constraints()
        
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            constraints=constraints,
            options={'maxiter': 8000, 'ftol': 1e-10, 'disp': False}
        )
        
        centers_opt = result.x[:52].reshape(-1, 2)
        radii_opt = result.x[52:]
        sum_r = np.sum(radii_opt)
        
        if sum_r > best_sum:
            best_sum = sum_r
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
    
    return best_centers, best_radii, best_sum


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

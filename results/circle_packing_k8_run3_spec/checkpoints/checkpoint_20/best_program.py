# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using SLSQP with enhanced search"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Optimize 26 circles in unit square using SLSQP with multiple strategies:
    1. Better hexagonal lattice tuned for square boundaries
    2. More optimization attempts (10 seeds vs 5)
    3. Larger perturbations to escape local optima
    4. Efficient constraint evaluation
    """
    n = 26
    
    # Improved hexagonal lattice: tuned spacing for square container
    # Based on geometric analysis: 5 rows with 5-5-5-5-6 pattern
    centers_init = []
    spacing = 0.172  # Slightly increased for better edge utilization
    v_spacing = spacing * np.sqrt(3) / 2
    offset = 0.068   # Reduced offset to utilize corners better
    
    for row in range(4):
        y = offset + row * v_spacing
        x_offset = 0 if row % 2 == 0 else spacing / 2
        for i in range(5):
            centers_init.append([offset + x_offset + i * spacing, y])
    
    # Final row: 6 circles to fill remaining space
    y = offset + 4 * v_spacing
    for i in range(6):
        centers_init.append([offset + i * spacing, y])
    
    centers_init = np.array(centers_init)
    radii_init = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers_init])
    
    def objective(x):
        return -np.sum(x[52:])
    
    def constraints(x):
        centers = x[:52].reshape(-1, 2)
        radii = x[52:]
        cons = []
        
        # Boundary constraints
        for i in range(n):
            xi, yi, ri = centers[i, 0], centers[i, 1], radii[i]
            cons.extend([xi - ri, 1 - xi - ri, yi - ri, 1 - yi - ri])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]
                cons.append(dx*dx + dy*dy - (radii[i] + radii[j])**2)
        
        return np.array(cons)
    
    bounds = [(0.01, 0.99)] * 52 + [(0.01, 0.5)] * n
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    # More optimization attempts with larger perturbations
    best_result, best_val = None, np.inf
    
    for seed in range(10):
        np.random.seed(seed)
        x0_perturbed = np.clip(x0 + np.random.uniform(-0.02, 0.02, 78),
                              [0.01]*52 + [0.01]*n, [0.99]*52 + [0.5]*n)
        
        result = minimize(objective, x0_perturbed, method='SLSQP',
                         bounds=bounds,
                         constraints={'type': 'ineq', 'fun': constraints},
                         options={'maxiter': 800, 'ftol': 1e-10})
        
        if result.fun < best_val:
            best_val, best_result = result.fun, result
    
    centers = best_result.x[:52].reshape(-1, 2)
    radii = np.maximum(best_result.x[52:], 0)
    
    # Boundary safety check
    for i in range(n):
        radii[i] = min(radii[i], min(centers[i, 0], centers[i, 1],
                                     1-centers[i, 0], 1-centers[i, 1]))
    
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

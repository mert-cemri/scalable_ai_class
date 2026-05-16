# EVOLVE-BLOCK-START
"""Linear programming optimized hexagonal layer packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog


def construct_packing():
    """
    Linear programming optimized hexagonal layer packing for n=26 circles.
    Uses scipy.optimize.linprog to find optimal spacing parameters.
    Row distribution: 5-5-5-5-4-2 = 26 circles.
    
    Linearizes boundary constraints for hexagonal pattern:
    - x coordinates are linear functions of spacing
    - y coordinates are linear functions of row_height
    - Boundary constraints become linear inequalities
    
    Hexagonal packing achieves ~15% better density than square grids
    (pi/(2*sqrt(3)) ≈ 0.9069 vs pi/4 ≈ 0.7854).
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal pattern: 6 rows with 5-5-5-5-4-2 distribution
    # This staggered pattern fits 26 circles efficiently in unit square
    row_counts = [5, 5, 5, 5, 4, 2]
    
    # For hexagonal packing, row_height = spacing * sqrt(3)/2
    # This gives us one degree of freedom: spacing
    # Use linprog to find optimal spacing within bounds
    
    # Objective: maximize spacing (negative for minimization)
    c = [-1]
    
    # Linear constraints for boundary conditions:
    # 5 circles in row: 4*spacing + 2*r <= 1 (r approx spacing/2)
    # 6 rows with hex spacing: 5*row_height + 2*r <= 1
    # row_height = spacing * sqrt(3)/2
    
    A_ub = np.array([
        [1],  # spacing <= 0.22 (fits 5 circles horizontally)
        [5 * np.sqrt(3)/2 + 1]  # 6 rows fit vertically with hexagonal spacing
    ])
    b_ub = np.array([0.22, 1.0])
    
    # Bounds for spacing variable
    bounds = [(0.15, 0.25)]
    
    # Solve linear program using HiGHS solver
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        hex_spacing = result.x[0]
    else:
        # Fallback to reasonable default
        hex_spacing = 0.185
    
    # Compute row_height from hexagonal geometry
    row_height = hex_spacing * np.sqrt(3) / 2
    
    # Build final centers with optimal parameters
    idx = 0
    for row_idx, count in enumerate(row_counts):
        # Offset every other row for hexagonal packing
        x_offset = hex_spacing / 2 if row_idx % 2 == 1 else 0
        y = hex_spacing / 2 + row_idx * row_height
        for col in range(count):
            if idx < n:
                x = hex_spacing / 2 + col * hex_spacing + x_offset
                centers[idx] = [x, y]
                idx += 1
    
    # Compute final radii respecting all constraints
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        r = min(x, y, 1-x, 1-y)  # Border constraint
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                r = min(r, d/2)  # Non-overlap constraint
        radii[i] = r
    
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

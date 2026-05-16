# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Uses scipy.optimize.minimize with custom objective function directly minimizing
slope changes, lag error, and false reversals. Warm-started from spline solution
to avoid local minima and ensure computational efficiency.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline


def compute_local_noise_variance(x, window_size=20):
    """Compute local noise variance using MAD-based estimation."""
    if len(x) < 3:
        return np.var(x) if len(x) > 0 else 0.0
    
    half_window = window_size // 2
    local_variances = []
    
    for i in range(len(x)):
        start = max(0, i - half_window)
        end = min(len(x), i + half_window + 1)
        window = x[start:end]
        if len(window) > 2:
            mad = np.median(np.abs(window - np.median(window)))
            variance_est = (mad / 0.6745) ** 2
            local_variances.append(variance_est)
        else:
            local_variances.append(np.var(window) if len(window) > 1 else 0.0)
    
    return float(np.mean(local_variances))


def get_spline_warm_start(x, window_size=20):
    """Get initial solution from spline for optimization warm start."""
    if len(x) < 5:
        return np.array(x)
    
    noise_variance = compute_local_noise_variance(x, window_size)
    s = noise_variance * len(x) * 1.5
    s = max(s, 0.1)
    
    x_indices = np.arange(len(x))
    spline = UnivariateSpline(x_indices, x, s=s, k=3)
    return spline(x_indices)


def compute_slope_changes(x):
    """Count number of slope direction changes in signal."""
    if len(x) < 3:
        return 0.0
    diffs = np.diff(x)
    signs = np.sign(diffs)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
    return float(sign_changes)


def compute_lag_error(x_filtered, x_input, window_size=20):
    """Estimate lag error between filtered and input signal using local trends."""
    if len(x_filtered) != len(x_input) or len(x_filtered) < 5:
        return 0.0
    
    half_window = min(window_size // 2, len(x_filtered) // 4)
    lag_errors = []
    
    for i in range(half_window, len(x_filtered) - half_window):
        x_range = range(-half_window, half_window + 1)
        filtered_trend = np.polyfit(x_range, x_filtered[i-half_window:i+half_window+1], 1)[0]
        input_trend = np.polyfit(x_range, x_input[i-half_window:i+half_window+1], 1)[0]
        lag_errors.append(abs(filtered_trend - input_trend))
    
    return float(np.mean(lag_errors)) if lag_errors else 0.0


def compute_false_reversals(x_filtered, x_input, window_size=20):
    """Estimate false reversals based on local trend consistency."""
    if len(x_filtered) < window_size:
        return 0.0
    
    half_window = window_size // 2
    false_reversals = 0.0
    
    for i in range(half_window, len(x_filtered) - half_window):
        filtered_diff = np.diff(x_filtered[i-half_window:i+half_window+1])
        input_diff = np.diff(x_input[i-half_window:i+half_window+1])
        
        if len(filtered_diff) > 0:
            filtered_signs = np.sign(filtered_diff)
            input_signs = np.sign(input_diff)
            
            filtered_changes = np.sum(np.abs(np.diff(filtered_signs)) > 0)
            input_changes = np.sum(np.abs(np.diff(input_signs)) > 0)
            false_reversals += max(0, filtered_changes - input_changes)
    
    return float(false_reversals)


def objective_function(x_filtered, x_input, window_size=20):
    """
    Objective function for signal filtering optimization.
    Minimizes weighted sum of slope changes, lag error, and false reversals.
    """
    # Weighted penalties for each metric
    slope_penalty = compute_slope_changes(x_filtered) * 2.5
    lag_penalty = compute_lag_error(x_filtered, x_input, window_size) * 1.5
    false_penalty = compute_false_reversals(x_filtered, x_input, window_size) * 2.0
    
    # Smoothness penalty (second derivative) to prevent overfitting
    if len(x_filtered) > 2:
        second_diff = np.diff(x_filtered, n=2)
        smoothness_penalty = np.sum(second_diff ** 2) * 0.1
    else:
        smoothness_penalty = 0.0
    
    return slope_penalty + lag_penalty + false_penalty + smoothness_penalty


def adaptive_optimization_filter(x, window_size=20):
    """
    Optimization-based signal filtering using scipy.optimize.minimize.
    
    Uses SLSQP method to directly minimize composite score components:
    - Slope changes (direction reversals)
    - Lag error (responsiveness)
    - False reversals (noise-induced trend changes)
    
    Warm-started from spline solution to avoid local minima.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < 5:
        output_length = len(x) - window_size + 1
        if output_length <= 0:
            return np.array(x)
        return np.array(x)[-output_length:]
    
    # Warm start from spline solution
    x_array = np.asarray(x, dtype=float)
    x0 = get_spline_warm_start(x_array, window_size)
    
    # Set bounds to keep filtered values within reasonable range of input
    signal_std = np.std(x_array)
    bounds = [(max(0, xi - 3 * signal_std), min(2 * np.max(x_array), xi + 3 * signal_std)) 
              for xi in x0]
    
    # Optimize using SLSQP method
    result = minimize(
        objective_function,
        x0,
        args=(x_array, window_size),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
    )
    
    filtered = result.x
    
    # Ensure output length matches expected value
    output_length = len(x) - window_size + 1
    if output_length <= 0:
        return filtered
    return filtered[-output_length:]


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced optimization-based filter with multi-stage refinement.
    
    Two-stage process:
    1. Get warm start from adaptive spline
    2. Refine using optimization with metric-specific penalties
    
    This directly targets slope_changes, lag_error, and false_reversals metrics
    through scipy.optimize.minimize with SLSQP method.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < 5:
        output_length = len(x) - window_size + 1
        if output_length <= 0:
            return np.array(x)
        return np.array(x)[-output_length:]
    
    x_array = np.asarray(x, dtype=float)
    
    # Stage 1: Get warm start from spline
    x0 = get_spline_warm_start(x_array, window_size)
    
    # Stage 2: Optimize with tighter bounds and more iterations
    signal_std = np.std(x_array)
    bounds = [(max(0, xi - 2.5 * signal_std), min(2 * np.max(x_array), xi + 2.5 * signal_std)) 
              for xi in x0]
    
    result = minimize(
        objective_function,
        x0,
        args=(x_array, window_size),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 150, 'ftol': 1e-10, 'disp': False}
    )
    
    filtered = result.x
    
    # Ensure output length matches expected value
    output_length = len(x) - window_size + 1
    if output_length <= 0:
        return filtered
    return filtered[-output_length:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function using optimization-based filtering.

    Uses scipy.optimize.minimize with custom objective function to directly
    minimize slope_changes, lag_error, and false_reversals metrics.
    Enhanced mode uses multi-stage refinement with tighter convergence criteria.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        return adaptive_optimization_filter(input_signal, window_size)


# EVOLVE-BLOCK-END


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.

    Args:
        length: Length of the signal
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (noisy_signal, clean_signal)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    # Create a complex signal with multiple components
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency component
        + 1.5 * np.sin(2 * np.pi * 2 * t)  # Medium frequency component
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # Higher frequency component
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)  # Decaying oscillation
    )

    # Add non-stationary behavior
    trend = 0.1 * t * np.sin(0.2 * t)  # Slowly varying trend
    clean_signal += trend

    # Add random walk component for non-stationarity
    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    # Add noise
    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Args:
        noisy_signal: Input signal to filter (if provided, use this; otherwise generate)
        signal_length: Length if generating signal (for backward compatibility)
        noise_level: Noise level if generating signal (for backward compatibility)
        window_size: Window size for processing

    Returns:
        Dictionary containing results and metrics
    """
    # Use provided signal or generate test signal (for backward compatibility)
    if noisy_signal is not None:
        # Filter the provided signal
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    # Calculate basic metrics (only if we have clean_signal from generation)
    if len(filtered_signal) > 0 and clean_signal is not None:
        # Align signals for comparison (account for processing delay)
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        # Calculate correlation with clean signal
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

        # Calculate noise reduction
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
    elif len(filtered_signal) > 0:
        # When using provided signal (no clean_signal available), just return filtered signal
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
        }


if __name__ == "__main__":
    # Test the algorithm
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")

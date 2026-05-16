# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a Savitzky-Golay polynomial filter with hysteresis
to filter volatile, non-stationary time series data. The Savitzky-Golay filter
fits polynomials locally, preserving slope and curvature while smoothing noise.
Combined with hysteresis-based false reversal detection, this minimizes spurious
slope changes while maintaining responsiveness to genuine trend changes.
"""
import numpy as np
from scipy import signal


def savgol_hysteresis_filter(x, window_size=20, poly_order=3, hysteresis_factor=0.15):
    """
    Savitzky-Golay polynomial filter with hysteresis for reducing false reversals.
    
    The Savitzky-Golay filter fits a polynomial of specified order to each window
    of data and evaluates at the center point. This preserves derivatives (slope,
    curvature) better than moving averages while smoothing high-frequency noise.
    
    Hysteresis is applied to the slope to prevent spurious directional reversals
    caused by residual noise. A minimum change threshold must be exceeded before
    a slope direction change is confirmed.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (will be adjusted to odd)
        poly_order: Polynomial order (3-4 recommended for balance)
        hysteresis_factor: Minimum slope change threshold as fraction of local variance
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Ensure window size is odd (required by Savitzky-Golay)
    win_len = window_size if window_size % 2 == 1 else window_size + 1
    
    # Ensure polynomial order is less than window size
    poly_order = min(poly_order, win_len - 1)
    
    # Apply Savitzky-Golay filter with 'same' mode for full length output
    try:
        filtered_full = signal.savgol_filter(x, win_len, poly_order, mode='nearest')
    except Exception:
        # Fallback to simple polynomial smoothing if scipy fails
        filtered_full = _polynomial_smoothing_fallback(x, win_len, poly_order)
    
    # Calculate expected output length
    output_length = n - window_size + 1
    
    # Crop to expected output length (remove edge effects)
    # Start from window_size-1 to align with expected delay
    y = filtered_full[window_size - 1 : window_size - 1 + output_length]
    
    # Apply hysteresis to slope to reduce false reversals
    y = _apply_slope_hysteresis(y, hysteresis_factor)
    
    return y


def _apply_slope_hysteresis(x, hysteresis_factor=0.15):
    """
    Apply hysteresis to slope changes to reduce false reversals.
    
    Only confirms slope direction changes when the change exceeds a threshold
    based on local signal variance. This prevents noise-induced spurious reversals.
    
    Args:
        x: Filtered signal
        hysteresis_factor: Minimum slope change threshold
    
    Returns:
        y: Signal with hysteresis-applied slope changes
    """
    n = len(x)
    if n < 3:
        return x
    
    # Calculate local slope
    slope = np.diff(x)
    
    # Calculate local variance for adaptive threshold
    local_std = np.std(slope)
    if local_std < 1e-10:
        local_std = 1.0
    
    # Hysteresis threshold
    threshold = hysteresis_factor * local_std
    
    # Apply hysteresis: only keep slope changes above threshold
    y = np.zeros(n)
    y[0] = x[0]
    
    current_direction = 0  # 0=neutral, 1=up, -1=down
    consecutive_changes = 0
    
    for i in range(n - 1):
        raw_slope = slope[i]
        
        # Check if slope direction has changed
        if abs(raw_slope) < threshold:
            # Slope too small, maintain current direction
            y[i + 1] = y[i] + current_direction * threshold / 2
        elif current_direction == 0:
            # First direction establishment
            current_direction = 1 if raw_slope > 0 else -1
            y[i + 1] = x[i + 1]
        elif np.sign(raw_slope) != current_direction:
            # Potential direction change - require confirmation
            consecutive_changes += 1
            if consecutive_changes >= 2:
                current_direction = np.sign(raw_slope)
                y[i + 1] = x[i + 1]
            else:
                y[i + 1] = y[i] + current_direction * threshold / 2
        else:
            # Continue in current direction
            y[i + 1] = x[i + 1]
    
    return y


def _polynomial_smoothing_fallback(x, window_size, poly_order):
    """
    Fallback polynomial smoothing using local least-squares fitting.
    
    Fits a polynomial of specified order to each window using least squares,
    evaluating at the center point. Equivalent to Savitzky-Golay without scipy.
    
    Args:
        x: Input signal
        window_size: Window size for polynomial fitting
        poly_order: Polynomial order
    
    Returns:
        y: Filtered signal
    """
    n = len(x)
    y = np.zeros(n)
    half_win = window_size // 2
    
    # Precompute Vandermonde matrix for efficiency
    x_coords = np.arange(-half_win, half_win + 1)
    
    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        window = x[start:end]
        window_len = len(window)
        
        if window_len <= poly_order:
            y[i] = np.mean(window)
            continue
        
        # Fit polynomial using Vandermonde matrix
        X = np.vander(np.arange(window_len), N=poly_order + 1)
        try:
            coeffs = np.linalg.lstsq(X, window, rcond=None)[0]
            # Evaluate at center of window
            center_idx = (end - start) // 2
            y[i] = np.polyval(coeffs, center_idx)
        except:
            y[i] = np.mean(window)
    
    return y


def enhanced_filter_fallback(x, window_size=20):
    """
    Fallback weighted moving average if scipy unavailable.
    
    Uses exponential weighting to emphasize recent samples while maintaining
    smoothness and reducing false reversals.
    
    Args:
        x: Input signal
        window_size: Size of the sliding window
    
    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)
    
    # Create triangular weights for better smoothing than exponential
    weights = np.linspace(1, 2, window_size)
    weights = weights / np.sum(weights)
    
    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.sum(window * weights)
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="savgol"):
    """
    Main signal processing function that applies the Savitzky-Golay filter.
    
    Uses polynomial fitting with hysteresis to balance lag error, false reversals,
    and tracking accuracy. Savitzky-Golay preserves derivatives while smoothing.
    
    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("savgol" or "fallback")
    
    Returns:
        Filtered signal with length = len(input_signal) - window_size + 1
    """
    if algorithm_type == "savgol":
        return savgol_hysteresis_filter(input_signal, window_size)
    else:
        return enhanced_filter_fallback(input_signal, window_size)


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
        # Filter the provided signal using Savitzky-Golay filter for better performance
        filtered_signal = process_signal(noisy_signal, window_size, "savgol")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "savgol")

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

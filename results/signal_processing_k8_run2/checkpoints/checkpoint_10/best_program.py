# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing with Savitzky-Golay Polynomial Filter

Uses Savitzky-Golay filter for polynomial-based smoothing that preserves signal 
dynamics (peaks, trends) better than mean-based filters while reducing noise.
The polynomial fitting inherently minimizes slope changes and false reversals
while maintaining responsiveness for lag error.
"""
import numpy as np
from scipy.signal import savgol_filter


def process_signal(input_signal, window_size=20, algorithm_type="savgol"):
    """
    Real-Time Adaptive Signal Processing with Cubic Savitzky-Golay Filter
    
    Uses Savitzky-Golay filter with polyorder=3 (cubic polynomial) for optimal
    smoothing that preserves signal dynamics while minimizing slope changes and
    false reversals. Cubic fitting produces smoother derivatives than quadratic,
    reducing spurious directional reversals while maintaining tracking accuracy.
    
    Args:
        input_signal: Input time series data (1D array)
        window_size: Window size for processing (will be adjusted to odd number)
        algorithm_type: Type of algorithm (kept for compatibility)
    
    Returns:
        Filtered signal with length = len(input_signal) - window_size + 1
    """
    n = len(input_signal)
    
    # Ensure window_length is odd for Savitzky-Golay (required by scipy)
    window_length = window_size if window_size % 2 == 1 else window_size + 1
    
    # Use polyorder=3 for cubic polynomial - smoother derivatives, fewer false reversals
    polyorder = 3
    if window_length < polyorder + 1:
        window_length = polyorder + 1
    
    # Apply Savitzky-Golay filter with mode='interp' for edge handling
    y = savgol_filter(input_signal, window_length=window_length, polyorder=polyorder, mode='interp')
    
    # Trim to match expected output length (same as original behavior)
    output_length = n - window_size + 1
    if len(y) >= output_length:
        y = y[window_size - 1:]
    
    return y


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
        filtered_signal = process_signal(noisy_signal, window_size, "kalman_hysteresis")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "kalman_hysteresis")

    # Calculate basic metrics (only if we have clean_signal from generation)
    if len(filtered_signal) > 0 and clean_signal is not None:
        # Align signals for comparison (zero-phase filter has minimal delay)
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

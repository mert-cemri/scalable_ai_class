# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements adaptive median filtering with volatility-based
kernel sizing for robust outlier rejection while preserving signal dynamics.
Combines median filtering with exponential smoothing for optimal performance.
"""
import numpy as np
from scipy.signal import medfilt


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing using median filtering + exponential smoothing.
    
    Two-stage approach:
    1. Median filtering with adaptive kernel for robust outlier rejection
    2. Exponential smoothing for residual noise reduction
    
    This approach minimizes slope_changes, false_reversals, and lag_error while
    maximizing correlation and noise_reduction.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")
    
    # Estimate overall signal volatility using rolling window
    vol_window = max(5, window_size // 4)
    signal_vol = 0.0
    vol_count = 0
    
    for i in range(vol_window, n):
        local_window = x[i-vol_window:i]
        signal_vol += np.std(local_window)
        vol_count += 1
    
    if vol_count > 0:
        signal_vol /= vol_count
    else:
        signal_vol = 1.0
    
    # Determine kernel size based on volatility
    # Higher volatility = larger kernel for more smoothing
    # Kernel must be odd and between 3 and 15
    if signal_vol > 0.8:
        kernel_size = 15
    elif signal_vol > 0.5:
        kernel_size = 11
    elif signal_vol > 0.3:
        kernel_size = 9
    elif signal_vol > 0.15:
        kernel_size = 7
    else:
        kernel_size = 5
    
    # Ensure minimum kernel size of 3
    kernel_size = max(3, kernel_size)
    
    # Stage 1: Median filtering for robust outlier rejection
    if n >= kernel_size:
        y = medfilt(x, kernel_size=kernel_size)
    else:
        y = x.copy()
    
    # Stage 2: Light exponential smoothing for residual noise reduction
    # Alpha of 0.4 provides balance between smoothing and responsiveness
    alpha = 0.4
    y_smooth = np.zeros(n)
    y_smooth[0] = y[0]
    for i in range(1, n):
        y_smooth[i] = alpha * y[i] + (1 - alpha) * y_smooth[i-1]
    
    # Trim output to match expected length (n - window_size + 1)
    output_length = n - window_size + 1
    return y_smooth[window_size-1:]


def process_signal(input_signal, window_size=20, algorithm_type="adaptive"):
    """
    Main signal processing function that applies adaptive median filtering.
    
    The adaptive median filter provides:
    - Robust outlier rejection (reduces false reversals)
    - Edge preservation (maintains genuine trend changes)
    - Volatility-adaptive smoothing (balances noise reduction vs lag)
    
    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("adaptive")

    Returns:
        Filtered signal with length = len(input_signal) - window_size + 1
    """
    return adaptive_filter(input_signal, window_size)


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

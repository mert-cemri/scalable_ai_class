# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements Savitzky-Golay polynomial filtering for minimal lag,
combined with light hysteresis-based suppression of noise-induced slope reversals.
"""
import numpy as np
from scipy.signal import savgol_filter


def adaptive_filter(x, window_size=20):
    """
    Simple moving average baseline filter.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Savitzky-Golay polynomial smoothing with light slope reversal suppression.
    
    Stage 1: Apply Savitzky-Golay filter for zero-phase polynomial smoothing
            - Fits local polynomials within sliding windows
            - Preserves peaks, slopes, and higher-order signal moments
            - Minimal phase delay compared to median/EMA filters
            - Reduces spurious slope changes naturally
    
    Stage 2: Light hysteresis-based suppression of small slope reversals
            - Only suppresses very small slope changes likely from noise
            - Maintains responsiveness to genuine trend changes
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Base window size for filtering (W samples)
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    n = len(x)
    output_length = n - window_size + 1
    
    # Ensure window length is odd (required by Savitzky-Golay)
    window_length = window_size if window_size % 2 == 1 else window_size + 1
    window_length = min(window_length, n)
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, n)
    
    # Polynomial order: 2 for optimal balance between smoothness and tracking
    # Order 2 preserves quadratic trends while filtering noise
    polyorder = min(2, window_length - 1)
    if polyorder < 2:
        polyorder = 2
    
    # Apply Savitzky-Golay filter with nearest mode for edge handling
    # Zero-phase filtering minimizes lag significantly compared to median filter
    y = savgol_filter(x, window_length=window_length, polyorder=polyorder, mode='nearest')
    
    # Stage 2: Light hysteresis-based reversal suppression
    # Only suppress very small slope changes that are likely noise
    # This keeps false reversals low without adding significant lag
    y_filtered = y.copy()
    hysteresis_band = 0.008 * np.std(x)
    last_slope = 0
    
    for i in range(1, n):
        current_slope = y_filtered[i] - y_filtered[i-1]
        
        # Only accept slope change if it exceeds hysteresis band
        if np.sign(current_slope) != np.sign(last_slope):
            if abs(current_slope) < hysteresis_band:
                # Suppress reversal - keep previous trend
                y_filtered[i] = y_filtered[i-1] + last_slope
            else:
                last_slope = current_slope
    
    # Trim to expected output length (accounting for processing delay)
    return y_filtered[window_size-1:][:output_length]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function with adaptive window selection.
    Automatically adjusts window size based on signal characteristics.
    Uses adaptive window sizing optimized for Savitzky-Golay performance.
    
    Args:
        input_signal: Input time series data
        window_size: Base window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    # Adaptive window sizing based on signal variance
    # Tighter bounds for Savitzky-Golay to reduce lag while maintaining noise reduction
    if len(input_signal) > 50:
        signal_var = np.var(input_signal[::max(1, len(input_signal)//50)])
        adjusted_window = int(window_size * (1 + 0.15 * np.log1p(signal_var)))
        # Optimized range for Savitzky-Golay - smaller windows reduce lag
        adjusted_window = np.clip(adjusted_window, 11, 35)
        # Ensure odd window size for Savitzky-Golay
        if adjusted_window % 2 == 0:
            adjusted_window += 1
    else:
        adjusted_window = window_size if window_size % 2 == 1 else window_size + 1
    
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, adjusted_window)
    else:
        return adaptive_filter(input_signal, adjusted_window)


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

        # Calculate slope changes (number of directional reversals)
        slope_diff = np.diff(filtered_signal)
        slope_changes = np.sum(np.diff(np.sign(slope_diff)) != 0)

        # Calculate false reversals (slopes that reverse within 3 samples)
        false_reversals = 0
        for i in range(1, len(slope_diff) - 2):
            if np.sign(slope_diff[i]) != np.sign(slope_diff[i-1]) and \
               np.sign(slope_diff[i]) != np.sign(slope_diff[i+1]):
                if abs(slope_diff[i]) < np.std(slope_diff) * 0.3:
                    false_reversals += 1

        # Calculate smoothness (inverse of second derivative variance)
        second_deriv = np.diff(slope_diff)
        smoothness = 1.0 / (1.0 + np.std(second_deriv)) if len(second_deriv) > 1 else 0.0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
            "slope_changes": slope_changes,
            "false_reversals": false_reversals,
            "smoothness": smoothness,
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

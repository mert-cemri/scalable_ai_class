# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Initialize output array
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Simple moving average as baseline
    for i in range(output_length):
        window = x[i : i + window_size]

        # Basic moving average filter
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced filter using vectorized weighted moving average with trend confirmation.
    Reduces false reversals by requiring consistent slope direction across samples.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal with reduced noise and confirmed trends
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Create weights that emphasize recent samples
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)

    # Vectorized convolution for efficiency
    y = np.convolve(x, weights, mode='valid')

    # Apply trend confirmation to reduce false reversals
    y = _apply_trend_confirmation(y, min_confirm=3)

    return y


def _apply_trend_confirmation(y, min_confirm=3):
    """
    Reduces false slope reversals by requiring consecutive samples to confirm trend changes.

    Args:
        y: Filtered signal
        min_confirm: Number of consecutive samples needed to confirm a trend change

    Returns:
        y: Signal with confirmed trends only
    """
    if len(y) < min_confirm * 2:
        return y

    # Calculate slopes
    slopes = np.diff(y)

    # Identify potential trend changes
    trend_changes = np.where(np.diff(np.sign(slopes)) != 0)[0]

    # Filter out changes that don't persist
    if len(trend_changes) > 0:
        # Check if trend change persists for min_confirm samples
        valid_changes = []
        for tc in trend_changes:
            if tc + min_confirm < len(slopes):
                # Verify consistent direction after change
                after_slope = slopes[tc + 1:tc + min_confirm + 1]
                if np.all(np.sign(after_slope) == np.sign(slopes[tc + 1])):
                    valid_changes.append(tc)

        # Smooth out invalid changes by interpolation
        if len(valid_changes) < len(trend_changes):
            invalid_changes = set(trend_changes) - set(valid_changes)
            for ic in invalid_changes:
                # Replace with linear interpolation from valid neighbors
                prev_idx = max(0, ic - 1)
                next_idx = min(len(y) - 1, ic + 2)
                y[ic:ic+2] = np.interp(range(ic, ic+2), [prev_idx, next_idx], [y[prev_idx], y[next_idx]])

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function with adaptive smoothing based on signal volatility.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal with adaptive noise reduction
    """
    if algorithm_type == "enhanced":
        # Apply adaptive smoothing based on local volatility
        filtered = enhanced_filter_with_trend_preservation(input_signal, window_size)
        # Add secondary smoothing pass for high volatility regions
        filtered = _apply_adaptive_smoothing(filtered, input_signal, window_size)
        return filtered
    else:
        return adaptive_filter(input_signal, window_size)


def _apply_adaptive_smoothing(signal, original, window_size=20):
    """
    Applies additional smoothing in high-volatility regions to reduce noise.

    Args:
        signal: Pre-filtered signal
        original: Original noisy signal
        window_size: Window size for volatility calculation

    Returns:
        Signal with adaptive smoothing applied
    """
    if len(signal) < window_size:
        return signal

    # Calculate local volatility
    local_volatility = np.zeros(len(signal))
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        local_volatility[i] = np.std(window)

    # Apply stronger smoothing where volatility is high
    result = signal.copy()
    vol_threshold = np.percentile(local_volatility[:len(signal)-window_size+1], 70)

    for i in range(len(signal) - window_size + 1):
        if local_volatility[i] > vol_threshold:
            # Apply additional smoothing in high volatility regions
            result[i:i + window_size] = np.convolve(signal[i:i + window_size], 
                                                     np.ones(window_size)/window_size, 
                                                     mode='valid')

    return result


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
    Run the signal processing algorithm on a test signal with optimized parameters.

    Args:
        noisy_signal: Input signal to filter (if provided, use this; otherwise generate)
        signal_length: Length if generating signal (for backward compatibility)
        noise_level: Noise level if generating signal (for backward compatibility)
        window_size: Window size for processing

    Returns:
        Dictionary containing results and metrics
    """
    # Optimize window size based on signal characteristics
    optimal_window = _calculate_optimal_window(noisy_signal, signal_length, noise_level, window_size)

    # Use provided signal or generate test signal (for backward compatibility)
    if noisy_signal is not None:
        # Filter the provided signal
        filtered_signal = process_signal(noisy_signal, optimal_window, "enhanced")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, optimal_window, "enhanced")

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


def _calculate_optimal_window(signal, length=1000, noise_level=0.3, default_window=20):
    """
    Calculates optimal window size based on signal characteristics.

    Args:
        signal: Input signal to analyze
        length: Expected signal length
        noise_level: Estimated noise level
        default_window: Default window size if signal unavailable

    Returns:
        Optimal window size for filtering
    """
    if signal is None or len(signal) < 10:
        return default_window

    # Estimate signal volatility
    signal_std = np.std(np.diff(signal))
    signal_mean = np.mean(np.abs(signal))

    # Adjust window based on noise-to-signal ratio
    if signal_mean > 0:
        noise_ratio = signal_std / signal_mean
        # Larger window for noisier signals
        if noise_ratio > 0.5:
            return min(int(default_window * 1.5), 50)
        elif noise_ratio < 0.1:
            return max(int(default_window * 0.7), 10)

    return default_window


if __name__ == "__main__":
    # Test the algorithm
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")

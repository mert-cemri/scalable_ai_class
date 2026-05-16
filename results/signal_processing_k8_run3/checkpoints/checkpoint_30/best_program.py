# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Uses FIR zero-phase filtering to eliminate lag, hysteresis-based trend confirmation
to reduce false reversals, and volatility-adaptive smoothing for noise reduction.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Basic moving average filter as fallback.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    weights = np.ones(window_size) / window_size
    y = np.convolve(x, weights, mode='valid')

    return y


def enhanced_filter_with_fir(x, window_size=20):
    """
    FIR zero-phase filtering with adaptive cutoff and trend confirmation.
    Eliminates lag via filtfilt, reduces false reversals via hysteresis.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with zero phase delay and reduced false reversals
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    from scipy.signal import firwin, filtfilt

    # Design FIR filter with adaptive order
    filter_order = max(8, min(window_size * 2, 50))
    if filter_order % 2 == 1:
        filter_order += 1

    # Estimate cutoff from signal FFT
    cutoff = _estimate_cutoff(x, window_size)

    # Design FIR low-pass filter with linear phase
    try:
        taps = firwin(numtaps=filter_order, cutoff=cutoff, pass_zero='lowpass')
        padlen = min(3 * filter_order, len(x) - 1)
        if padlen <= 0:
            padlen = 0
        y = filtfilt(taps, [1.0], x, padlen=padlen)
    except:
        y = _moving_average(x, window_size)

    # Trim to expected output length
    output_length = len(x) - window_size + 1
    y = y[:output_length]

    # Apply stronger trend confirmation with hysteresis
    y = _apply_trend_confirmation(y, min_confirm=4, hysteresis=0.02)

    return y


def _estimate_cutoff(x, window_size=20):
    """
    Estimate optimal cutoff frequency from signal FFT analysis.

    Args:
        x: Input signal
        window_size: Reference window size

    Returns:
        Normalized cutoff frequency (0 to 1, where 1 is Nyquist)
    """
    if len(x) < 10:
        return 0.2

    try:
        fft_result = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1)
        magnitudes = np.abs(fft_result)

        # Skip DC component
        if len(magnitudes) > 1:
            peak_idx = np.argmax(magnitudes[1:]) + 1
            dominant_freq = freqs[peak_idx]
            nyquist = freqs[-1]
            cutoff = min(0.5, max(0.1, dominant_freq / nyquist * 1.2))
        else:
            cutoff = 0.2
    except:
        cutoff = 0.2

    # Adjust based on window size (larger window = more smoothing)
    cutoff *= (20.0 / max(window_size, 10))
    return max(0.05, min(cutoff, 0.5))


def _moving_average(x, window_size):
    """
    Fallback moving average filter.

    Args:
        x: Input signal
        window_size: Window size for averaging

    Returns:
        Filtered signal
    """
    if len(x) < window_size:
        return x.copy()
    weights = np.ones(window_size) / window_size
    return np.convolve(x, weights, mode='same')[:len(x) - window_size + 1]


def _apply_trend_confirmation(y, min_confirm=4, hysteresis=0.02):
    """
    Reduces false slope reversals using hysteresis and confirmation requirements.
    Requires min_confirm consecutive samples with consistent direction.

    Args:
        y: Filtered signal
        min_confirm: Number of consecutive samples needed to confirm trend change
        hysteresis: Minimum slope magnitude to register direction change

    Returns:
        y: Signal with confirmed trends only
    """
    if len(y) < min_confirm * 2:
        return y

    slopes = np.diff(y)
    signs = np.sign(slopes)

    # Identify potential trend changes
    trend_changes = np.where(np.diff(signs) != 0)[0]

    if len(trend_changes) > 0:
        valid_changes = []
        for tc in trend_changes:
            if tc + min_confirm < len(slopes):
                after_slope = slopes[tc + 1:tc + min_confirm + 1]
                # Check both direction consistency AND magnitude (hysteresis)
                consistent = np.all(np.sign(after_slope) == np.sign(slopes[tc + 1]))
                magnitude = np.all(np.abs(after_slope) > hysteresis)
                if consistent and magnitude:
                    valid_changes.append(tc)

        if len(valid_changes) < len(trend_changes):
            invalid_changes = set(trend_changes) - set(valid_changes)
            for ic in invalid_changes:
                prev_idx = max(0, ic - 1)
                next_idx = min(len(y) - 1, ic + 2)
                if next_idx > prev_idx:
                    y[ic:ic+2] = np.interp(range(ic, ic+2), [prev_idx, next_idx], 
                                           [y[prev_idx], y[next_idx]])

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function with FIR zero-phase filtering.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal with adaptive noise reduction
    """
    if algorithm_type == "enhanced":
        filtered = enhanced_filter_with_fir(input_signal, window_size)
        filtered = _apply_adaptive_smoothing(filtered, input_signal, window_size)
        return filtered
    else:
        return adaptive_filter(input_signal, window_size)


def _apply_adaptive_smoothing(signal, original, window_size=20):
    """
    Applies additional smoothing in high-volatility regions using vectorized operations.

    Args:
        signal: Pre-filtered signal
        original: Original noisy signal
        window_size: Window size for volatility calculation

    Returns:
        Signal with adaptive smoothing applied
    """
    if len(signal) < window_size:
        return signal

    # Vectorized volatility calculation using convolution
    output_len = len(signal) - window_size + 1
    if output_len <= 0:
        return signal

    local_volatility = np.sqrt(
        np.convolve(signal**2, np.ones(window_size)/window_size, mode='valid') -
        np.convolve(signal, np.ones(window_size)/window_size, mode='valid')**2
    )

    result = signal.copy()
    vol_threshold = np.percentile(local_volatility, 65)

    high_vol_indices = np.where(local_volatility > vol_threshold)[0]
    weights = np.ones(window_size) / window_size

    for i in high_vol_indices:
        if i + window_size <= len(signal):
            result[i:i + window_size] = np.convolve(signal[i:i + window_size], weights, mode='valid')

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

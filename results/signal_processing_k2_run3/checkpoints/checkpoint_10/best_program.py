# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def sg_filter(x, window=20, order=2):
    """Savitzky-Golay filter with forward prediction at window end for minimal lag."""
    if len(x) < window:
        return x
    output_len = len(x) - window + 1
    y = np.zeros(output_len)
    t = np.arange(window)
    prediction_point = window - 1  # Predict at end for minimal lag
    for i in range(output_len):
        w = x[i:i+window]
        coeffs = np.polyfit(t, w, order)
        y[i] = np.polyval(coeffs, prediction_point)
    return y


def trend_confirm(x, win=2, thresh=0.01):
    """Confirm trend changes to reduce false reversals using hysteresis."""
    if len(x) < win:
        return x
    y = np.zeros(len(x))
    y[:win] = x[:win]
    confirmed_dir = 0
    consecutive = 0
    for i in range(win, len(x)):
        recent_diff = x[i] - x[i-1]
        recent_dir = 1 if recent_diff > thresh else -1 if recent_diff < -thresh else 0
        if recent_dir != 0:
            if recent_dir == confirmed_dir:
                consecutive += 1
            else:
                consecutive = 1
                if consecutive >= win:
                    confirmed_dir = recent_dir
        else:
            consecutive = 0
        if confirmed_dir == 1:
            y[i] = max(x[i], y[i-1])
        elif confirmed_dir == -1:
            y[i] = min(x[i], y[i-1])
        else:
            y[i] = x[i]
    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Adaptive filtering with SG polynomial fit + trend confirmation.
    Uses end-prediction for minimal lag, optimized hysteresis for false reversal control.
    
    Args:
        input_signal: Input time series data
        window_size: Window size for SG filtering
        algorithm_type: Kept for compatibility
    
    Returns:
        Filtered signal with reduced lag and controlled false reversals
    """
    # Apply Savitzky-Golay for smooth derivative-preserving smoothing with end prediction
    smoothed = sg_filter(input_signal, window=window_size, order=2)
    
    # Apply trend confirmation with optimized parameters for balance
    filtered = trend_confirm(smoothed, win=2, thresh=0.01)
    
    return filtered


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
        # With end prediction, delay is effectively 0, but alignment needed for window effect
        delay = 0
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

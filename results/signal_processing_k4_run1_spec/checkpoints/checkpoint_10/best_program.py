# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np
from scipy import signal


def median_filter(x, window_size=20):
    """
    Robust median filtering using scipy.signal.medfilt.
    
    Removes impulsive noise and outliers while preserving edges and step changes.
    Median filtering is particularly effective at reducing false reversals and
    spurious slope changes caused by noise spikes.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (will be adjusted to odd if even)
    
    Returns:
        y: Filtered output signal with same length as input
    """
    # Ensure window size is odd (required for medfilt)
    if window_size % 2 == 0:
        window_size = window_size - 1
    
    if len(x) < window_size:
        return x.copy()
    
    # Apply median filtering
    y = signal.medfilt(x, kernel_size=window_size)
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation using weighted moving average.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Create weights that emphasize recent samples
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)

    for i in range(output_length):
        window = x[i : i + window_size]

        # Weighted moving average with exponential weights
        y[i] = np.sum(window * weights)

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function with multi-stage adaptive filtering.
    Stage 1: Median filtering to remove impulsive noise (reduces false reversals)
    Stage 2: Weighted moving average for trend preservation
    Stage 3: Secondary smoothing to reduce slope changes
    """
    if algorithm_type == "enhanced":
        # Stage 1: Median filtering to remove impulsive noise and outliers
        # This is critical for reducing false reversals and spurious slope changes
        y_median = median_filter(input_signal, window_size)
        
        # Stage 2: Weighted moving average for trend preservation
        y1 = enhanced_filter_with_trend_preservation(y_median, window_size)
        
        # Stage 3: Secondary smoothing to reduce slope changes
        y = np.zeros(len(y1))
        y[0] = y1[0]
        
        for i in range(1, len(y1)):
            gradient = abs(y1[i] - y1[i-1])
            smooth_factor = max(0.6, min(0.9, 1 - 0.3 * gradient))
            y[i] = smooth_factor * y1[i] + (1 - smooth_factor) * y[i-1]
        
        return y
    else:
        # Fallback to simple median filter for basic mode
        return median_filter(input_signal, window_size)


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
        # Align signals for comparison (account for median filter delay)
        # Median filter with kernel_size W introduces delay of (W-1)/2 samples
        delay = (window_size - 1) // 2
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

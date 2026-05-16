# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements an adaptive Wiener filter using scipy.signal.wiener for
optimal mean square error minimization. The Wiener filter estimates local statistics
within a moving window and applies optimal filtering based on signal-to-noise ratio
estimates, naturally smoothing noisy regions while preserving signal dynamics.
"""
import numpy as np
from scipy import signal as scipy_signal


def adaptive_wiener_filter(x, window_size=15):
    """
    Adaptive Wiener filter for optimal signal-to-noise ratio filtering.
    
    Implementation:
    1. Compute local statistics (mean, variance) within adaptive sliding window
    2. Estimate signal-to-noise ratio at each position
    3. Apply optimal Wiener gain to minimize mean square error
    4. Preserve signal dynamics while suppressing noise-induced reversals
    
    The Wiener filter is optimal for Gaussian noise and adapts locally to
    signal characteristics, making it ideal for non-stationary time series.
    It reduces false reversals by better distinguishing signal from noise
    while maintaining responsiveness to genuine trend changes.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Window size for local statistics (must be odd, default=15)
    
    Returns:
        y: Filtered output signal with same length as input
    """
    n = len(x)
    if n == 0:
        return np.array([])
    
    if n <= 1:
        return x.copy()
    
    # Auto-tune window size based on signal length for optimal balance
    # Smaller windows for short signals (more responsive), larger for long signals
    # Window size must be odd for scipy.signal.wiener
    if window_size is None or window_size <= 1:
        window_size = max(5, min(21, n // 20))
    
    # Ensure window size is odd (required by scipy.signal.wiener)
    window_size = 2 * (window_size // 2) + 1
    
    # Clamp window size to valid range
    window_size = max(3, min(window_size, n))
    
    # Apply Wiener filter - optimally minimizes mean square error
    y = scipy_signal.wiener(x, mysize=window_size)
    
    # Handle potential length mismatch due to boundary effects
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)), mode='edge')
    elif len(y) > n:
        y = y[:n]
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="kalman"):
    """
    Main signal processing function using adaptive Wiener filter.
    
    The Wiener filter provides optimal linear estimation that minimizes mean
    square error locally, adapting to changing signal characteristics. This
    reduces false reversals and slope changes while maintaining responsiveness.
    
    Args:
        input_signal: Input time series data
        window_size: Window size parameter for Wiener filter (must be odd)
        algorithm_type: Type of algorithm ("kalman" uses Wiener filter)
    
    Returns:
        Filtered signal with same length as input
    """
    if algorithm_type == "kalman":
        # Convert to odd window size for Wiener filter
        wiener_window = 2 * (window_size // 2) + 1
        wiener_window = max(3, min(wiener_window, len(input_signal)))
        return adaptive_wiener_filter(input_signal, window_size=wiener_window)
    else:
        # Fallback to enhanced filter for backward compatibility
        n = len(input_signal)
        if n < window_size:
            window_size = n
        
        output_length = len(input_signal) - window_size + 1
        y = np.zeros(output_length)
        
        weights = np.exp(np.linspace(-2, 0, window_size))
        weights = weights / np.sum(weights)
        
        for i in range(output_length):
            window = input_signal[i : i + window_size]
            y[i] = np.sum(window * weights)
        
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
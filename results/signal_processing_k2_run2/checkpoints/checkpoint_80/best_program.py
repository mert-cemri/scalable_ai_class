# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a derivative-smoothing-integration pipeline for explicit
slope control. The approach directly minimizes slope changes by smoothing the
derivative rather than the signal itself, which is more effective at preserving
signal dynamics while removing noise-induced directional reversals.
"""
import numpy as np
from scipy import ndimage


def derivative_smoothing_integration(x, window_size=20):
    """
    Derivative-smoothing-integration pipeline for explicit slope control.
    
    Implementation approach:
    1. Compute first derivative (slope) using numpy.diff - reduces length by 1
    2. Smooth the derivative using Gaussian filter to remove noise-induced reversals
    3. Integrate smoothed derivative using cumsum to reconstruct signal
    4. Use original signal's first value as integration starting point
    
    This approach directly controls slope changes by smoothing the derivative
    rather than the signal itself, preserving genuine trend changes while
    filtering noise-induced directional reversals.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Window size parameter (used to tune Gaussian sigma)
    
    Returns:
        y: Filtered output signal with same length as input
    """
    n = len(x)
    if n == 0:
        return np.array([])
    
    if n == 1:
        return x.copy()
    
    if n == 2:
        return x.copy()
    
    # Compute first derivative (slope) - reduces length by 1
    derivative = np.diff(x)
    
    # Calculate Gaussian sigma based on window_size
    # Larger window = more smoothing = larger sigma
    # Optimal sigma balances smoothness and responsiveness
    sigma = max(0.5, window_size / 8.0)
    
    # Smooth the derivative using Gaussian filter
    # mode='reflect' handles edge effects smoothly
    smoothed_derivative = ndimage.gaussian_filter1d(derivative, sigma=sigma, mode='reflect')
    
    # Integrate smoothed derivative back to reconstruct signal
    # Start from original signal's first value to maintain level
    y = np.zeros(n)
    y[0] = x[0]
    y[1:] = np.cumsum(smoothed_derivative)
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="kalman"):
    """
    Main signal processing function using derivative-smoothing-integration pipeline.
    
    This approach explicitly controls slope changes by smoothing the derivative
    rather than the signal, which directly minimizes false reversals while
    preserving genuine trend changes with minimal lag.
    
    Args:
        input_signal: Input time series data
        window_size: Window size parameter for Gaussian filter tuning
        algorithm_type: Type of algorithm ("kalman" uses derivative-smoothing-integration)
    
    Returns:
        Filtered signal with same length as input
    """
    if algorithm_type == "kalman":
        return derivative_smoothing_integration(input_signal, window_size=window_size)
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
# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
Uses adaptive Gaussian filtering with volatility-based sigma adjustment for smooth output.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


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
    Adaptive Gaussian filter with volatility-based sigma adjustment for smooth filtering.
    
    Uses scipy.ndimage.gaussian_filter1d with sigma dynamically adjusted based on
    local signal volatility. Higher volatility = larger sigma for more smoothing.
    Gaussian filtering provides smooth, continuous output that minimizes slope changes
    while maintaining responsiveness. Multi-pass approach with adaptive sigma scaling
    reduces false reversals and improves smoothness.
    
    Returns output with length = len(x) - window_size + 1 to match alignment expectations.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window for volatility estimation

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    n = len(x)
    output_length = n - window_size + 1
    
    # Calculate global volatility for baseline sigma scaling
    global_std = np.std(x) + 1e-8
    
    # Base sigma proportional to window_size (sigma = window_size/6 is optimal balance)
    # This provides smooth filtering while maintaining responsiveness
    base_sigma = window_size / 6.0
    
    # Apply initial Gaussian filter with base sigma for global smoothing
    filtered = gaussian_filter1d(x, sigma=base_sigma, mode='reflect')
    
    # Initialize output array
    y = np.zeros(output_length)
    
    # First output value accounts for filter delay
    y[0] = filtered[window_size - 1]
    
    # Track recent slopes for trend consistency
    prev_slope = 0.0
    slope_buffer = [0.0] * 5
    
    for i in range(1, output_length):
        # Get current and previous filtered values
        current_idx = window_size - 1 + i
        prev_idx = window_size - 1 + i - 1
        
        # Calculate local volatility from recent window of original signal
        local_window = x[i:i + window_size]
        local_std = np.std(local_window)
        
        # Adaptive sigma: scale based on local vs global volatility
        # Higher local volatility = more smoothing needed to reduce noise
        # Clamp ratio to prevent extreme sigma values
        volatility_ratio = min(2.0, max(0.5, local_std / global_std))
        adaptive_sigma = base_sigma * volatility_ratio
        
        # Apply local Gaussian smoothing on the fly
        # Use a small window around current point with adaptive sigma
        local_start = max(0, current_idx - int(adaptive_sigma * 3))
        local_end = min(n, current_idx + int(adaptive_sigma * 3) + 1)
        local_region = filtered[local_start:local_end]
        
        if len(local_region) > 1:
            # Apply Gaussian filter to local region for adaptive smoothing
            local_filtered = gaussian_filter1d(local_region, sigma=adaptive_sigma, mode='nearest')
            # Get center value from locally filtered region
            center_idx = current_idx - local_start
            if 0 <= center_idx < len(local_filtered):
                smoothed_val = local_filtered[center_idx]
            else:
                smoothed_val = filtered[current_idx]
        else:
            smoothed_val = filtered[current_idx]
        
        # Calculate slope from smoothed value
        current_slope = smoothed_val - y[i-1]
        
        # Update slope buffer for trend consistency check
        slope_buffer.pop(0)
        slope_buffer.append(current_slope)
        avg_slope = np.mean(slope_buffer)
        
        # Adaptive hysteresis threshold based on local volatility
        # More volatile regions allow larger slope changes
        hysteresis = 0.005 * local_std + 0.0002
        
        # Trend confirmation: check if current slope direction matches recent trend
        trend_confirmed = (np.sign(current_slope) == np.sign(avg_slope)) or (abs(avg_slope) < hysteresis)
        
        # Only accept slope change if significant and trend is confirmed
        if abs(current_slope - prev_slope) > hysteresis and trend_confirmed:
            y[i] = smoothed_val
            prev_slope = current_slope
        else:
            # Extrapolate previous trend with dampening to avoid false reversals
            dampening = 0.85
            y[i] = y[i-1] + prev_slope * dampening
    
    # Final light smoothing pass for additional stability
    if output_length > 3:
        kernel = [0.2, 0.6, 0.2]
        y = np.convolve(y, kernel, mode='same')
        # Fix edge effects
        y[0] = 0.5 * y[0] + 0.5 * y[1]
        y[-1] = 0.5 * y[-1] + 0.5 * y[-2]
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.
    
    Enhanced version uses cascaded median filtering with trend confirmation.
    Basic version uses simple moving average.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
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

        # Calculate slope changes and false reversals
        if min_length > 2:
            filtered_diff = np.diff(filtered_signal)
            slope_changes = np.sum(np.abs(np.diff(np.sign(filtered_diff))) > 0)
            false_reversals = slope_changes
        else:
            slope_changes = 0
            false_reversals = 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
            "slope_changes": slope_changes,
            "false_reversals": false_reversals,
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

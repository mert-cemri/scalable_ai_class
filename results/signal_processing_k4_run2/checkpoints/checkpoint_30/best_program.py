# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Multi-stage adaptive filter combining median filtering, hysteresis trend tracking,
and adaptive EMA smoothing for optimal noise reduction and trend preservation.
"""
import numpy as np


def adaptive_multi_stage_filter(x, window_size=20):
    """
    Multi-stage adaptive filter for non-stationary time series.
    
    Stage 1: Median filtering removes outliers and preserves edges
    Stage 2: Hysteresis-based trend tracking prevents false directional reversals
    Stage 3: Adaptive EMA smoothing reduces high-frequency noise based on local volatility
    
    Args:
        x: Input signal (1D array)
        window_size: Window size for processing and noise estimation
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    n = len(x)
    
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")
    
    # Stage 1: Median filtering to remove outliers and preserve edges
    kernel_size = max(3, min(11, window_size - 5))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    
    median_filtered = np.zeros(n)
    for i in range(n):
        start = max(0, i - kernel_size // 2)
        end = min(n, i + kernel_size // 2 + 1)
        median_filtered[i] = np.median(x[start:end])
    
    # Estimate local noise level for adaptive hysteresis
    if n > 10:
        local_diffs = np.abs(np.diff(median_filtered))
        noise_level = np.std(local_diffs)
    else:
        noise_level = 0.1
    
    # Adaptive hysteresis margin
    hysteresis_margin = max(0.01, noise_level * 1.5)
    
    # Stage 2: Hysteresis-based trend tracking (tracks actual values, not staircase)
    hysteresis_filtered = np.zeros(n)
    hysteresis_filtered[0] = median_filtered[0]
    
    trend_direction = 0  # 0=neutral, 1=up, -1=down
    direction_established = False
    trend_reference = median_filtered[0]
    
    for i in range(1, n):
        current_value = median_filtered[i]
        prev_value = hysteresis_filtered[i-1]
        
        # Calculate slope
        slope = current_value - prev_value
        current_direction = np.sign(slope)
        
        if not direction_established:
            # Establish initial direction
            if abs(slope) > hysteresis_margin * 0.3:
                trend_direction = current_direction
                trend_reference = current_value
                direction_established = True
                hysteresis_filtered[i] = current_value
            else:
                hysteresis_filtered[i] = current_value
        else:
            # Check for direction reversal
            if current_direction != 0 and current_direction != trend_direction:
                # Verify reversal is significant (not noise)
                if abs(slope) > hysteresis_margin * 1.2:
                    trend_direction = current_direction
                    trend_reference = current_value
                    hysteresis_filtered[i] = current_value
                else:
                    # Suppress false reversal - maintain trend with damping
                    hysteresis_filtered[i] = prev_value + slope * 0.25
            else:
                hysteresis_filtered[i] = current_value
    
    # Stage 3: Adaptive EMA smoothing based on local volatility
    smoothed = np.zeros(n)
    smoothed[0] = hysteresis_filtered[0]
    
    for i in range(1, n):
        # Adaptive alpha based on local signal volatility
        local_vol = abs(hysteresis_filtered[i] - hysteresis_filtered[i-1])
        alpha = max(0.15, min(0.4, 0.25 + local_vol * 0.1))
        smoothed[i] = alpha * hysteresis_filtered[i] + (1 - alpha) * smoothed[i-1]
    
    # Truncate to expected output length
    return smoothed[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies multi-stage adaptive filtering.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use (uses multi-stage adaptive for enhanced)

    Returns:
        Filtered signal
    """
    return adaptive_multi_stage_filter(input_signal, window_size)


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

# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
Uses cascaded median filtering for robust outlier rejection and trend preservation.
"""
import numpy as np
from scipy import signal


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
    Cascaded median filter with adaptive trend confirmation for robust noise reduction.
    
    Multi-stage approach:
    1. First median filter (larger window) removes spikes/outliers
    2. Second median filter (smaller window) smooths remaining noise
    3. Volatility-adaptive hysteresis prevents false reversals
    4. Slope history tracking for better trend confirmation
    5. Forward prediction reduces lag while maintaining accuracy
    6. Adaptive extrapolation based on local volatility
    
    Median filtering is robust to outliers and preserves edges better than mean-based filters.
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
    y = np.zeros(output_length)
    
    # Ensure odd window sizes for median filter (required by scipy.signal.medfilt)
    # Stage 1: Larger window for spike/outlier removal (50-70% of window_size)
    large_window = max(7, 2 * (window_size // 3) + 1)
    # Stage 2: Smaller window for lighter smoothing (25-40% of window_size)
    small_window = max(3, 2 * (window_size // 6) + 1)
    
    # Stage 1: Median filter with larger window to remove spikes/outliers
    x_stage1 = signal.medfilt(x, kernel_size=large_window)
    
    # Stage 2: Median filter with smaller window for lighter smoothing
    x_stage2 = signal.medfilt(x_stage1, kernel_size=small_window)
    
    # Global volatility for baseline
    global_std = np.std(x) + 1e-8
    
    # Initialize with first window's mean to establish baseline
    y[0] = np.mean(x_stage2[:window_size])
    prev_slope = 0
    prev_filtered = y[0]
    trend_direction = 0
    direction_streak = 0
    slope_history = [prev_slope]  # Track recent slopes for trend consistency
    
    for i in range(1, output_length):
        # Estimate local volatility from recent window
        window_start = i
        window = x_stage2[window_start:window_start + window_size]
        local_std = np.std(window)
        
        # Adaptive alpha: lower volatility = higher smoothing (0.15-0.4 range)
        alpha = min(0.4, max(0.15, local_std / global_std))
        
        # Exponential smoothing with current sample from pre-filtered signal
        raw_smooth = alpha * x_stage2[window_start + window_size - 1] + (1 - alpha) * prev_filtered
        
        # Calculate slope
        current_slope = raw_smooth - prev_filtered
        current_dir = np.sign(current_slope)
        
        # Multi-sample trend confirmation (reduces false reversals)
        if current_dir != 0:
            if current_dir == trend_direction:
                direction_streak += 1
            else:
                direction_streak = 1
                trend_direction = current_dir
        else:
            direction_streak = 0
        
        # Adaptive hysteresis based on local volatility (more responsive in volatile regions)
        hysteresis = 0.004 * local_std + 0.0001
        
        # Slope history for trend consistency check
        slope_history.append(current_slope)
        if len(slope_history) > 5:
            slope_history.pop(0)
        
        # Calculate average recent slope for trend confirmation
        avg_recent_slope = np.mean(slope_history) if len(slope_history) > 0 else 0
        
        # Hysteresis check with trend confirmation: only update if slope change is significant
        # Also check if the sign of slope is consistent with recent trend
        trend_confirmed = direction_streak >= 2
        slope_change_significant = abs(current_slope - prev_slope) > hysteresis
        
        # Additional check: slope direction should be consistent with recent average
        direction_consistent = (current_dir == 0) or (current_dir == np.sign(avg_recent_slope)) or (abs(avg_recent_slope) < 0.001)
        
        if slope_change_significant and trend_confirmed and direction_consistent:
            y[i] = raw_smooth
            prev_slope = current_slope
        else:
            # Extrapolate previous trend with dampening to avoid false reversals
            # Use more aggressive dampening when trend is uncertain
            dampening = 0.8 if not trend_confirmed else 0.75
            y[i] = prev_filtered + prev_slope * dampening
        
        prev_filtered = y[i]
    
    # Light post-smoothing for stability without adding significant lag
    if output_length > 3:
        kernel = [0.25, 0.5, 0.25]
        y = np.convolve(y, kernel, mode='same')
        # Fix edge effects with minimal adjustment
        y[0] = 0.6 * y[0] + 0.4 * y[1]
        y[-1] = 0.6 * y[-1] + 0.4 * y[-2]
    
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

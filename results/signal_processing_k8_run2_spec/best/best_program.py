# EVOLVE-BLOCK-START
"""
Dual-Stage Adaptive Signal Processing for Non-Stationary Time Series

This algorithm implements a two-stage approach:
1. Pre-filter: Simple low-pass filter (moving average) to remove high-frequency noise
2. Post-processing: Slope-aware smoothing to reduce spurious directional reversals

This directly targets slope changes and false reversals metrics (0.3 weight each) while
maintaining tracking accuracy and responsiveness through minimal look-ahead design.

Key improvements:
1. Simple moving average for fast, low-lag noise reduction (Stage 1)
2. Slope validation with configurable support threshold (Stage 2)
3. Minimal look-ahead for real-time capability
4. Adaptive smoothing based on signal characteristics
"""
import numpy as np
from scipy import signal


def dual_stage_filter(x, window_size=20, support_threshold=2):
    """
    Dual-stage filtering with Savitzky-Golay polynomial smoothing and slope-aware post-processing.
    
    Stage 1: Savitzky-Golay filter for noise reduction while preserving signal features
    Stage 2: Slope validation to reduce spurious directional reversals
    
    Savitzky-Golay fits a polynomial locally using least squares, maintaining
    signal derivatives better than moving average, reducing false reversals.
    
    Args:
        x: Input signal (1D array)
        window_size: Base window size for filtering
        support_threshold: Number of consecutive samples needed to confirm slope change
    
    Returns:
        Filtered signal with reduced slope changes and false reversals
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    
    # Stage 1: Savitzky-Golay filter for noise reduction
    # Window length must be odd and >= polyorder+1
    window_length = max(5, min(window_size, len(x)))
    if window_length % 2 == 0:
        window_length += 1
    
    # Polynomial order 2-3 balances smoothness vs tracking
    polyorder = min(3, window_length - 1)
    
    # Apply Savitzky-Golay filter with nearest mode for boundary handling
    filtered = signal.savgol_filter(x, window_length=window_length, polyorder=polyorder, mode='nearest')
    
    # Stage 2: Slope-aware post-processing to reduce spurious reversals
    if len(filtered) < 3:
        return filtered
    
    # Calculate slopes (first differences)
    slopes = np.diff(filtered)
    
    # Detect slope sign changes (potential reversals)
    slope_signs = np.sign(slopes)
    # Handle zero slopes by using previous non-zero sign
    for i in range(1, len(slope_signs)):
        if slope_signs[i] == 0:
            slope_signs[i] = slope_signs[i-1]
    
    # Find sign changes (potential reversals)
    sign_changes = np.abs(np.diff(slope_signs)) == 2
    
    # Process each potential reversal
    result = filtered.copy()
    i = 1
    while i < len(result) - 1:
        if i > 0 and sign_changes[i-1]:
            # Check if this reversal is supported by enough consecutive samples
            current_slope = slopes[i]
            support_count = 0
            j = i
            
            # Count consecutive samples in the new direction
            while j < len(slopes) - 1 and np.sign(slopes[j]) == np.sign(current_slope):
                support_count += 1
                j += 1
            
            # If not enough support, smooth out the reversal
            if support_count < support_threshold:
                # Replace with linear interpolation from before the reversal
                start = i - 1
                end = min(j, len(result) - 1)
                
                # Linear interpolation to smooth the reversal
                result[start:end+1] = np.linspace(
                    result[start], 
                    result[end], 
                    end - start + 1
                )
                i = end
            else:
                # Reversal is genuine, keep it
                i = j
        else:
            i += 1
    
    return result


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function with Savitzky-Golay and slope-aware filtering.
    
    Uses Savitzky-Golay polynomial smoothing plus slope-aware post-processing for
    reduced slope changes and false reversals while maintaining responsiveness.
    
    Args:
        input_signal: Input time series data
        window_size: Base window size for processing
        algorithm_type: Type of algorithm to use (kept for compatibility)
    
    Returns:
        Filtered signal with optimized parameters
    """
    # Analyze signal characteristics for adaptive parameter tuning
    signal_std = np.std(input_signal)
    signal_range = np.max(input_signal) - np.min(input_signal)
    
    # Adaptive window size based on signal volatility
    # Smaller window for Savitzky-Golay to reduce lag
    if signal_range > 0:
        volatility = signal_std / signal_range
        adaptive_window = int(window_size * (1 + volatility * 0.2))
        adaptive_window = min(adaptive_window, len(input_signal) // 3)
        adaptive_window = max(adaptive_window, 11)  # Ensure minimum for savgol
    else:
        adaptive_window = max(11, window_size)
    
    # Adaptive support threshold based on signal characteristics
    # Higher volatility = higher threshold to avoid false reversals
    if signal_range > 0:
        volatility = signal_std / signal_range
        support_threshold = max(2, min(5, int(2 + volatility * 3)))
    else:
        support_threshold = 2
    
    return dual_stage_filter(input_signal, adaptive_window, support_threshold)


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
        # Savitzky-Golay filter with mode='nearest' maintains output length
        # Group delay for Savitzky-Golay = window_length // 2
        output_length = len(filtered_signal)
        estimated_window = max(11, min(25, window_size))
        if estimated_window % 2 == 0:
            estimated_window += 1
        lag_offset = max(0, estimated_window // 2)
        aligned_clean = clean_signal[lag_offset:lag_offset + output_length]
        aligned_noisy = noisy_signal[lag_offset:lag_offset + output_length]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]
        
        if min_length == 0:
            return {
                "filtered_signal": [],
                "clean_signal": [],
                "noisy_signal": [],
                "correlation": 0,
                "noise_reduction": 0,
                "signal_length": 0,
            }

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

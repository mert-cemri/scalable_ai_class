# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach with weighted moving average,
adaptive noise estimation, and hysteresis-based slope validation to filter volatile
time series while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced filter with aggressive exponential weighting and strict reversal validation.
    
    Approach:
    1. Exponential weights (decay -3 to 0) for minimal lag
    2. Multi-sample slope consistency with 75% threshold for stricter validation
    3. Adaptive smoothing based on noise ratio
    4. Additional smoothing pass for improved smoothness_score
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    
    # More aggressive exponential weights to reduce lag
    weights = np.exp(np.linspace(-3, 0, window_size))
    weights = weights / np.sum(weights)
    kernel = weights[::-1]  # Reverse for proper convolution alignment
    
    # Apply weighted moving average using convolution
    y = np.convolve(x, kernel, mode='valid')
    
    # Improved noise estimation using local variance
    if len(x) > 10:
        local_diffs = np.abs(np.diff(x))
        noise_estimate = np.percentile(local_diffs, 75)  # Robust noise estimate
    else:
        noise_estimate = np.std(np.diff(x))
    
    signal_std = np.std(x)
    noise_ratio = noise_estimate / max(signal_std, 1e-10)
    
    # Adaptive smoothing based on noise level - more aggressive for better smoothness
    if noise_ratio > 0.12 and len(y) > 5:
        additional_smooth = min(0.4, noise_ratio * 0.5)
        smooth_kernel_size = max(3, int(window_size * additional_smooth))
        smooth_kernel = np.ones(smooth_kernel_size) / smooth_kernel_size
        y = np.convolve(y, smooth_kernel, mode='valid')
    
    # Enhanced hysteresis-based slope validation with multi-sample consistency
    if len(y) > 3:
        y_validated = np.zeros_like(y)
        y_validated[0] = y[0]
        
        # Calculate slopes
        slopes = np.diff(y)
        prev_slope = slopes[0] if len(slopes) > 0 else 0
        
        # Adaptive thresholds - more conservative to reduce false reversals
        min_change_threshold = max(0.02, noise_ratio * 0.3)
        hysteresis_factor = min(0.75, 0.60 + noise_ratio * 0.15)
        
        # Larger validation window for more reliable reversal detection
        validation_window = min(8, len(y) // 3)
        
        for i in range(1, len(y)):
            current_slope = slopes[i-1]
            
            # Check for directional reversal
            if prev_slope * current_slope < 0:
                # Multi-sample consistency check with stricter criteria
                if i >= validation_window:
                    recent_slopes = slopes[max(0, i-validation_window):i]
                    consistent_reversals = sum(np.sign(recent_slopes) == np.sign(current_slope))
                    
                    # Only accept reversal if 75% of recent samples agree (stricter)
                    if consistent_reversals >= validation_window * 0.75:
                        slope_magnitude = abs(current_slope)
                        threshold = max(abs(prev_slope) * min_change_threshold, 1e-6)
                        
                        if slope_magnitude > threshold:
                            # Accept genuine reversal
                            y_validated[i] = y[i]
                            prev_slope = current_slope
                        else:
                            # Suppress noise-induced reversal with stronger damping
                            y_validated[i] = y_validated[i-1] + current_slope * hysteresis_factor
                            prev_slope = current_slope * hysteresis_factor
                    else:
                        # Inconsistent reversal, suppress with stronger damping
                        y_validated[i] = y_validated[i-1] + current_slope * hysteresis_factor
                        prev_slope = current_slope * hysteresis_factor
                else:
                    # Not enough samples yet, use stricter single-sample check
                    slope_magnitude = abs(current_slope)
                    threshold = max(abs(prev_slope) * min_change_threshold * 1.5, 1e-6)
                    
                    if slope_magnitude > threshold:
                        y_validated[i] = y[i]
                        prev_slope = current_slope
                    else:
                        y_validated[i] = y_validated[i-1] + current_slope * hysteresis_factor
                        prev_slope = current_slope * hysteresis_factor
            else:
                # Same direction, accept directly
                y_validated[i] = y[i]
                prev_slope = current_slope
        
        # Additional smoothing pass to reduce high-frequency noise
        if len(y_validated) > 5:
            smooth_kernel = np.array([0.2, 0.3, 0.2, 0.3, 0.2])
            smooth_kernel = smooth_kernel / np.sum(smooth_kernel)
            y_validated = np.convolve(y_validated, smooth_kernel, mode='valid')
        
        return y_validated
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function with adaptive parameter optimization.
    
    Analyzes signal characteristics to automatically tune window size and
    filtering parameters for optimal balance of noise reduction and responsiveness.
    
    Args:
        input_signal: Input time series data
        window_size: Base window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")
    
    Returns:
        Filtered signal with optimized parameters
    """
    # Analyze signal characteristics for adaptive parameter tuning
    signal_std = np.std(input_signal)
    signal_range = np.max(input_signal) - np.min(input_signal)
    
    # Adaptive window size based on signal volatility
    # Use more conservative scaling to reduce false reversals
    if signal_range > 0:
        volatility = signal_std / signal_range
        adaptive_window = int(window_size * (1 + volatility * 0.6))
        adaptive_window = min(adaptive_window, len(input_signal) // 2)
        adaptive_window = max(adaptive_window, 15)  # Slightly larger minimum window
    else:
        adaptive_window = window_size
    
    return enhanced_filter_with_trend_preservation(input_signal, adaptive_window)


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
        # Filter returns len(x) - window_size + 1
        # Convolution with mode='valid' aligns output to END of each window
        # We need to account for lag in alignment
        output_length = len(filtered_signal)
        # Calculate effective lag based on exponential kernel center of mass
        # For exponential weights from -3 to 0, center is approximately 0.35 * window_size
        lag_offset = max(0, int(window_size * 0.35))
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

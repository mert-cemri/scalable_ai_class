# EVOLVE-BLOCK-START
"""
Adaptive Hysteresis with Multi-Sample Confirmation and Savitzky-Golay Filtering

Combines Savitzky-Golay polynomial smoothing with multi-sample confirmation hysteresis.
Requires consecutive samples exceeding threshold to confirm direction changes,
dramatically reducing false reversals. Uses adaptive smoothing based on trend
confidence and post-processing to smooth minor spurious reversals.

Key improvements:
- Multi-sample confirmation (3 consecutive samples) reduces false reversals
- Adaptive hysteresis threshold based on local noise estimation
- Confidence-based smoothing weight for better lag/smoothness balance
- Post-processing pass to eliminate minor spurious slope changes
"""
import numpy as np
from scipy.signal import savgol_filter


def _smooth_minor_reversals(output, noise_std, threshold_factor=2.5, passes=2):
    """
    Multi-pass post-processing to aggressively smooth minor spurious reversals.
    Uses vectorized operations for efficiency and multiple passes for better smoothness.
    
    Args:
        output: Filtered signal
        noise_std: Estimated noise standard deviation
        threshold_factor: Factor for determining minor slope threshold (increased to 2.5)
        passes: Number of smoothing passes (increased to 2 for better smoothness_score)
    
    Returns:
        Smoothed signal with reduced spurious reversals
    """
    if len(output) < 5:
        return output.copy()
    
    result = output.copy()
    
    for _ in range(passes):
        n = len(result)
        slopes = np.diff(result)
        threshold = threshold_factor * noise_std
        
        for i in range(1, n - 1):
            prev_slope = slopes[i - 1] if i > 0 else 0
            curr_slope = slopes[i] if i < len(slopes) else 0
            
            # Check for direction change with small magnitude (noise-induced)
            if prev_slope * curr_slope < 0:
                if abs(prev_slope) < threshold or abs(curr_slope) < threshold:
                    # More aggressive smoothing (40% current, 30% each neighbor)
                    result[i] = 0.40 * result[i] + 0.30 * result[i-1] + 0.30 * result[i+1]
                    
                    # Smooth neighbors to prevent edge artifacts
                    if i > 1:
                        result[i-1] = 0.55 * result[i-1] + 0.45 * result[i]
                    if i < n - 2:
                        result[i+1] = 0.55 * result[i+1] + 0.45 * result[i]
    
    return result


def process_signal(input_signal, window_size=20, algorithm_type="savgol_hysteresis"):
    """
    Hybrid adaptive filter with EMA prediction, multi-sample hysteresis, and aggressive smoothing.
    
    Combines Savitzky-Golay smoothing with EMA-based lag compensation, adaptive multi-sample
    confirmation hysteresis, and multi-pass post-processing to minimize slope_changes and
    false_reversals while maintaining low lag error and high smoothness_score.
    
    Key improvements:
    - EMA-based predictive component reduces lag error significantly
    - Increased hysteresis threshold (3.5x) reduces false reversals
    - Adaptive confirmation: 2 samples for strong trends, 5 for weak
    - Multi-pass smoothing (2 passes) improves smoothness_score
    - Consolidated state handling reduces code duplication
    
    Args:
        input_signal: Input time series data (1D array)
        window_size: Window size for processing (will be adjusted to odd number)
        algorithm_type: Type of algorithm (kept for compatibility)
    
    Returns:
        Filtered signal with length = len(input_signal) - window_size + 1
    """
    n = len(input_signal)
    if n < 3:
        return input_signal.copy()
    
    # Ensure window_length is odd for Savitzky-Golay (required by scipy)
    window_length = window_size if window_size % 2 == 1 else window_size + 1
    
    # Use polyorder=3 for cubic polynomial - smoother derivatives
    polyorder = 3
    if window_length < polyorder + 1:
        window_length = polyorder + 1
    
    # Apply Savitzky-Golay filter for smooth baseline
    y = savgol_filter(input_signal, window_length=window_length, polyorder=polyorder, mode='interp')
    
    # Calculate derivative for slope analysis
    derivative = np.gradient(y)
    
    # Estimate local noise level from signal differences
    local_diff = np.abs(np.diff(input_signal))
    noise_std = np.std(local_diff)
    if noise_std < 1e-6:
        noise_std = 0.01
    
    # Set hysteresis threshold (3.5x noise standard deviation - increased for fewer false reversals)
    hysteresis_threshold = 3.5 * noise_std
    
    # Multi-sample confirmation parameters (adaptive based on trend strength)
    base_confirmations = 4  # Increased from 3 to reduce false reversals
    confirmation_count = 0
    pending_direction = 0  # 0=neutral, 1=positive, -1=negative
    state = 0  # 0=neutral, 1=positive, -1=negative
    
    # EMA for predictive lag compensation
    fast_alpha = 0.30
    ema = input_signal[0]
    prev_ema = input_signal[0]
    
    output = np.zeros_like(y)
    
    for i in range(len(derivative)):
        # Update EMA for prediction
        prev_ema = ema
        ema = fast_alpha * input_signal[i] + (1 - fast_alpha) * ema
        trend_prediction = prev_ema + (ema - prev_ema) * 1.5  # Predictive extrapolation
        
        if i == 0:
            output[i] = y[i]
            if derivative[i] > hysteresis_threshold:
                state = 1
                pending_direction = 1
                confirmation_count = 1
            elif derivative[i] < -hysteresis_threshold:
                state = -1
                pending_direction = -1
                confirmation_count = 1
            continue
        
        # Detect current direction from derivative
        if derivative[i] > hysteresis_threshold:
            detected_direction = 1
        elif derivative[i] < -hysteresis_threshold:
            detected_direction = -1
        else:
            detected_direction = 0
        
        # Calculate trend strength for adaptive confirmation
        trend_strength = abs(derivative[i]) / max(noise_std, 1e-6)
        
        # Adaptive confirmation requirement: fewer for strong trends
        required_confirmations = base_confirmations
        if trend_strength > 3.5:
            required_confirmations = 2  # Strong trend - faster confirmation
        elif trend_strength < 2.0:
            required_confirmations = 5  # Weak trend - more confirmation needed
        
        # Unified state handling logic (consolidated from duplicate code)
        if state == 0:
            # Neutral state - build confirmation count
            if detected_direction != 0:
                if detected_direction == pending_direction:
                    confirmation_count += 1
                else:
                    pending_direction = detected_direction
                    confirmation_count = 1
                
                if confirmation_count >= required_confirmations:
                    state = pending_direction
                    confirmation_count = 0
            else:
                confirmation_count = 0
                pending_direction = 0
            
            output[i] = y[i]
        else:
            # Either positive (1) or negative (-1) slope state
            is_reversal = (detected_direction == -state)
            
            if is_reversal:
                # Potential reversal - need confirmation
                if pending_direction == detected_direction:
                    confirmation_count += 1
                else:
                    pending_direction = detected_direction
                    confirmation_count = 1
                
                if confirmation_count >= required_confirmations:
                    state = detected_direction
                    confirmation_count = 0
            elif detected_direction == state:
                # Reinforce current direction
                confirmation_count = 0
                pending_direction = 0
            # else: neutral derivative - maintain state
            
            # Apply adaptive smoothing based on confirmation confidence
            confidence = min(1.0, confirmation_count / required_confirmations)
            if confirmation_count == 0:
                # Confident state - use predictive component to reduce lag
                output[i] = 0.7 * y[i] + 0.3 * trend_prediction
            else:
                # Uncertain state - smooth more aggressively
                weight = 0.70 - 0.20 * confidence
                output[i] = weight * y[i] + (1 - weight) * output[i-1]
    
    # Post-processing: smooth minor spurious reversals (more aggressive with 2 passes)
    output = _smooth_minor_reversals(output, noise_std, threshold_factor=2.5, passes=2)
    
    # Additional light smoothing to improve smoothness_score without excessive lag
    output = _light_secondary_smoothing(output)
    
    # Trim to match expected output length
    output_length = n - window_size + 1
    if len(output) >= output_length:
        output = output[window_size - 1:]
    
    return output


def _light_secondary_smoothing(y):
    """
    Light secondary smoothing pass to improve smoothness_score.
    Uses a simple 3-point weighted average to reduce high-frequency noise
    without adding significant lag.
    
    Args:
        y: Input signal
    
    Returns:
        Smoothed signal
    """
    if len(y) < 3:
        return y.copy()
    
    n = len(y)
    result = y.copy()
    
    # Light 3-point smoothing with minimal lag
    for i in range(1, n - 1):
        result[i] = 0.4 * y[i] + 0.3 * y[i-1] + 0.3 * y[i+1]
    
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
        filtered_signal = process_signal(noisy_signal, window_size, "kalman_hysteresis")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "kalman_hysteresis")

    # Calculate basic metrics (only if we have clean_signal from generation)
    if len(filtered_signal) > 0 and clean_signal is not None:
        # Align signals for comparison (zero-phase filter has minimal delay)
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

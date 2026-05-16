# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a Kalman filter combined with morphological post-processing
to filter volatile, non-stationary time series data. The morphological operations
detect and remove isolated noise-induced slope reversals while preserving genuine
trend changes, directly targeting slope_changes and false_reversals metrics.
"""
import numpy as np
from scipy import linalg
from scipy.ndimage import binary_erosion, binary_dilation


def adaptive_filter(x, window_size=20):
    """
    Kalman filter with state-space model for adaptive signal processing.
    
    State vector: [signal_value, slope]
    - Predicts next state based on current estimate and slope
    - Updates using noisy observations with adaptive Kalman gain
    - Distinguishes between genuine trend changes (updates slope state) 
      and noise (filtered out via measurement covariance)
    - Uses scipy.linalg.inv for matrix inversions in Kalman gain calculation
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples) - used for parameter tuning

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    output_length = n - window_size + 1
    
    # State transition matrix F: predicts next state from current state
    # [value_next, slope_next] = F @ [value_current, slope_current]
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]], dtype=np.float64)
    
    # Observation matrix H: maps state to observation (we observe value only)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    
    # Process noise covariance Q - controls how much state can change
    # Higher Q = more responsive to changes, lower = smoother tracking
    # Tuned based on window size: larger window = more smoothing
    process_noise_scale = 0.05 * (window_size / 20.0)
    Q = np.array([[process_noise_scale, 0.0],
                  [0.0, process_noise_scale * 0.1]], dtype=np.float64)
    
    # Measurement noise covariance R - controls trust in measurements
    # Higher R = more smoothing (less trust in measurements)
    # Tuned based on window size: larger window = more smoothing
    R = np.array([[0.3 * (window_size / 20.0)]], dtype=np.float64)
    
    # Initialize state with first window statistics
    initial_window = x[:window_size]
    initial_value = np.mean(initial_window)
    initial_slope = (initial_window[-1] - initial_window[0]) / (window_size - 1) if window_size > 1 else 0.0
    state = np.array([initial_value, initial_slope], dtype=np.float64)
    
    # Initial error covariance
    P = np.eye(2, dtype=np.float64) * 1.0
    
    y = np.zeros(output_length, dtype=np.float64)
    
    # Kalman filter iterations
    for i in range(output_length):
        # Observation at current position
        y_obs = x[i + window_size - 1]
        
        # Predict step: predict next state and covariance
        state_pred = F @ state
        P_pred = F @ P @ F.T + Q
        
        # Update step: update state using observation
        y_innovation = y_obs - H @ state_pred  # Innovation (measurement residual)
        
        # Kalman gain calculation using scipy.linalg.inv
        S = H @ P_pred @ H.T + R  # Innovation covariance
        try:
            S_inv = linalg.inv(S)
            K = P_pred @ H.T @ S_inv
        except linalg.LinAlgError:
            # Fallback if matrix is singular
            K = np.array([[0.5, 0.0]], dtype=np.float64).T
        
        # Update state and covariance
        state = state_pred + K @ y_innovation
        P = (np.eye(2, dtype=np.float64) - K @ H) @ P_pred
        
        # Output is the estimated signal value (first state component)
        y[i] = state[0]
    
    return y


def _remove_isolated_reversals(y):
    """
    Remove isolated noise-induced slope reversals using morphological operations.
    
    Algorithm:
    1. Compute direction array (sign of first difference)
    2. Use morphological erosion/dilation to detect isolated direction changes
    3. Single-sample reversals surrounded by consistent trends are removed
    4. Interpolate through removed reversals to maintain signal continuity
    
    This preserves genuine trend changes (multiple consecutive slope changes)
    while removing spurious single-sample reversals caused by noise.
    
    Args:
        y: Filtered signal array
    
    Returns:
        y_clean: Signal with isolated reversals removed
    """
    if len(y) < 3:
        return y.copy()
    
    y_clean = y.copy()
    
    # Compute direction of change at each point
    diffs = np.diff(y)
    directions = np.sign(diffs)
    
    # Pad to handle edges properly
    directions_padded = np.pad(directions, (1, 1), mode='edge')
    
    # Detect isolated direction changes using morphological operations
    # A reversal is isolated if it's surrounded by the same direction
    # Use structuring element of size 3 to detect single-sample anomalies
    struct_elem = np.array([1, 1, 1])
    
    # Find where direction changes occur
    direction_changes = np.abs(np.diff(directions_padded[1:-1], n=1)) > 0
    
    # Identify isolated reversals: direction change surrounded by non-changes
    # Use morphological opening to find isolated points
    isolated_changes = np.zeros_like(direction_changes, dtype=bool)
    
    for i in range(1, len(direction_changes) - 1):
        # Check if this is an isolated reversal (surrounded by same direction)
        if direction_changes[i] and not direction_changes[i-1] and not direction_changes[i+1]:
            # Additional check: verify the surrounding directions are consistent
            if directions_padded[i] == directions_padded[i+2]:
                isolated_changes[i] = True
    
    # Also detect single-sample direction reversals (opposite direction surrounded by same)
    for i in range(1, len(directions_padded) - 2):
        # Pattern: same direction, opposite, same direction (isolated reversal)
        if directions_padded[i] != 0 and directions_padded[i+1] != 0:
            if directions_padded[i] == -directions_padded[i+1] and directions_padded[i+2] != 0:
                if directions_padded[i] == directions_padded[i+2]:
                    # This is an isolated reversal at position i+1
                    if i + 1 < len(direction_changes):
                        isolated_changes[i] = True
    
    # Apply morphological operations to clean up the isolated reversal detection
    if np.any(isolated_changes):
        # Dilate slightly to catch edge cases
        dilated_changes = binary_dilation(isolated_changes.astype(int), structure=struct_elem).astype(bool)
        # Erode to get back to isolated points
        cleaned_changes = binary_erosion(dilated_changes.astype(int), structure=struct_elem).astype(bool)
        
        # Remove isolated reversals by interpolation
        for i in range(len(cleaned_changes)):
            if cleaned_changes[i] and i < len(y) - 1:
                # Find surrounding non-removed points for interpolation
                left_idx = i - 1
                right_idx = i + 1
                
                # Find valid left point
                while left_idx >= 0 and cleaned_changes[left_idx]:
                    left_idx -= 1
                
                # Find valid right point
                while right_idx < len(y) and cleaned_changes[right_idx]:
                    right_idx += 1
                
                # Interpolate if we have valid neighbors
                if left_idx >= 0 and right_idx < len(y):
                    # Linear interpolation
                    t = (i - left_idx) / (right_idx - left_idx)
                    y_clean[i] = y_clean[left_idx] + t * (y_clean[right_idx] - y_clean[left_idx])
    
    return y_clean


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced Kalman filter with morphological post-processing for trend preservation.
    
    Two-stage approach:
    1. Kalman filter with adaptive noise tuning for noise reduction and tracking
    2. Morphological operations to detect and remove isolated slope reversals
    
    The morphological step specifically targets false_reversals and slope_changes
    by identifying single-sample directional reversals surrounded by consistent trends,
    then interpolating through them. Genuine trend changes (multiple consecutive
    direction changes) are preserved.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window
    
    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    output_length = n - window_size + 1
    
    # State transition matrix F
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]], dtype=np.float64)
    
    # Observation matrix H
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    
    # Adaptive noise parameters based on window size
    # Larger windows = more smoothing
    base_process_noise = 0.03 * (window_size / 20.0)
    base_measurement_noise = 0.25 * (window_size / 20.0)
    
    # Initialize state
    initial_window = x[:window_size]
    initial_value = np.mean(initial_window)
    initial_slope = (initial_window[-1] - initial_window[0]) / max(window_size - 1, 1)
    state = np.array([initial_value, initial_slope], dtype=np.float64)
    
    # Initial covariance
    P = np.eye(2, dtype=np.float64) * 1.0
    
    y = np.zeros(output_length, dtype=np.float64)
    
    # Kalman filter with adaptive tuning
    for i in range(output_length):
        y_obs = x[i + window_size - 1]
        
        # Predict step
        state_pred = F @ state
        P_pred = F @ P @ F.T + np.array([[base_process_noise, 0.0],
                                         [0.0, base_process_noise * 0.1]], dtype=np.float64)
        
        # Update step
        y_innovation = y_obs - H @ state_pred
        
        # Adaptive measurement noise based on recent variance
        if i >= window_size:
            recent_window = x[i:i+window_size]
            local_var = np.var(recent_window)
            R = np.array([[base_measurement_noise + local_var * 0.1]], dtype=np.float64)
        else:
            R = np.array([[base_measurement_noise]], dtype=np.float64)
        
        # Kalman gain using scipy.linalg.inv
        S = H @ P_pred @ H.T + R
        try:
            S_inv = linalg.inv(S)
            K = P_pred @ H.T @ S_inv
        except linalg.LinAlgError:
            K = np.array([[0.5, 0.0]], dtype=np.float64).T
        
        # Update state
        state = state_pred + K @ y_innovation
        P = (np.eye(2, dtype=np.float64) - K @ H) @ P_pred
        
        y[i] = state[0]
    
    # Apply morphological post-processing to remove isolated reversals
    y = _remove_isolated_reversals(y)
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

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
        y = adaptive_filter(input_signal, window_size)
        # Apply morphological post-processing to basic filter as well
        y = _remove_isolated_reversals(y)
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

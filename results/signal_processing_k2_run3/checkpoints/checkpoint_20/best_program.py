# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing with Savitzky-Golay + Kalman Filter

Uses polynomial fitting for initial noise reduction and Kalman filter for optimal
state-space estimation with position and velocity states. Minimizes lag through
velocity prediction and reduces false reversals via adaptive noise estimation.
"""
import numpy as np


def sg_filter(x, window_size=14):
    """
    Savitzky-Golay filter with forward prediction at window end for minimal lag.
    Reduced window_size (14) balances lag error with noise reduction.
    Fits quadratic polynomial in sliding window, predicts at end point.
    """
    if len(x) < window_size:
        raise ValueError(f"Input length ({len(x)}) < window_size ({window_size})")
    
    n = len(x) - window_size + 1
    y = np.zeros(n)
    t = np.arange(window_size)
    
    for i in range(n):
        coeffs = np.polyfit(t, x[i:i+window_size], 2)
        y[i] = np.polyval(coeffs, window_size - 1)
    
    return y


def kalman_filter(x, process_noise=0.001, measure_noise_scale=0.5):
    """
    Adaptive Kalman filter with position-velocity state-space model.
    
    Implements optimal state estimation using numpy matrix operations. Models signal
    as state-space system where position and velocity evolve according to constant
    velocity model. Measurement noise is estimated from signal differences for
    adaptive filtering. Velocity prediction minimizes lag while state estimation
    reduces false reversals compared to hysteresis approaches.
    
    Args:
        x: Input signal
        process_noise: Process noise variance for state evolution
        measure_noise_scale: Scale factor for measurement noise estimation
    
    Returns:
        Filtered signal with reduced lag and controlled false reversals
    """
    if len(x) < 2:
        return x
    
    n = len(x)
    y = np.zeros(n)
    
    # Estimate measurement noise from signal differences
    diffs = np.diff(x)
    noise_var = np.var(diffs) * measure_noise_scale
    noise_var = max(noise_var, 0.01)  # Minimum noise floor
    
    # Process noise for smoothness vs responsiveness balance
    process_var = process_noise
    
    # State transition matrix (constant velocity model)
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    
    # Process noise covariance
    Q = np.array([[process_var, 0.0],
                  [0.0, process_var * 2.0]])
    
    # Measurement matrix (observe position only)
    H = np.array([[1.0, 0.0]])
    
    # Measurement noise covariance
    R = np.array([[noise_var]])
    
    # Initial state covariance
    P = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    
    # Initial state estimate [position, velocity]
    x_state = np.array([x[0], 0.0])
    
    y[0] = x_state[0]
    
    for i in range(1, n):
        # Predict step
        x_pred = F @ x_state
        P_pred = F @ P @ F.T + Q
        
        # Measurement
        z = x[i]
        
        # Update step
        y_innov = z - H @ x_pred  # Innovation
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T / S  # Kalman gain (scalar division for efficiency)
        
        x_state = x_pred + K @ y_innov
        P = (np.eye(2) - K @ H) @ P_pred
        
        # Store position estimate
        y[i] = x_state[0]
    
    return y


def light_smooth(x, factor=0.55):
    """
    Light exponential smoothing to improve smoothness without adding significant lag.
    
    Args:
        x: Input signal
        factor: Smoothing factor (0-1), higher = more responsive
    
    Returns:
        Lightly smoothed signal with improved smoothness score
    """
    if len(x) < 2:
        return x
    
    y = np.zeros(len(x))
    y[0] = x[0]
    
    for i in range(1, len(x)):
        y[i] = factor * x[i] + (1 - factor) * y[i-1]
    
    return y


def process_signal(input_signal, window_size=14, algorithm_type="enhanced"):
    """
    Adaptive SG filtering with Kalman filter state-space estimation.
    SG filter removes high-frequency noise, Kalman filter tracks underlying trend
    with velocity prediction for minimal lag. Light smoothing improves smoothness.
    
    Args:
        input_signal: Input time series data
        window_size: Window size for SG filtering (reduced for less lag)
        algorithm_type: Kept for compatibility
    
    Returns:
        Filtered signal with reduced lag, controlled false reversals, and improved smoothness
    """
    # Apply Savitzky-Golay with reduced window for minimal lag
    smoothed = sg_filter(input_signal, window_size)
    
    # Apply Kalman filter for optimal state-space estimation
    # Replaces momentum hysteresis with mathematically optimal filtering
    filtered = kalman_filter(smoothed, process_noise=0.001, measure_noise_scale=0.5)
    
    # Apply light smoothing to improve smoothness score
    final = light_smooth(filtered, factor=0.55)
    
    return final


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


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=14):
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
        # Align signals for comparison - forward prediction has minimal delay
        delay = 0
        aligned_clean = clean_signal[:len(filtered_signal)]
        aligned_noisy = noisy_signal[:len(filtered_signal)]

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

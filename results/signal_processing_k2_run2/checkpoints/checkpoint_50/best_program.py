# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements an adaptive Kalman filter with dynamic noise estimation
to filter volatile, non-stationary time series data. The filter recursively estimates
the true signal state while adapting to changing noise levels by monitoring the
innovation sequence (difference between predicted and observed values).
"""
import numpy as np


def adaptive_kalman_filter(x, initial_var=1.0, initial_process_noise=0.01, 
                           initial_measurement_noise=1.0, adapt_rate=0.1, 
                           min_process_noise=0.001, max_process_noise=1.0,
                           min_measurement_noise=0.01, max_measurement_noise=10.0):
    """
    Adaptive Kalman filter with dynamic noise estimation for real-time signal processing.
    
    Implementation approach:
    1. Maintain state estimate (signal value) and covariance (uncertainty)
    2. At each step: predict state forward, then update with measurement
    3. Track innovation sequence to estimate actual noise characteristics
    4. Adapt process noise (Q) and measurement noise (R) based on innovation variance
    5. This provides optimal linear estimation with minimal lag and no phase delay
    
    The filter naturally handles non-stationarity by adapting to changing signal
    characteristics. Each output depends only on current and past inputs, making
    it suitable for real-time processing.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        initial_var: Initial state covariance estimate
        initial_process_noise: Initial process noise variance (Q)
        initial_measurement_noise: Initial measurement noise variance (R)
        adapt_rate: Rate of adaptation for noise parameters (0 to 1)
        min_process_noise: Lower bound for process noise
        max_process_noise: Upper bound for process noise
        min_measurement_noise: Lower bound for measurement noise
        max_measurement_noise: Upper bound for measurement noise
    
    Returns:
        y: Filtered output signal with same length as input
    """
    n = len(x)
    if n == 0:
        return np.array([])
    
    if n == 1:
        return x.copy()
    
    # Initialize state: position only (simple random walk model)
    x_state = x[0]
    
    # Initialize covariance (uncertainty in state estimate)
    P = initial_var
    
    # Initialize noise parameters (will be adapted)
    Q = initial_process_noise
    R = initial_measurement_noise
    
    # Innovation tracking for adaptive noise estimation
    innovation_sum = 0.0
    innovation_sq_sum = 0.0
    innovation_count = 0
    
    # Output array
    y = np.zeros(n)
    y[0] = x_state
    
    # Kalman filter loop
    for i in range(1, n):
        # Prediction step: predict state forward
        x_pred = x_state  # Simple model: state persists (random walk)
        P_pred = P + Q    # Uncertainty increases by process noise
        
        # Measurement
        z = x[i]
        
        # Calculate Kalman gain: optimal weighting of prediction vs measurement
        K = P_pred / (P_pred + R)
        
        # Update step: correct prediction with measurement
        x_state = x_pred + K * (z - x_pred)
        P = (1.0 - K) * P_pred
        
        # Track innovation (measurement residual) for adaptive noise estimation
        innovation = z - x_pred
        innovation_sum += innovation
        innovation_sq_sum += innovation ** 2
        innovation_count += 1
        
        # Adaptive noise estimation (after warm-up period)
        if innovation_count > 10:
            # Calculate innovation variance
            mean_innovation = innovation_sum / innovation_count
            innovation_var = innovation_sq_sum / innovation_count - mean_innovation ** 2
            
            # Ensure innovation variance is positive
            innovation_var = max(innovation_var, 1e-10)
            
            # Adapt measurement noise R based on innovation variance
            # Higher innovation variance suggests higher measurement noise
            R_new = R * (1.0 - adapt_rate) + innovation_var * adapt_rate
            R = np.clip(R_new, min_measurement_noise, max_measurement_noise)
            
            # Adapt process noise Q based on signal dynamics
            # If innovation is consistently large, increase process noise
            if innovation_var > R * 1.5:
                Q_new = Q * (1.0 - adapt_rate) + innovation_var * 0.5 * adapt_rate
            else:
                Q_new = Q * (1.0 - adapt_rate) + R * 0.1 * adapt_rate
            Q = np.clip(Q_new, min_process_noise, max_process_noise)
        
        y[i] = x_state
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="kalman"):
    """
    Main signal processing function using adaptive Kalman filter.
    
    The Kalman filter provides optimal state estimation with minimal lag,
    making it ideal for real-time processing of non-stationary signals.
    
    Args:
        input_signal: Input time series data
        window_size: Window size parameter (used for compatibility)
        algorithm_type: Type of algorithm ("kalman" for adaptive Kalman filter)
    
    Returns:
        Filtered signal with same length as input
    """
    if algorithm_type == "kalman":
        return adaptive_kalman_filter(input_signal, initial_var=1.0, 
                                      initial_process_noise=0.01, 
                                      initial_measurement_noise=1.0, 
                                      adapt_rate=0.1)
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
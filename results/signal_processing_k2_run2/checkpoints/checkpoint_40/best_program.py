# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements FFT-based filtering with directional hysteresis to filter
volatile, non-stationary time series data. The directional hysteresis reduces
spurious reversals by requiring sustained movement before confirming direction changes.
"""
import numpy as np
from scipy import fft


def hybrid_fft_hysteresis_filter(x, cutoff_ratio=0.25, sigma=0.8, 
                                  hysteresis_threshold=0.02, confirm_count=3):
    """
    Hybrid FFT filtering with directional hysteresis for reduced false reversals.
    
    Combines frequency-domain filtering with directional confirmation logic:
    1. Apply Gaussian low-pass filter in frequency domain (removes high-freq noise)
    2. Track signal direction with normalized change detection
    3. Require sustained movement to confirm direction changes
    4. Constrain output to confirmed direction to suppress spurious reversals
    
    This multi-scale approach separates fast noise (filtered by FFT) from
    directional trends (confirmed by hysteresis), reducing false reversals
    while maintaining responsiveness to genuine trend changes.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        cutoff_ratio: Frequency cutoff ratio (0 to 0.5)
        sigma: Gaussian taper width (larger = smoother transition)
        hysteresis_threshold: Normalized change threshold for direction detection
        confirm_count: Consecutive samples needed to confirm direction change
    
    Returns:
        y: Filtered output signal with same length as input
    """
    n = len(x)
    if n == 0:
        return np.array([])
    
    if n == 1:
        return x.copy()
    
    # Step 1: FFT-based frequency domain filtering
    X = fft.fft(x)
    freq_indices = np.fft.fftfreq(n)
    freq_abs = np.abs(freq_indices)
    cutoff_freq = cutoff_ratio * 0.5
    
    # Gaussian filter with smooth transition
    gaussian_taper = np.exp(-(freq_abs / cutoff_freq) ** 2 / (2 * sigma ** 2))
    X_filtered = X * gaussian_taper
    y = np.real(fft.ifft(X_filtered))
    
    # Step 2: Apply directional hysteresis to reduce false reversals
    if n < confirm_count:
        return y
    
    y_hyst = np.zeros(n)
    y_hyst[0] = y[0]
    
    # Track direction state
    current_direction = 0  # 0 = unknown, 1 = up, -1 = down
    direction_confirmed = False
    consecutive_count = 0
    
    for i in range(1, n):
        diff = y[i] - y[i-1]
        abs_diff = abs(diff)
        
        # Estimate local signal variability for normalization
        window_start = max(0, i - 5)
        local_std = np.std(y[window_start:i+1])
        local_std = max(local_std, 1e-10)  # Prevent division by zero
        
        # Normalize change by local variability
        normalized_diff = abs_diff / local_std
        
        # Determine potential new direction
        if normalized_diff > hysteresis_threshold:
            new_direction = 1 if diff > 0 else -1
            
            if not direction_confirmed:
                # First direction establishment
                consecutive_count += 1
                if consecutive_count >= confirm_count:
                    current_direction = new_direction
                    direction_confirmed = True
            elif new_direction == current_direction:
                # Same direction continues
                consecutive_count += 1
            else:
                # Potential reversal - need confirmation
                consecutive_count = 1
        else:
            # Small change - maintain current direction if established
            if direction_confirmed:
                consecutive_count += 1
        
        # Apply direction constraint to output
        if direction_confirmed:
            if current_direction == 1:
                # Only allow upward movement
                y_hyst[i] = max(y[i], y_hyst[i-1])
            else:
                # Only allow downward movement
                y_hyst[i] = min(y[i], y_hyst[i-1])
        else:
            # No direction established yet
            y_hyst[i] = y[i]
    
    return y_hyst


def process_signal(input_signal, window_size=20, algorithm_type="fft"):
    """
    Main signal processing function using FFT with directional hysteresis.
    
    Args:
        input_signal: Input time series data
        window_size: Window size parameter (used for compatibility)
        algorithm_type: Type of algorithm ("fft" for hybrid filter)
    
    Returns:
        Filtered signal with same length as input
    """
    if algorithm_type == "fft":
        return hybrid_fft_hysteresis_filter(input_signal, cutoff_ratio=0.25, 
                                            sigma=0.8, hysteresis_threshold=0.02, 
                                            confirm_count=3)
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
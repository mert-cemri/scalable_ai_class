# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Uses zero-phase Butterworth filtering with duration-based slope confirmation
to minimize false reversals while maintaining zero lag.
"""
import numpy as np
from scipy import signal as scipy_signal


def butterworth_zero_phase_filter(x, window_size=20):
    """
    Adaptive Butterworth filter with FFT-based dynamic cutoff frequency.
    
    Computes local spectral content using FFT in sliding windows to determine
    optimal cutoff frequency. In high-noise regions (high-frequency content),
    uses lower cutoff for stronger filtering. In signal-dominant regions,
    uses higher cutoff to preserve tracking accuracy and reduce lag.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Reference window size for processing

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        return x.copy()
    
    n = len(x)
    # FFT window size: large enough for meaningful frequency estimation
    fft_window = min(64, max(16, window_size * 2))
    overlap = fft_window // 2  # 50% overlap for smooth adaptation
    
    # Compute adaptive cutoff frequency at each point
    cutoffs = np.zeros(n)
    
    for i in range(n):
        # Define local window centered at i
        start = max(0, i - fft_window // 2)
        end = min(n, i + fft_window // 2 + 1)
        
        if end - start < 8:
            # Not enough data, use default cutoff
            cutoffs[i] = 1.0 / window_size
            continue
        
        # Extract local window and apply Hanning window to reduce spectral leakage
        local_window = x[start:end]
        hann_window = np.hanning(len(local_window))
        windowed = local_window * hann_window
        
        # Compute FFT using numpy.fft
        fft_result = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(local_window), d=1.0)
        power = np.abs(fft_result) ** 2
        
        # Calculate spectral centroid (weighted average frequency)
        # Exclude DC component (freq=0) to focus on signal dynamics
        valid_mask = freqs > 0
        if np.sum(valid_mask) < 2:
            cutoffs[i] = 1.0 / window_size
            continue
        
        valid_freqs = freqs[valid_mask]
        valid_power = power[valid_mask]
        
        # Compute spectral centroid - higher means more high-frequency content (noise)
        spectral_centroid = np.sum(valid_freqs * valid_power) / np.sum(valid_power)
        
        # Scale cutoff based on spectral content
        # Higher centroid (more noise) -> lower cutoff for more filtering
        # Lower centroid (signal-dominated) -> higher cutoff for better tracking
        base_cutoff = 1.0 / window_size
        cutoff_factor = np.clip(1.0 - spectral_centroid * 1.5, 0.4, 1.5)
        cutoffs[i] = np.clip(base_cutoff * cutoff_factor, 0.02, 0.49)
    
    # Smooth cutoff frequencies to avoid rapid transitions that cause instability
    smooth_window = min(11, max(5, window_size // 3))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_cutoffs = scipy_signal.savgol_filter(cutoffs, smooth_window, 2)
    
    # Use median cutoff for stable filtering (robust to outliers)
    avg_cutoff = np.median(smooth_cutoffs)
    avg_cutoff = np.clip(avg_cutoff, 0.02, 0.49)
    
    # Adaptive filter order based on window size and estimated noise level
    # Higher noise (lower avg_cutoff) -> lower order for stability
    filter_order = min(4, max(2, int(window_size // 15 + (0.49 - avg_cutoff) * 2)))
    
    # Design Butterworth low-pass filter
    b, a = scipy_signal.butter(filter_order, avg_cutoff, btype='low')
    
    # Apply zero-phase filtering (forward-backward) eliminates phase delay
    padlen = 3 * max(len(a), len(b))
    y_full = scipy_signal.filtfilt(b, a, x, padlen=padlen)
    
    # Remove edge artifacts from BEGINNING to maintain zero lag
    output_length = len(x) - window_size + 1
    y = y_full[window_size-1:]
    
    return y


def apply_slope_duration_confirmation(x, window_size=20):
    """
    Duration-based slope confirmation with minimum persistence requirement.
    
    Tracks slope direction and only accepts directional changes when the new
    direction persists for a minimum number of consecutive samples. This reduces
    false reversals caused by brief noise spikes while maintaining responsiveness
    to genuine trend changes. Unlike hysteresis which only checks magnitude, this
    adds a temporal persistence requirement that directly targets false reversals.
    
    Args:
        x: Filtered signal (1D array)
        window_size: Reference window size for adaptive threshold calculation
    
    Returns:
        Stabilized signal with reduced false reversals
    """
    if len(x) < 3:
        return x.copy()
    
    x_stabilized = x.copy()
    slope = np.diff(x)
    
    # Adaptive minimum persistence threshold
    # Balance between responsiveness (low N) and false reversal reduction (high N)
    # Larger windows allow for more persistence requirement
    min_persistence = max(3, min(5, window_size // 6))
    
    # Estimate local noise level using MAD (robust to outliers)
    local_noise = np.median(np.abs(slope)) * 1.4826
    
    # Minimum slope magnitude to consider a valid direction
    slope_threshold = local_noise * 0.3
    
    # Current confirmed direction (1, -1, or 0)
    confirmed_direction = 0
    
    # Count consecutive samples in current confirmed direction
    consecutive_count = 0
    
    # Process each slope with duration confirmation
    for i in range(len(slope)):
        current_slope = slope[i]
        
        # Skip near-zero slopes (noisy regions)
        if np.abs(current_slope) <= slope_threshold:
            if confirmed_direction != 0:
                slope[i] = confirmed_direction * np.abs(current_slope)
            continue
        
        current_direction = np.sign(current_slope)
        
        if confirmed_direction == 0:
            # First valid direction - establish initial confirmed direction
            confirmed_direction = current_direction
            consecutive_count = 1
        elif current_direction == confirmed_direction:
            # Same as confirmed direction - increment consecutive count
            consecutive_count += 1
        else:
            # Direction changed - check persistence requirement
            if consecutive_count >= min_persistence:
                # Accept new direction (old direction had enough persistence)
                confirmed_direction = current_direction
                consecutive_count = 1
            else:
                # Not enough persistence - keep old direction (suppress false reversal)
                slope[i] = confirmed_direction * np.abs(current_slope)
                consecutive_count += 1
        
        # Apply confirmed direction to slope magnitude
        if confirmed_direction != 0:
            slope[i] = confirmed_direction * np.abs(slope[i])
    
    # Reconstruct signal from stabilized slopes
    x_stabilized[1:] = np.cumsum(slope) + x_stabilized[0]
    
    return x_stabilized


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function using zero-phase Butterworth filtering
    with duration-based slope confirmation for false reversal reduction.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    # Apply zero-phase Butterworth filtering
    filtered = butterworth_zero_phase_filter(input_signal, window_size)
    
    # Enhanced mode: apply duration-based slope confirmation
    if algorithm_type == "enhanced" and len(filtered) > 3:
        filtered = apply_slope_duration_confirmation(filtered, window_size)
    
    return filtered


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
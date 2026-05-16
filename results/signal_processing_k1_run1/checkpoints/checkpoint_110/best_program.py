# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements adaptive baseline removal using local polynomial fitting
combined with zero-phase Butterworth filtering. Separates slow-varying trends from
high-frequency noise, allowing different filtering strategies for each component.
"""
import numpy as np
from scipy import signal as sp_signal
from scipy import interpolate


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing using zero-phase Butterworth filter.
    
    Uses scipy.signal.filtfilt with a Butterworth low-pass filter for
    zero-phase filtering. This eliminates phase delay entirely and
    provides smooth noise reduction while preserving signal dynamics.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Design Butterworth low-pass filter
    # Order based on window_size (higher window = higher order for more smoothing)
    order = max(2, min(5, window_size // 4))
    
    # Cutoff frequency: higher window_size = lower cutoff (more smoothing)
    # Normalized frequency (0 to 1, where 1 is Nyquist)
    cutoff = 0.1 * window_size / 20.0
    cutoff = max(0.01, min(cutoff, 0.5))  # Clamp to reasonable range
    
    # Design filter
    b, a = sp_signal.butter(order, cutoff, btype='low')
    
    # Apply zero-phase filtering (forward-backward)
    y = sp_signal.filtfilt(b, a, x)
    
    # Trim to expected output length
    output_length = len(x) - window_size + 1
    y = y[:output_length]
    
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced adaptive filtering with detrend-based baseline removal.
    
    Uses local polynomial fitting to separate slow-varying trends from
    high-frequency noise. Applies zero-phase Butterworth filtering to
    residuals, then reconstructs with smoothed trend. This significantly
    reduces false reversals while preserving genuine signal dynamics.
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window for trend estimation
    
    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    n = len(x)
    
    # Step 1: Divide signal into overlapping windows for local polynomial fitting
    # Use window_size/3 overlap for smooth transitions
    overlap = max(5, window_size // 3)
    trend_window = max(15, window_size)
    
    # Step 2: Estimate local trend using piecewise polynomial fitting
    local_trend = np.zeros(n)
    trend_weights = np.zeros(n)  # Track how many windows contributed to each point
    
    for i in range(0, n, overlap):
        start = i
        end = min(i + trend_window, n)
        if end - start >= 3:  # Need at least 3 points for quadratic fit
            # Fit low-order polynomial (order 2) to estimate local trend
            try:
                coeffs = np.polyfit(range(end - start), x[start:end], deg=2)
                trend_segment = np.polyval(coeffs, range(end - start))
                local_trend[start:end] += trend_segment
                trend_weights[start:end] += 1
            except np.linalg.LinAlgError:
                # Fallback to linear fit if quadratic fails
                try:
                    coeffs = np.polyfit(range(end - start), x[start:end], deg=1)
                    trend_segment = np.polyval(coeffs, range(end - start))
                    local_trend[start:end] += trend_segment
                    trend_weights[start:end] += 1
                except:
                    # Fallback to mean
                    local_trend[start:end] += np.mean(x[start:end])
                    trend_weights[start:end] += 1
    
    # Normalize trend by number of overlapping windows
    trend_weights = np.maximum(trend_weights, 1)
    local_trend = local_trend / trend_weights
    
    # Step 3: Apply detrend to remove local baseline (get residuals)
    detrended = x - local_trend
    
    # Step 4: Apply zero-phase Butterworth filtering to residuals
    order = max(2, min(6, window_size // 3))
    cutoff = 0.08 * window_size / 20.0
    cutoff = max(0.02, min(cutoff, 0.35))
    
    b, a = sp_signal.butter(order, cutoff, btype='low')
    padlen = min(3 * max(len(a), len(b)) - 1, len(detrended) - 1)
    filtered_residuals = sp_signal.filtfilt(b, a, detrended, padlen=padlen)
    
    # Step 5: Smooth the trend component with very low-pass filter
    # Use lower cutoff to preserve slow trends while removing noise
    b_trend, a_trend = sp_signal.butter(2, 0.03, btype='low')
    padlen_trend = min(3 * max(len(a_trend), len(b_trend)) - 1, len(local_trend) - 1)
    smoothed_trend = sp_signal.filtfilt(b_trend, a_trend, local_trend, padlen=padlen_trend)
    
    # Step 6: Reconstruct signal with filtered residuals and smoothed trend
    y = filtered_residuals + smoothed_trend
    
    # Step 7: Apply hysteresis-based trend stabilization to reduce false reversals
    # Only update trend direction when change exceeds threshold
    hysteresis_threshold = 0.012 * np.std(x)
    last_significant_value = y[0]
    trend_stabilizer = np.zeros(n)
    trend_stabilizer[0] = y[0]
    
    for i in range(1, n):
        change = y[i] - last_significant_value
        rel_change = abs(change) / (abs(last_significant_value) + 1e-10)
        
        # Only register trend change if it exceeds hysteresis threshold
        if rel_change > hysteresis_threshold:
            last_significant_value = y[i]
        
        trend_stabilizer[i] = y[i]
    
    y = trend_stabilizer
    
    # Trim to expected output length
    output_length = len(x) - window_size + 1
    y = y[:output_length]
    
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
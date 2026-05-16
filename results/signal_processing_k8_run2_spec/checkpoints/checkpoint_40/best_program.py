# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach using scipy.signal.firwin2
for custom multi-band FIR filter design with arbitrary frequency response.
The custom frequency response preserves low-frequency trends while aggressively
attenuating mid-frequency noise but allowing some high-frequency content for
responsiveness, directly addressing the multi-objective tradeoff between
smoothness (slope changes) and responsiveness (lag error).

Key improvements over previous attempts:
1. Multi-band frequency response - granular control over frequency preservation
2. Adaptive Kaiser window beta based on SNR - dynamic noise rejection
3. Low-pass trend preservation (0-0.05: amplitude 1.0)
4. Mid-frequency noise attenuation (0.05-0.2: amplitude 0.1-0.3)
5. High-frequency responsiveness (0.2+: amplitude 0.2-0.4)
"""
import numpy as np
from scipy import signal


def fir_filter_with_firwin2(x, window_size=20):
    """
    Multi-band FIR filter using scipy.signal.firwin2 with custom frequency response.
    
    Approach:
    1. Design custom frequency response with multiple bands
    2. Preserve low-frequency trends (0-0.05: amplitude 1.0)
    3. Attenuate mid-frequency noise (0.05-0.2: amplitude 0.1-0.3)
    4. Allow some high-frequency for responsiveness (0.2+: amplitude 0.2-0.4)
    5. Use Kaiser window with adaptive beta based on SNR
    
    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Base parameter for adaptive filter length
    
    Returns:
        y: Filtered output signal with length = len(x) - numtaps + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    
    # Adaptive filter length - balance between lag and selectivity
    # Shorter filters = less lag, but need enough taps for frequency resolution
    numtaps = min(max(2 * window_size, 20), min(40, len(x) // 3))
    
    # Ensure even number of taps for linear phase with firwin2
    if numtaps % 2 != 0:
        numtaps += 1
    
    # Analyze signal characteristics for adaptive filtering
    signal_std = np.std(x)
    local_diffs = np.abs(np.diff(x))
    noise_estimate = np.percentile(local_diffs, 75) if len(x) > 10 else np.std(local_diffs)
    snr = max(signal_std, 1e-10) / max(noise_estimate, 1e-10)
    
    # Adaptive beta for Kaiser window (higher = better sidelobe suppression)
    # Lower beta = more responsiveness, higher beta = better noise rejection
    beta = min(10.0, max(3.0, 5.0 + (snr - 1) * 2))
    
    # Multi-band frequency response design
    # Preserve low frequencies (trend), attenuate mid (noise), partial high (responsiveness)
    freq_points = np.array([0.0, 0.03, 0.05, 0.15, 0.25, 0.5, 1.0])
    
    # Adaptive amplitude based on SNR
    # Higher SNR = more aggressive filtering, Lower SNR = more responsive
    low_pass = 1.0
    mid_attenuate = max(0.1, min(0.4, 0.3 - (snr - 1) * 0.1))
    high_partial = max(0.1, min(0.4, 0.25 + (snr - 1) * 0.05))
    
    desired_response = np.array([low_pass, low_pass, mid_attenuate, mid_attenuate, high_partial, high_partial, high_partial])
    
    # Design FIR filter with custom frequency response using firwin2
    try:
        filter_coefficients = signal.firwin2(numtaps, freq_points, desired_response, window=('kaiser', beta))
    except Exception:
        # Fallback to simple lowpass if firwin2 fails
        cutoff = 0.1
        filter_coefficients = signal.firwin(numtaps, cutoff, window=('kaiser', 5.0))
    
    # Normalize filter coefficients to ensure unity DC gain
    filter_coefficients = filter_coefficients / np.sum(filter_coefficients)
    
    # Apply filter via convolution
    y = np.convolve(x, filter_coefficients, mode='valid')
    
    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function with adaptive parameter optimization.
    
    Uses multi-band FIR filter with custom frequency response for
    linear-phase response and predictable lag with enhanced noise rejection.
    
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
    # More conservative adjustment to maintain responsiveness
    if signal_range > 0:
        volatility = signal_std / signal_range
        adaptive_window = int(window_size * (1 + volatility * 0.3))
        adaptive_window = min(adaptive_window, len(input_signal) // 4)
        adaptive_window = max(adaptive_window, 10)
    else:
        adaptive_window = window_size
    
    return fir_filter_with_firwin2(input_signal, adaptive_window)


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
        # Filter returns len(x) - numtaps + 1
        # FIR filter has linear phase with known group delay
        output_length = len(filtered_signal)
        # Group delay for FIR filter = (numtaps - 1) / 2
        # Estimate numtaps from window_size used in process_signal
        estimated_numtaps = max(25, 2 * window_size + 1)
        if estimated_numtaps % 2 == 0:
            estimated_numtaps += 1
        lag_offset = max(0, int((estimated_numtaps - 1) / 2))
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

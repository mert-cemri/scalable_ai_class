# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing with PCHIP Monotonicity-Preserving Filter

Uses Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) which mathematically
guarantees monotonicity between data points, preventing spurious reversals by construction.
Downsamples signal with moving average, then interpolates back to original resolution.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def pchip_filter(x, downsample_factor=4):
    """
    PCHIP-based monotonicity-preserving filter.
    
    Downsamples signal using moving average, then uses PCHIP interpolation
    to reconstruct at original resolution. PCHIP guarantees monotonicity
    between data points, preventing spurious reversals by construction.
    
    Args:
        x: Input noisy signal
        downsample_factor: Factor to reduce resolution before interpolation
    
    Returns:
        Filtered signal with preserved monotonicity and reduced noise
    """
    if len(x) < 5:
        return x
    
    n = len(x)
    
    # Downsample using moving average
    step = max(1, n // downsample_factor)
    downsampled_indices = np.arange(0, n, step)
    
    # Calculate moving average at each downsampled point
    downsampled_values = []
    for idx in downsampled_indices:
        start = max(0, idx - step // 2)
        end = min(n, idx + step // 2 + 1)
        downsampled_values.append(np.mean(x[start:end]))
    
    downsampled_values = np.array(downsampled_values)
    
    # Create PCHIP interpolator (monotonicity-preserving)
    pchip = PchipInterpolator(downsampled_indices, downsampled_values)
    
    # Interpolate back to original resolution
    filtered = pchip(np.arange(n))
    
    return filtered


def sg_filter(x, window_size=10):
    """
    Savitzky-Golay filter with forward prediction at window end for minimal lag.
    Fits quadratic polynomial in sliding window, predicts at end point.
    Reduced window_size (10) for lower latency.
    """
    if len(x) < window_size:
        return x
    
    n = len(x) - window_size + 1
    y = np.zeros(n)
    t = np.arange(window_size)
    
    for i in range(n):
        coeffs = np.polyfit(t, x[i:i+window_size], 2)
        y[i] = np.polyval(coeffs, window_size - 1)
    
    return y


def light_smooth(x, alpha=0.7):
    """
    Light exponential smoothing to improve smoothness without significant lag.
    Single-stage EMA for computational efficiency.
    
    Args:
        x: Input signal
        alpha: Smoothing factor (0-1), higher = more responsive
    
    Returns:
        Lightly smoothed signal
    """
    if len(x) < 2:
        return x
    
    y = np.zeros(len(x))
    y[0] = x[0]
    
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    
    return y


def process_signal(input_signal, window_size=10, algorithm_type="enhanced"):
    """
    PCHIP monotonicity-preserving filter with SG pre-filtering and light smoothing.
    PCHIP mathematically prevents spurious reversals through monotonicity constraint.
    SG filter reduces high-frequency noise, light smoothing improves smoothness score.
    
    Args:
        input_signal: Input time series data
        window_size: Window size for SG filtering
        algorithm_type: Kept for compatibility
    
    Returns:
        Filtered signal with reduced lag, controlled false reversals, and monotonicity
    """
    # Apply SG filter for initial high-frequency noise reduction
    smoothed = sg_filter(input_signal, window_size)
    
    # Apply PCHIP filter to preserve monotonicity (prevents spurious reversals)
    filtered = pchip_filter(smoothed, downsample_factor=4)
    
    # Apply light smoothing to improve smoothness score
    final = light_smooth(filtered, alpha=0.7)
    
    return final


# EVOLVE-BLOCK-END


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)
    )

    trend = 0.1 * t * np.sin(0.2 * t)
    clean_signal += trend

    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=10):
    """
    Run the signal processing algorithm on a test signal.
    """
    if noisy_signal is not None:
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        clean_signal = None
    else:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    if len(filtered_signal) > 0 and clean_signal is not None:
        delay = 0
        aligned_clean = clean_signal[:len(filtered_signal)]
        aligned_noisy = noisy_signal[:len(filtered_signal)]

        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

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
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")
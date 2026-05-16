# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing with Total Variation Denoising

Uses Total Variation (TV) denoising to preserve edges while smoothing noise.
The ROF (Rudin-Osher-Fatemi) model minimizes ||y-x||² + λ||∇y||₁ where
the L1-norm of differences preserves discontinuities (genuine trend changes)
while reducing spurious reversals caused by noise.
"""
import numpy as np
from scipy.optimize import minimize


def tv_denoise(x, lambda_tv=0.15):
    """
    Total Variation (TV) denoising using scipy.optimize.minimize.
    
    Implements the Rudin-Osher-Fatemi (ROF) model that minimizes:
    ||y - x||² + λ||∇y||₁
    
    The L1-norm of differences preserves edges while smoothing noise,
    making it ideal for reducing spurious reversals while maintaining
    genuine trend changes.
    
    Args:
        x: Input noisy signal
        lambda_tv: Regularization parameter (0.1-0.3 recommended)
    
    Returns:
        Denoised signal with preserved edges
    """
    if len(x) < 3:
        return x
    
    n = len(x)
    
    def objective(y):
        # Data fidelity term: ||y - x||²
        data_term = np.sum((y - x) ** 2)
        # Total variation term: ||∇y||₁ (L1 norm of differences)
        tv_term = lambda_tv * np.sum(np.abs(np.diff(y)))
        return data_term + tv_term
    
    def gradient(y):
        # Gradient of objective function for SLSQP
        grad = np.zeros(n)
        # Data fidelity gradient: 2*(y - x)
        grad += 2 * (y - x)
        # TV term gradient (subgradient of L1 norm)
        if n > 1:
            # For interior points
            grad[1:-1] += lambda_tv * (np.sign(y[1:-1] - y[:-2]) - np.sign(y[2:] - y[1:-1]))
            # Boundary conditions
            grad[0] += lambda_tv * np.sign(y[0] - y[1])
            grad[-1] -= lambda_tv * np.sign(y[-2] - y[-1])
        return grad
    
    # Initial guess: noisy signal
    y0 = x.copy()
    
    # Bounds for optimization
    bounds = [(None, None)] * n
    
    # Use SLSQP for constrained optimization with gradient
    result = minimize(objective, y0, method='SLSQP', jac=gradient, 
                     bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
    
    return result.x


def sg_filter(x, window_size=12):
    """
    Savitzky-Golay filter with forward prediction at window end for minimal lag.
    Reduced window_size (12) for lower latency.
    Fits quadratic polynomial in sliding window, predicts at end point.
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


def trend_confirm(x, hysteresis=0.02, min_momentum=0.03):
    """
    Hysteresis-based trend confirmation to reduce false reversals.
    Uses momentum accumulation to require sustained directional change.
    """
    if len(x) < 3:
        return x
    
    n = len(x)
    y = np.zeros(n)
    y[:2] = x[:2]
    
    dir_confirmed = 0
    momentum = 0.0
    
    for i in range(2, n):
        diff = x[i] - x[i-1]
        
        if dir_confirmed == 0:
            # No direction yet, accumulate momentum
            if abs(diff) > hysteresis:
                momentum += diff
                if abs(momentum) > min_momentum:
                    dir_confirmed = 1 if momentum > 0 else -1
                    momentum = 0
            else:
                momentum *= 0.5  # Decay momentum
        else:
            # Direction confirmed, check for reversal
            expected_sign = 1 if dir_confirmed == 1 else -1
            
            if diff * expected_sign > -hysteresis:
                momentum = max(momentum + diff * expected_sign, 0)
            else:
                momentum -= diff * expected_sign
                if momentum > min_momentum * 1.5:
                    dir_confirmed = -dir_confirmed
                    momentum = 0
        
        # Apply confirmed direction
        if dir_confirmed == 1:
            y[i] = max(x[i], y[i-1])
        elif dir_confirmed == -1:
            y[i] = min(x[i], y[i-1])
        else:
            y[i] = x[i]
    
    return y


def process_signal(input_signal, window_size=12, algorithm_type="enhanced"):
    """
    Adaptive TV denoising with SG pre-filtering and trend confirmation.
    TV denoising preserves edges while smoothing noise, reducing false
    reversals. SG filter reduces high-frequency noise, trend confirmation
    prevents spurious direction changes.
    
    Args:
        input_signal: Input time series data
        window_size: Window size for SG filtering
        algorithm_type: Kept for compatibility
    
    Returns:
        Filtered signal with reduced lag, controlled false reversals, and edge preservation
    """
    # Apply SG filter for initial high-frequency noise reduction
    smoothed = sg_filter(input_signal, window_size)
    
    # Apply TV denoising to preserve edges while smoothing (key innovation)
    denoised = tv_denoise(smoothed, lambda_tv=0.15)
    
    # Apply trend confirmation to reduce false reversals
    filtered = trend_confirm(denoised, hysteresis=0.02, min_momentum=0.03)
    
    return filtered


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


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=12):
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
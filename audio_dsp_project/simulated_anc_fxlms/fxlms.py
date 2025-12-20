"""
Filtered-x Least Mean Squares (FxLMS) Adaptive Algorithm

This module implements the core FxLMS algorithm for active noise cancellation.
"""

import numpy as np
from scipy import signal as scipy_signal


class FxLMSFilter:
    """
    FxLMS adaptive filter for active noise cancellation.
    
    The filter adapts to minimize the error signal by generating anti-noise.
    """
    
    def __init__(self, filter_length,step_size, secondary_path_estimate):
        """
        Initialize FxLMS filter.
        
        Parameters:
        -----------
        filter_length : int
            Number of filter taps (L)
        step_size : float
            Learning rate / step size (μ)
        secondary_path_estimate : np.ndarray
            Estimate of secondary path Ŝ(z)
        """
        self.L = filter_length
        self.mu = step_size
        self.s_hat = secondary_path_estimate
        
        # Initialize filter weights to zero
        self.w = np.zeros(filter_length)
        
        # Input signal buffer (reference signal x[n])
        self.x_buffer = np.zeros(filter_length)
        
        # Filtered reference signal buffer (x'[n] = x[n] * ŝ[n])
        self.xf_buffer = np.zeros(filter_length)
        
        # For tracking
        self.weight_history = []
        self.error_history = []
        
    def update(self, x_n, e_n):
        """
        Perform single-sample FxLMS update.
        
        Parameters:
        -----------
        x_n : float
            Current reference signal sample
        e_n : float
            Current error signal sample
            
        Returns:
        --------
        float
            Anti-noise output y[n]
        """
        # Shift x buffer and insert new sample
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x_n
        
        # Generate anti-noise output: y[n] = w^T[n] · x[n]
        y_n = np.dot(self.w, self.x_buffer)
        
        # Filter reference signal through secondary path estimate
        # x'[n] = x[n] * ŝ[n]
        x_full = np.concatenate([self.x_buffer, np.zeros(len(self.s_hat) - 1)])
        xf_full = np.convolve(x_full, self.s_hat, mode='valid')
        xf_n = xf_full[0] if len(xf_full) > 0 else 0
        
        # Shift filtered reference buffer
        self.xf_buffer = np.roll(self.xf_buffer, 1)
        self.xf_buffer[0] = xf_n
        
        # FxLMS weight update: w[n+1] = w[n] + μ · e[n] · x'[n]
        self.w += self.mu * e_n * self.xf_buffer
        
        # Store for analysis
        self.error_history.append(e_n)
        
        return y_n
    
    def reset(self):
        """Reset filter to initial state."""
        self.w = np.zeros(self.L)
        self.x_buffer = np.zeros(self.L)
        self.xf_buffer = np.zeros(self.L)
        self.weight_history = []
        self.error_history = []
    
    def get_weights(self):
        """Get current filter weights."""
        return self.w.copy()
    
    def get_error_history(self):
        """Get error signal history."""
        return np.array(self.error_history)


def run_fxlms_simulation(reference_signal, primary_path, secondary_path, 
                         secondary_path_estimate, filter_length=64, step_size=0.01):
    """
    Run complete FxLMS ANC simulation.
    
    Parameters:
    -----------
    reference_signal : np.ndarray
        Noise reference signal x[n]
    primary_path : np.ndarray
        Primary acoustic path P(z)
    secondary_path : np.ndarray
        Secondary acoustic path S(z)
    secondary_path_estimate : np.ndarray
        Estimated secondary path Ŝ(z)
    filter_length : int
        Adaptive filter length
    step_size : float
        Learning rate μ
        
    Returns:
    --------
    dict
        Simulation results including:
        - 'error': error signal e[n]
        - 'output': anti-noise signal y[n]
        - 'primary_noise': primary noise d[n]
        - 'weights': final filter weights
        - 'convergence_point': sample where converged
    """
    n_samples = len(reference_signal)
    
    # Apply primary path to get primary noise: d[n] = x[n] * p[n]
    primary_noise = np.convolve(reference_signal, primary_path, mode='same')
    
    # Initialize FxLMS filter
    fxlms = FxLMSFilter(filter_length, step_size, secondary_path_estimate)
    
    # Storage for signals
    anti_noise = np.zeros(n_samples)
    error = np.zeros(n_samples)
    
    # Run adaptation
    for n in range(n_samples):
        x_n = reference_signal[n]
        
        # Generate anti-noise
        y_n = fxlms.update(x_n, 0)  # Update first with e=0 to get y[n]
        anti_noise[n] = y_n
    
    # Apply secondary path to anti-noise: y'[n] = y[n] * s[n]
    anti_noise_at_mic = np.convolve(anti_noise, secondary_path, mode='same')
    
    # Compute error: e[n] = d[n] + y'[n]
    error = primary_noise + anti_noise_at_mic
    
    # Now run again with correct error feedback
    fxlms.reset()
    anti_noise = np.zeros(n_samples)
    
    for n in range(n_samples):
        x_n = reference_signal[n]
        e_n = error[n] if n > 0 else primary_noise[n]
        
        # Update filter
        y_n = fxlms.update(x_n, e_n)
        anti_noise[n] = y_n
        
        # Apply secondary path
        if n >= len(secondary_path):
            y_filtered = np.dot(anti_noise[n-len(secondary_path)+1:n+1][::-1], 
                               secondary_path)
        else:
            y_filtered = np.dot(np.concatenate([np.zeros(len(secondary_path)-n-1), 
                                               anti_noise[:n+1]])[::-1], 
                               secondary_path)
        
        # Update error
        error[n] = primary_noise[n] + y_filtered
    
    # Find convergence point (where error is reduced significantly)
    error_power = error ** 2
    window = min(1000, n_samples // 10)
    smoothed_error = np.convolve(error_power, np.ones(window)/window, mode='same')
    
    initial_error_power = np.mean(smoothed_error[:window])
    convergence_threshold = initial_error_power * 0.1  # 90% reduction
    
    converged_indices = np.where(smoothed_error < convergence_threshold)[0]
    convergence_point = converged_indices[0] if len(converged_indices) > 0 else n_samples
    
    return {
        'error': error,
        'output': anti_noise,
        'primary_noise': primary_noise,
        'weights': fxlms.get_weights(),
        'convergence_point': convergence_point,
        'error_history': fxlms.get_error_history()
    }


def compute_cancellation_db(primary_noise, error_signal):
    """
    Compute noise cancellation in dB.
    
    Parameters:
    -----------
    primary_noise : np.ndarray
        Original noise d[n]
    error_signal : np.ndarray
        Residual error e[n]
        
    Returns:
    --------
    float
        Cancellation in dB (positive = good cancellation)
    """
    # Compute power
    primary_power = np.mean(primary_noise ** 2)
    error_power = np.mean(error_signal ** 2)
    
    # Avoid log(0)
    if error_power < 1e-10:
        return 100.0  # Perfect cancellation
    
    cancellation_db = 10 * np.log10(primary_power / error_power)
    return cancellation_db


def estimate_optimal_step_size(reference_signal, secondary_path_estimate, filter_length):
    """
    Estimate a safe step size for stability.
    
    The stability condition is: 0 < μ < 2 / (L · σ_x'^2)
    
    Parameters:
    -----------
    reference_signal : np.ndarray
        Reference signal x[n]
    secondary_path_estimate : np.ndarray
        Secondary path Ŝ(z)
    filter_length : int
        Filter length L
        
    Returns:
    --------
    float
        Suggested maximum step size
    """
    # Filter reference signal
    xf = np.convolve(reference_signal, secondary_path_estimate, mode='same')
    
    # Estimate power of filtered reference
    sigma_xf_sq = np.mean(xf ** 2)
    
    # Maximum stable step size
    mu_max = 2.0 / (filter_length * sigma_xf_sq + 1e-10)
    
    # Use 50% of maximum for safety margin
    mu_safe = 0.5 * mu_max
    
    return mu_safe


if __name__ == "__main__":
    print("FxLMS Adaptive Filter Module")
    print("=" * 60)
    print()
    print("This module implements the Filtered-x LMS algorithm")
    print("for active noise cancellation.")
    print()
    print("Key equation: w[n+1] = w[n] + μ · e[n] · x'[n]")
    print("where x'[n] = x[n] * ŝ[n] (filtered reference)")
    print()
    print("Use with test_b.py to run ANC simulations")

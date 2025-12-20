"""High-level ANC simulation framework and experiments."""

import numpy as np
import matplotlib.pyplot as plt
from . import noise_gen, paths, fxlms


def run_single_simulation(noise_type='tonal', frequency=100, duration=5,
                          filter_length=64, step_size=0.01, 
                          path_mismatch=0, sample_rate=44100):
    """
    Run a single ANC simulation with specified parameters.
    
    Parameters:
    -----------
    noise_type : str
        'tonal', 'white', or 'pink'
    frequency : float
        Frequency for tonal noise (Hz)
    duration : float
        Duration in seconds
    filter_length : int
        Adaptive filter length
    step_size : float
        Learning rate μ
    path_mismatch : float
        Secondary path mismatch percentage (0-100)
    sample_rate : int
        Sample rate
        
    Returns:
    --------
    dict
        Simulation results
    """
    # Generate noise
    if noise_type == 'tonal':
        reference = noise_gen.generate_tonal_noise(frequency, duration, sample_rate)
    elif noise_type == 'white':
        reference = noise_gen.generate_white_noise(duration, sample_rate)
    elif noise_type == 'pink':
        reference = noise_gen.generate_pink_noise(duration, sample_rate)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Generate acoustic paths
    primary_path = paths.generate_primary_path(length=128, delay=5, decay=0.95)
    secondary_path = paths.generate_secondary_path(length=128, delay=3, decay=0.9)
    
    # Create secondary path estimate (with optional mismatch)
    if path_mismatch > 0:
        secondary_path_est = paths.create_path_mismatch(secondary_path, path_mismatch)
    else:
        secondary_path_est = secondary_path.copy()
    
    # Run FxLMS simulation
    results = fxlms.run_fxlms_simulation(
        reference, primary_path, secondary_path, secondary_path_est,
        filter_length, step_size
    )
    
    # Add metadata
    results['params'] = {
        'noise_type': noise_type,
        'frequency': frequency,
        'filter_length': filter_length,
        'step_size': step_size,
        'path_mismatch': path_mismatch,
        'duration': duration,
        'sample_rate': sample_rate
    }
    
    # Compute cancellation
    results['cancellation_db'] = fxlms.compute_cancellation_db(
        results['primary_noise'], results['error']
    )
    
    return results


def run_parameter_sweep(param_name, param_values, **fixed_params):
    """
    Run multiple simulations varying one parameter.
    
    Parameters:
    -----------
    param_name : str
        Parameter to vary ('step_size', 'filter_length', etc.)
    param_values : list
        Values to test
    **fixed_params : dict
        Fixed parameters for all runs
        
    Returns:
    --------
    list of dict
        Results for each parameter value
    """
    results = []
    
    for value in param_values:
        params = fixed_params.copy()
        params[param_name] = value
        
        print(f"Running simulation with {param_name}={value}...")
        result = run_single_simulation(**params)
        results.append(result)
    
    return results


def compare_noise_types(filter_length=64, step_size=0.01, duration=5):
    """
    Compare ANC performance across different noise types.
    
    Returns:
    --------
    dict
        Results for each noise type
    """
    noise_types = ['tonal', 'white', 'pink']
    results = {}
    
    for noise_type in noise_types:
        print(f"Testing {noise_type} noise...")
        results[noise_type] = run_single_simulation(
            noise_type=noise_type,
            frequency=100,
            duration=duration,
            filter_length=filter_length,
            step_size=step_size
        )
    
    return results


def test_path_mismatch(mismatch_values=[0, 5, 10, 20, 30], 
                       filter_length=64, step_size=0.01):
    """
    Test robustness to secondary path modeling errors.
    
    Returns:
    --------
    list of dict
        Results for each mismatch level
    """
    results = run_parameter_sweep(
        'path_mismatch',
        mismatch_values,
        noise_type='tonal',
        frequency=100,
        duration=5,
        filter_length=filter_length,
        step_size=step_size
    )
    
    return results


def plot_convergence_comparison(results_list, labels, title="Convergence Comparison"):
    """
    Plot error convergence for multiple simulations.
    
    Parameters:
    -----------
    results_list : list of dict
        List of simulation results
    labels : list of str
        Labels for each simulation
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for results, label in zip(results_list, labels):
        error = results['error']
        t = np.arange(len(error)) / results['params']['sample_rate']
        
        # Plot error power in dB
        error_db = 10 * np.log10(error ** 2 + 1e-10)
        plt.plot(t, error_db, label=label, alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Error Power (dB)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_cancellation_vs_parameter(results_list, param_values, param_name, 
                                   xlabel=None, title=None):
    """
    Plot cancellation performance vs parameter value.
    
    Parameters:
    -----------
    results_list : list of dict
        Simulation results
    param_values : list
        Parameter values tested
    param_name : str
        Parameter name
    """
    cancellations = [r['cancellation_db'] for r in results_list]
    convergence_times = [r['convergence_point'] / r['params']['sample_rate'] 
                        for r in results_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cancellation vs parameter
    ax1.plot(param_values, cancellations, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel(xlabel or param_name)
    ax1.set_ylabel('Cancellation (dB)')
    ax1.set_title(f'Cancellation vs {param_name}')
    ax1.grid(True, alpha=0.3)
    
    # Convergence time vs parameter
    ax2.plot(param_values, convergence_times, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel(xlabel or param_name)
    ax2.set_ylabel('Convergence Time (s)')
    ax2.set_title(f'Convergence Speed vs {param_name}')
    ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("ANC Simulation Framework")
    print("=" * 60)
    print()
    print("This module provides high-level simulation functions:")
    print("  - run_single_simulation(): Run one ANC test")
    print("  - run_parameter_sweep(): Vary one parameter")
    print("  - compare_noise_types(): Test different noise")
    print("  - test_path_mismatch(): Robustness testing")
    print()
    print("Use test_b.py for complete experiments")

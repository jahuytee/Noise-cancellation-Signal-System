"""Test script for Phase B - Simulated Active Noise Cancellation (FxLMS)"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulated_anc_fxlms import noise_gen, paths, fxlms, simulate
from realtime_noise_reduction.utils import save_wav


def test_basic_fxlms():
    """Test 1: Basic FxLMS with 100 Hz tonal noise"""
    print("=" * 70)
    print("TEST 1: Basic FxLMS - Tonal Noise (100 Hz)")
    print("=" * 70)
    print()
    
    # Run simulation
    print("Running ANC simulation...")
    results = simulate.run_single_simulation(
        noise_type='tonal',
        frequency=100,
        duration=5,
        filter_length=64,
        step_size=0.01
    )
    
    # Save audio files
    sr = results['params']['sample_rate']
    save_wav('results/audio_samples/b_primary_noise.wav', 
             results['primary_noise'], sr)
    save_wav('results/audio_samples/b_error_after_anc.wav',
             results['error'], sr)
    
    print(f"✓ Cancellation achieved: {results['cancellation_db']:.1f} dB")
    print(f"✓ Convergence point: {results['convergence_point']/sr:.2f} seconds")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    t = np.arange(len(results['error'])) / sr
    
    # Error signal
    axes[0].plot(t, results['primary_noise'], alpha=0.7, label='Primary Noise')
    axes[0].plot(t, results['error'], alpha=0.7, label='Error (After ANC)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Noise Cancellation: Before vs After')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error power in dB
    error_db = 10 * np.log10(results['error'] ** 2 + 1e-10)
    axes[1].plot(t, error_db)
    axes[1].axvline(results['convergence_point']/sr, color='r', linestyle='--', 
                   label='Convergence')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error Power (dB)')
    axes[1].set_title('Error Convergence (Lower is Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Filter weights
    axes[2].plot(results['weights'])
    axes[2].set_xlabel('Tap Index')
    axes[2].set_ylabel('Weight Value')
    axes[2].set_title('Final Adaptive Filter Weights')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/b_basic_fxlms.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Test 1 complete!")
    print()


def test_noise_types():
    """Test 2: Compare different noise types"""
    print("=" * 70)
    print("TEST 2: ANC Performance on Different Noise Types")
    print("=" * 70)
    print()
    
    # Run comparisons
    print("Testing tonal, white, and pink noise...")
    results = simulate.compare_noise_types(filter_length=64, step_size=0.01, duration=5)
    
    # Print results
    print("\nResults:")
    print(f"{'Noise Type':<15} {'Cancellation (dB)':<20} {'Convergence Time (s)':<20}")
    print("-" * 55)
    for noise_type, result in results.items():
        cancellation = result['cancellation_db']
        conv_time = result['convergence_point'] / result['params']['sample_rate']
        print(f"{noise_type:<15} {cancellation:>15.1f}       {conv_time:>15.2f}")
    
    # Plot comparison
    fig = simulate.plot_convergence_comparison(
        list(results.values()),
        list(results.keys()),
        title="Convergence Comparison: Different Noise Types"
    )
    plt.savefig('results/plots/b_noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Test 2 complete!")
    print()


def test_step_size_sweep():
    """Test 3: Vary step size mu"""
    print("=" * 70)
    print("TEST 3: Step Size Variation")
    print("=" * 70)
    print()
    
    step_sizes = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    print(f"Testing step sizes: {step_sizes}")
    results_list = simulate.run_parameter_sweep(
        'step_size',
        step_sizes,
        noise_type='tonal',
        frequency=100,
        duration=5,
        filter_length=64
    )
    
    # Print results
    print("\nResults:")
    print(f"{'Step Size (mu)':<15} {'Cancellation (dB)':<20} {'Convergence Time (s)':<20}")
    print("-" * 55)
    for result in results_list:
        mu = result['params']['step_size']
        cancellation = result['cancellation_db']
        conv_time = result['convergence_point'] / result['params']['sample_rate']
        print(f"{mu:<15.4f} {cancellation:>17.1f}       {conv_time:>17.2f}")
    
    # Plot
    fig = simulate.plot_cancellation_vs_parameter(
        results_list, step_sizes, 'step_size',
        xlabel='Step Size (mu)',
        title='ANC Performance vs Step Size'
    )
    plt.savefig('results/plots/b_step_size_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Test 3 complete!")
    print()


def test_filter_length_sweep():
    """Test 4: Vary filter length L"""
    print("=" * 70)
    print("TEST 4: Filter Length Variation")
    print("=" * 70)
    print()
    
    filter_lengths = [16, 32, 64, 128, 256]
    
    print(f"Testing filter lengths: {filter_lengths}")
    results_list = simulate.run_parameter_sweep(
        'filter_length',
        filter_lengths,
        noise_type='tonal',
        frequency=100,
        duration=5,
        step_size=0.01
    )
    
    # Print results
    print("\nResults:")
    print(f"{'Filter Length (L)':<20} {'Cancellation (dB)':<20} {'Convergence Time (s)':<20}")
    print("-" * 60)
    for result in results_list:
        L = result['params']['filter_length']
        cancellation = result['cancellation_db']
        conv_time = result['convergence_point'] / result['params']['sample_rate']
        print(f"{L:<20} {cancellation:>17.1f}       {conv_time:>17.2f}")
    
    # Plot
    fig = simulate.plot_cancellation_vs_parameter(
        results_list, filter_lengths, 'filter_length',
        xlabel='Filter Length (L)',
        title='ANC Performance vs Filter Length'
    )
    plt.savefig('results/plots/b_filter_length_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Test 4 complete!")
    print()


def test_path_mismatch():
    """Test 5: Secondary path modeling errors"""
    print("=" * 70)
    print("TEST 5: Secondary Path Mismatch Robustness")
    print("=" * 70)
    print()
    
    mismatch_levels = [0, 5, 10, 20, 30]
    
    print(f"Testing mismatch levels: {mismatch_levels}%")
    results_list = simulate.test_path_mismatch(
        mismatch_values=mismatch_levels,
        filter_length=64,
        step_size=0.01
    )
    
    # Print results
    print("\nResults:")
    print(f"{'Mismatch (%)':<15} {'Cancellation (dB)':<20} {'Degradation (dB)':<20}")
    print("-" * 55)
    
    baseline_cancellation = results_list[0]['cancellation_db']
    for result in results_list:
        mismatch = result['params']['path_mismatch']
        cancellation = result['cancellation_db']
        degradation = baseline_cancellation - cancellation
        print(f"{mismatch:<15} {cancellation:>17.1f}       {degradation:>17.1f}")
    
    # Plot
    fig = simulate.plot_cancellation_vs_parameter(
        results_list, mismatch_levels, 'path_mismatch',
        xlabel='Path Mismatch (%)',
        title='ANC Robustness to Secondary Path Errors'
    )
    plt.savefig('results/plots/b_path_mismatch.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Test 5 complete!")
    print()


def main():
    """Run Phase B test suite"""
    print("\n")
    print("=" * 70)
    print(" PHASE B TEST: SIMULATED ACTIVE NOISE CANCELLATION (FxLMS)".center(70))
    print("=" * 70)
    print()
    print("This test suite includes:")
    print("  1. Basic FxLMS with 100 Hz tonal noise")
    print("  2. Comparison of noise types (tonal, white, pink)")
    print("  3. Step size variation (mu)")
    print("  4. Filter length variation (L)")
    print("  5. Secondary path mismatch robustness")
    print()
    
    choice = input("Run all tests? (y/n, or enter test number 1-5): ").strip().lower()
    
    if choice == 'y' or choice == '':
        test_basic_fxlms()
        test_noise_types()
        test_step_size_sweep()
        test_filter_length_sweep()
        test_path_mismatch()
    elif choice == '1':
        test_basic_fxlms()
    elif choice == '2':
        test_noise_types()
    elif choice == '3':
        test_step_size_sweep()
    elif choice == '4':
        test_filter_length_sweep()
    elif choice == '5':
        test_path_mismatch()
    else:
        print("Exiting...")
        return
    
    print("\n")
    print("=" * 70)
    print(" PHASE B TESTS COMPLETE!".center(70))
    print("=" * 70)
    print()
    print("Results saved to:")
    print("   - Audio: results/audio_samples/")
    print("   - Plots: results/plots/")
    print()
    print("Listen to the before/after audio to hear the active cancellation!")
    print()


if __name__ == "__main__":
    main()

"""Test script for Phase A2 - Advanced Noise Reduction"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_noise_reduction.stream import record_audio
from realtime_noise_reduction.spectral_subtraction import (
    spectral_subtraction,
    adaptive_spectral_subtraction,
    estimate_noise_spectrum
)
from realtime_noise_reduction.wiener import (
    wiener_filter,
    adaptive_wiener_filter,
    estimate_noise_psd
)
from realtime_noise_reduction.filters import compare_audio_filtering, compute_snr_improvement
from realtime_noise_reduction.utils import save_wav, plot_spectrum, plot_spectrogram
import matplotlib.pyplot as plt


def test_spectral_subtraction():
    """Test 1: Spectral Subtraction"""
    print("=" * 70)
    print("TEST 1: Spectral Subtraction Noise Reduction")
    print("=" * 70)
    print()
    print("Spectral subtraction estimates noise during silent periods")
    print("and subtracts it from the signal spectrum.")
    print()
    print("Recording 10 seconds - speak normally with pauses.")
    print("(Pauses help estimate noise better)")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording...")
    audio = record_audio(duration=10, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a2_original_spectral.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    # Apply spectral subtraction (alpha=2.0, beta=0.02)
    print("\nApplying spectral subtraction...")
    print("  Parameters: alpha=2.0 (over-subtraction), beta=0.02 (floor)")
    filtered = spectral_subtraction(audio, 44100, alpha=2.0, beta=0.02)
    
    # Save filtered
    filtered_file = "results/audio_samples/a2_spectral_subtraction.wav"
    save_wav(filtered_file, filtered, 44100)
    print(f"Filtered saved to: {filtered_file}")
    
    # Compute SNR improvement
    snr_data = compute_snr_improvement(audio, filtered, 44100)
    print("\n📊 SNR Analysis:")
    print(f"   Original SNR:  {snr_data['original_snr']:.2f} dB")
    print(f"   Filtered SNR:  {snr_data['filtered_snr']:.2f} dB")
    print(f"   Improvement:   {snr_data['improvement']:.2f} dB")
    
    # Create comparison plot
    print("\nGenerating comparison plots...")
    compare_audio_filtering(
        audio, filtered, 44100,
        filter_name="Spectral Subtraction",
        save_path="results/plots/a2_spectral_subtraction_comparison.png"
    )
    plt.show()
    
    print("\n✅ Spectral subtraction test complete!")
    print(f"   Listen to both files to hear the difference!")
    print()


def test_adaptive_spectral_subtraction():
    """Test 2: Adaptive Spectral Subtraction"""
    print("=" * 70)
    print("TEST 2: Adaptive Spectral Subtraction")
    print("=" * 70)
    print()
    print("Adaptive version adjusts over-subtraction based on local SNR.")
    print("More aggressive in noisy sections, gentler in clean sections.")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording...")
    audio = record_audio(duration=10, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a2_original_adaptive.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    # Apply adaptive spectral subtraction
    print("\nApplying adaptive spectral subtraction...")
    print("  Alpha range: 1.0 (high SNR) to 4.0 (low SNR)")
    filtered = adaptive_spectral_subtraction(audio, 44100, 
                                            alpha_min=1.0, alpha_max=4.0, beta=0.02)
    
    # Save filtered
    filtered_file = "results/audio_samples/a2_adaptive_spectral.wav"
    save_wav(filtered_file, filtered, 44100)
    print(f"Filtered saved to: {filtered_file}")
    
    # Compute SNR improvement
    snr_data = compute_snr_improvement(audio, filtered, 44100)
    print("\n📊 SNR Analysis:")
    print(f"   Original SNR:  {snr_data['original_snr']:.2f} dB")
    print(f"   Filtered SNR:  {snr_data['filtered_snr']:.2f} dB")
    print(f"   Improvement:   {snr_data['improvement']:.2f} dB")
    
    # Create comparison plot
    print("\nGenerating comparison plots...")
    compare_audio_filtering(
        audio, filtered, 44100,
        filter_name="Adaptive Spectral Subtraction",
        save_path="results/plots/a2_adaptive_spectral_comparison.png"
    )
    plt.show()
    
    print("\n✅ Adaptive spectral subtraction test complete!")
    print()


def test_wiener_filter():
    """Test 3: Wiener Filter"""
    print("=" * 70)
    print("TEST 3: Wiener Filter Noise Reduction")
    print("=" * 70)
    print()
    print("Wiener filter is optimal in the minimum mean-square error sense.")
    print("It estimates signal and noise PSDs to compute optimal gain.")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording...")
    audio = record_audio(duration=10, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a2_original_wiener.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    # Apply Wiener filter
    print("\nApplying Wiener filter...")
    print("  Computing optimal gain: H[k] = S_xx[k] / (S_xx[k] + S_nn[k])")
    filtered = wiener_filter(audio, 44100, smoothing_factor=0.98)
    
    # Save filtered
    filtered_file = "results/audio_samples/a2_wiener.wav"
    save_wav(filtered_file, filtered, 44100)
    print(f"Filtered saved to: {filtered_file}")
    
    # Compute SNR improvement
    snr_data = compute_snr_improvement(audio, filtered, 44100)
    print("\n📊 SNR Analysis:")
    print(f"   Original SNR:  {snr_data['original_snr']:.2f} dB")
    print(f"   Filtered SNR:  {snr_data['filtered_snr']:.2f} dB")
    print(f"   Improvement:   {snr_data['improvement']:.2f} dB")
    
    # Create comparison plot
    print("\nGenerating comparison plots...")
    compare_audio_filtering(
        audio, filtered, 44100,
        filter_name="Wiener Filter",
        save_path="results/plots/a2_wiener_comparison.png"
    )
    plt.show()
    
    print("\n✅ Wiener filter test complete!")
    print()


def test_adaptive_wiener():
    """Test 4: Adaptive Wiener Filter"""
    print("=" * 70)
    print("TEST 4: Adaptive Wiener Filter")
    print("=" * 70)
    print()
    print("Adaptive version tracks noise PSD over time.")
    print("Better for time-varying noise environments.")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording...")
    audio = record_audio(duration=10, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a2_original_adaptive_wiener.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    # Apply adaptive Wiener filter
    print("\nApplying adaptive Wiener filter...")
    print("  Adaptive noise tracking with minimum gain floor")
    filtered = adaptive_wiener_filter(audio, 44100, noise_floor_db=-60)
    
    # Save filtered
    filtered_file = "results/audio_samples/a2_adaptive_wiener.wav"
    save_wav(filtered_file, filtered, 44100)
    print(f"Filtered saved to: {filtered_file}")
    
    # Compute SNR improvement
    snr_data = compute_snr_improvement(audio, filtered, 44100)
    print("\n📊 SNR Analysis:")
    print(f"   Original SNR:  {snr_data['original_snr']:.2f} dB")
    print(f"   Filtered SNR:  {snr_data['filtered_snr']:.2f} dB")
    print(f"   Improvement:   {snr_data['improvement']:.2f} dB")
    
    # Create comparison plot
    print("\nGenerating comparison plots...")
    compare_audio_filtering(
        audio, filtered, 44100,
        filter_name="Adaptive Wiener Filter",
        save_path="results/plots/a2_adaptive_wiener_comparison.png"
    )
    plt.show()
    
    print("\n✅ Adaptive Wiener filter test complete!")
    print()


def test_comparison():
    """Test 5: Compare All Methods"""
    print("=" * 70)
    print("TEST 5: Method Comparison")
    print("=" * 70)
    print()
    print("This test applies ALL noise reduction methods to the same audio")
    print("so you can compare their effectiveness.")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording 10 seconds...")
    audio = record_audio(duration=10, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a2_comparison_original.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    print("\nApplying all methods...")
    
    # Method 1: Spectral Subtraction
    print("  1. Spectral subtraction...")
    ss_filtered = spectral_subtraction(audio, 44100, alpha=2.0, beta=0.02)
    save_wav("results/audio_samples/a2_comparison_spectral.wav", ss_filtered, 44100)
    ss_snr = compute_snr_improvement(audio, ss_filtered, 44100)
    
    # Method 2: Adaptive Spectral
    print("  2. Adaptive spectral subtraction...")
    ass_filtered = adaptive_spectral_subtraction(audio, 44100)
    save_wav("results/audio_samples/a2_comparison_adaptive_spectral.wav", ass_filtered, 44100)
    ass_snr = compute_snr_improvement(audio, ass_filtered, 44100)
    
    # Method 3: Wiener
    print("  3. Wiener filter...")
    wiener_filtered = wiener_filter(audio, 44100)
    save_wav("results/audio_samples/a2_comparison_wiener.wav", wiener_filtered, 44100)
    wiener_snr = compute_snr_improvement(audio, wiener_filtered, 44100)
    
    # Method 4: Adaptive Wiener
    print("  4. Adaptive Wiener filter...")
    awiener_filtered = adaptive_wiener_filter(audio, 44100)
    save_wav("results/audio_samples/a2_comparison_adaptive_wiener.wav", awiener_filtered, 44100)
    awiener_snr = compute_snr_improvement(audio, awiener_filtered, 44100)
    
    print("\n📊 Method Comparison:")
    print(f"{'Method':<30} {'Original SNR':<15} {'Filtered SNR':<15} {'Improvement':<12}")
    print("=" * 72)
    print(f"{'Spectral Subtraction':<30} {ss_snr['original_snr']:>10.2f} dB  {ss_snr['filtered_snr']:>10.2f} dB  {ss_snr['improvement']:>8.2f} dB")
    print(f"{'Adaptive Spectral':<30} {ass_snr['original_snr']:>10.2f} dB  {ass_snr['filtered_snr']:>10.2f} dB  {ass_snr['improvement']:>8.2f} dB")
    print(f"{'Wiener Filter':<30} {wiener_snr['original_snr']:>10.2f} dB  {wiener_snr['filtered_snr']:>10.2f} dB  {wiener_snr['improvement']:>8.2f} dB")
    print(f"{'Adaptive Wiener':<30} {awiener_snr['original_snr']:>10.2f} dB  {awiener_snr['filtered_snr']:>10.2f} dB  {awiener_snr['improvement']:>8.2f} dB")
    
    # Create 4-way comparison plot
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    
    methods = [
        ("Spectral Subtraction", ss_filtered),
        ("Adaptive Spectral", ass_filtered),
        ("Wiener Filter", wiener_filtered),
        ("Adaptive Wiener", awiener_filtered)
    ]
    
    for idx, (name, filtered) in enumerate(methods):
        # Spectrum comparison
        plot_spectrum(audio, 44100, title=f"{name} - Original", ax=axes[idx, 0])
        plot_spectrum(filtered, 44100, title=f"{name} - Filtered", ax=axes[idx, 1])
    
    plt.tight_layout()
    plt.savefig("results/plots/a2_method_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Method comparison test complete!")
    print("   Listen to all the filtered files to compare!")
    print()


def main():
    """Run Phase A2 tests"""
    print("\n")
    print("=" * 70)
    print("  PHASE A2 TEST: ADVANCED NOISE REDUCTION".center(70))
    print("=" * 70)
    print()
    print("This test suite includes:")
    print("  1. Spectral Subtraction")
    print("  2. Adaptive Spectral Subtraction")
    print("  3. Wiener Filter")
    print("  4. Adaptive Wiener Filter")
    print("  5. Method Comparison (all 4 on same audio)")
    print()
    print("Each test will record audio and apply noise reduction.")
    print()
    
    choice = input("Run all tests? (y/n, or enter test number 1-5): ").strip().lower()
    
    if choice == 'y' or choice == '':
        test_spectral_subtraction()
        test_adaptive_spectral_subtraction()
        test_wiener_filter()
        test_adaptive_wiener()
        test_comparison()
    elif choice == '1':
        test_spectral_subtraction()
    elif choice == '2':
        test_adaptive_spectral_subtraction()
    elif choice == '3':
        test_wiener_filter()
    elif choice == '4':
        test_adaptive_wiener()
    elif choice == '5':
        test_comparison()
    else:
        print("Exiting...")
        return
    
    print("\n")
    print("=" * 70)
    print("  PHASE A2 TESTS COMPLETE!".center(70))
    print("=" * 70)
    print()
    print("Results saved to:")
    print("   - Audio: results/audio_samples/")
    print("   - Plots: results/plots/")
    print()
    print("Listen to the filtered audio to hear the improvements!")
    print("   Wiener filter typically provides the best quality.")
    print()


if __name__ == "__main__":
    main()

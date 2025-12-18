"""Test script for Phase A1 - Baseline Filters"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_noise_reduction.stream import record_audio
from realtime_noise_reduction.filters import (
    apply_bandpass_filter,
    apply_notch_filter,
    apply_cascaded_notch,
    compare_audio_filtering,
    compute_snr_improvement,
    plot_filter_response
)
from realtime_noise_reduction.utils import save_wav
import matplotlib.pyplot as plt


def test_filter_responses():
    """Test 1: Visualize filter frequency responses."""
    print("=" * 70)
    print("TEST 1: Filter Frequency Responses")
    print("=" * 70)
    print()
    print("Generating filter response plots...")
    
    # Band-pass filter
    print("  - Band-pass filter (80-8000 Hz)")
    plot_filter_response('bandpass', lowcut=80, highcut=8000, order=5)
    plt.savefig('results/plots/a1_bandpass_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Notch filter
    print("  - Notch filter (60 Hz)")
    plot_filter_response('notch', notch_freq=60, quality=30)
    plt.savefig('results/plots/a1_notch_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Filter response plots saved to results/plots/")
    print()


def test_bandpass_on_recording():
    """Test 2: Apply band-pass filter to recorded audio."""
    print("=" * 70)
    print("TEST 2: Band-Pass Filter on Your Voice")
    print("=" * 70)
    print()
    print("This will record 5 seconds and apply a band-pass filter.")
    print("The filter keeps only 80-8000 Hz (speech frequencies).")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording...")
    audio = record_audio(duration=5, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a1_original.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    # Apply band-pass filter
    print("\nApplying band-pass filter (80-8000 Hz)...")
    filtered = apply_bandpass_filter(audio, 44100, lowcut=80, highcut=8000, order=5)
    
    # Save filtered
    filtered_file = "results/audio_samples/a1_bandpass.wav"
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
        filter_name="Band-Pass Filtered",
        save_path="results/plots/a1_bandpass_comparison.png"
    )
    plt.show()
    
    print("\n✅ Band-pass filter test complete!")
    print()


def test_notch_on_recording():
    """Test 3: Apply notch filter to recorded audio."""
    print("=" * 70)
    print("TEST 3: Notch Filter (60 Hz Hum Removal)")
    print("=" * 70)
    print()
    print("This will record 5 seconds and remove 60 Hz + 120 Hz.")
    print("Try recording near electronics to see the effect!")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording...")
    audio = record_audio(duration=5, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a1_original_notch.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    # Apply cascaded notch filters (60 Hz + 120 Hz)
    print("\nApplying notch filters (60 Hz and 120 Hz)...")
    filtered = apply_cascaded_notch(audio, 44100, notch_freqs=[60, 120], quality=30)
    
    # Save filtered
    filtered_file = "results/audio_samples/a1_notch.wav"
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
        filter_name="Notch Filtered (60+120 Hz)",
        save_path="results/plots/a1_notch_comparison.png"
    )
    plt.show()
    
    print("\n✅ Notch filter test complete!")
    print()


def test_combined_filtering():
    """Test 4: Apply both band-pass and notch filters."""
    print("=" * 70)
    print("TEST 4: Combined Filtering (Band-Pass + Notch)")
    print("=" * 70)
    print()
    print("This applies BOTH filters for maximum noise reduction:")
    print("  1. Band-pass: Keep only 80-8000 Hz")
    print("  2. Notch: Remove 60 Hz + 120 Hz hum")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio
    print("\nRecording...")
    audio = record_audio(duration=5, sample_rate=44100, visualize=False)
    
    # Save original
    original_file = "results/audio_samples/a1_original_combined.wav"
    save_wav(original_file, audio, 44100)
    print(f"Original saved to: {original_file}")
    
    # Apply filters in sequence
    print("\nStep 1: Applying band-pass filter...")
    filtered = apply_bandpass_filter(audio, 44100, lowcut=80, highcut=8000, order=5)
    
    print("Step 2: Applying notch filters...")
    filtered = apply_cascaded_notch(filtered, 44100, notch_freqs=[60, 120], quality=30)
    
    # Save filtered
    filtered_file = "results/audio_samples/a1_combined.wav"
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
        filter_name="Combined Filtered (BP + Notch)",
        save_path="results/plots/a1_combined_comparison.png"
    )
    plt.show()
    
    print("\n✅ Combined filter test complete!")
    print()


def main():
    """Run all Phase A1 tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  PHASE A1 TEST: BASELINE FILTERS".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    print("This test suite will:")
    print("  1. Show filter frequency responses")
    print("  2. Test band-pass filter on your voice")
    print("  3. Test notch filter for 60 Hz hum removal")
    print("  4. Test combined filtering")
    print()
    print("Each test will create audio files and comparison plots.")
    print()
    
    choice = input("Run all tests? (y/n, or enter test number 1-4): ").strip().lower()
    
    if choice == 'y' or choice == '':
        test_filter_responses()
        test_bandpass_on_recording()
        test_notch_on_recording()
        test_combined_filtering()
    elif choice == '1':
        test_filter_responses()
    elif choice == '2':
        test_bandpass_on_recording()
    elif choice == '3':
        test_notch_on_recording()
    elif choice == '4':
        test_combined_filtering()
    else:
        print("Exiting...")
        return
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  ✅ PHASE A1 TESTS COMPLETE!".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    print("📁 Results saved to:")
    print("   - Audio: results/audio_samples/")
    print("   - Plots: results/plots/")
    print()
    print("🎧 Listen to the filtered audio files to hear the difference!")
    print()


if __name__ == "__main__":
    main()

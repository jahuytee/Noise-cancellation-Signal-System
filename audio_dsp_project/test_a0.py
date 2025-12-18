"""Quick test script for Phase A0 - Minimal Pipeline"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_noise_reduction.stream import record_audio
from realtime_noise_reduction.utils import estimate_snr_from_audio, plot_spectrum, plot_spectrogram
import matplotlib.pyplot as plt

def test_a0():
    """Test Phase A0: Minimal Pipeline functionality."""
    
    print("=" * 60)
    print("PHASE A0 TEST: Minimal Audio Pipeline")
    print("=" * 60)
    print()
    print("This test will:")
    print("  1. Record 5 seconds of audio from your microphone")
    print("  2. Show real-time waveform and FFT spectrum")
    print("  3. Save the recording as a WAV file")
    print("  4. Generate spectrum and spectrogram plots")
    print("  5. Estimate the SNR")
    print()
    print("Make sure your microphone is working!")
    print("Try speaking, playing music, or making noise during recording.")
    print()
    input("Press ENTER to start recording...")
    
    # Record audio with visualization
    output_file = "results/audio_samples/test_recording_a0.wav"
    print(f"\nRecording will be saved to: {output_file}")
    
    audio = record_audio(
        duration=5,
        sample_rate=44100,
        visualize=True,
        output_file=output_file
    )
    
    # Analyze the recording
    print("\nAnalyzing recorded audio...")
    print(f"  Samples: {len(audio)}")
    print(f"  Duration: {len(audio)/44100:.2f} seconds")
    
    # Estimate SNR
    snr = estimate_snr_from_audio(audio, sample_rate=44100)
    print(f"  Estimated SNR: {snr:.2f} dB")
    
    # Create analysis plots
    print("\nGenerating analysis plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_spectrum(audio, 44100, title="Audio Spectrum", ax=ax1)
    plot_spectrogram(audio, 44100, title="Audio Spectrogram", ax=ax2)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = "results/plots/test_recording_a0_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved analysis plot to: {plot_file}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("PHASE A0 TEST COMPLETE!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  - {output_file}")
    print(f"  - {plot_file}")
    print()
    print("✅ Phase A0 verified successfully!")
    print()

if __name__ == "__main__":
    test_a0()

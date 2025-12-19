"""
Audio DSP Project - Command Line Interface

A comprehensive noise reduction tool supporting multiple methods:
- Band-pass filtering
- Notch filtering  
- Spectral subtraction
- Wiener filtering
- And adaptive variants of each

Usage examples:
    python main.py --input audio.wav --method wiener --output clean.wav
    python main.py --record 10 --method spectral --output recording.wav
    python main.py --input noisy.wav --method bandpass --lowcut 100 --highcut 7000
"""

import argparse
import sys
import os
import numpy as np

from realtime_noise_reduction.stream import record_audio
from realtime_noise_reduction.filters import (
    apply_bandpass_filter,
    apply_notch_filter,
    apply_cascaded_notch,
    compare_audio_filtering
)
from realtime_noise_reduction.spectral_subtraction import (
    spectral_subtraction,
    adaptive_spectral_subtraction
)
from realtime_noise_reduction.wiener import (
    wiener_filter,
    adaptive_wiener_filter
)
from realtime_noise_reduction.utils import save_wav, estimate_snr_from_audio
import wave
import matplotlib.pyplot as plt


def load_wav(filename):
    """Load WAV file and return audio data + sample rate."""
    try:
        with wave.open(filename, 'r') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)


def process_audio(audio_data, sample_rate, args):
    """Apply selected noise reduction method to audio."""
    
    method = args.method.lower()
    
    print(f"\nApplying {method} noise reduction...")
    
    # Band-pass filter
    if method == 'bandpass' or method == 'bpf':
        lowcut = args.lowcut if args.lowcut else 80
        highcut = args.highcut if args.highcut else 8000
        order = args.order if args.order else 5
        
        print(f"  Parameters: {lowcut}-{highcut} Hz, order={order}")
        filtered = apply_bandpass_filter(audio_data, sample_rate, lowcut, highcut, order)
    
    # Notch filter
    elif method == 'notch':
        freqs = args.notch_freqs if args.notch_freqs else [60, 120]
        quality = args.quality if args.quality else 30
        
        print(f"  Parameters: frequencies={freqs} Hz, Q={quality}")
        filtered = apply_cascaded_notch(audio_data, sample_rate, freqs, quality)
    
    # Combined band-pass + notch
    elif method == 'combined':
        print("  Applying band-pass + notch filters...")
        filtered = apply_bandpass_filter(audio_data, sample_rate, 80, 8000, 5)
        filtered = apply_cascaded_notch(filtered, sample_rate, [60, 120], 30)
    
    # Spectral subtraction
    elif method == 'spectral' or method == 'sub':
        alpha = args.alpha if args.alpha else 2.0
        beta = args.beta if args.beta else 0.02
        
        print(f"  Parameters: alpha={alpha}, beta={beta}")
        filtered = spectral_subtraction(audio_data, sample_rate, alpha, beta)
    
    # Adaptive spectral subtraction
    elif method == 'adaptive_spectral':
        alpha_min = args.alpha_min if args.alpha_min else 1.0
        alpha_max = args.alpha_max if args.alpha_max else 4.0
        
        print(f"  Parameters: alpha range={alpha_min}-{alpha_max}")
        filtered = adaptive_spectral_subtraction(audio_data, sample_rate, alpha_min, alpha_max)
    
    # Wiener filter
    elif method == 'wiener':
        smoothing = args.smoothing if args.smoothing else 0.98
        
        print(f"  Parameters: smoothing={smoothing}")
        filtered = wiener_filter(audio_data, sample_rate, smoothing_factor=smoothing)
    
    # Adaptive Wiener filter
    elif method == 'adaptive_wiener':
        noise_floor = args.noise_floor if args.noise_floor else -60
        
        print(f"  Parameters: noise_floor={noise_floor} dB")
        filtered = adaptive_wiener_filter(audio_data, sample_rate, noise_floor_db=noise_floor)
    
    else:
        print(f"Error: Unknown method '{method}'")
        print("Available methods: bandpass, notch, combined, spectral, adaptive_spectral, wiener, adaptive_wiener")
        sys.exit(1)
    
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description='Audio DSP Noise Reduction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Record and filter
  python main.py --record 10 --method wiener --output clean.wav
  
  # Process existing file
  python main.py --input noisy.wav --method spectral --output clean.wav
  
  # Band-pass filter with custom parameters
  python main.py --input audio.wav --method bandpass --lowcut 100 --highcut 7000
  
  # Compare before/after with plots
  python main.py --input noisy.wav --method wiener --output clean.wav --compare
  
Available methods:
  bandpass, bpf       - Band-pass filter (80-8000 Hz default)
  notch               - Notch filter (60, 120 Hz default)
  combined            - Band-pass + notch
  spectral, sub       - Spectral subtraction
  adaptive_spectral   - Adaptive spectral subtraction
  wiener              - Wiener filter (recommended)
  adaptive_wiener     - Adaptive Wiener filter (best for varying noise)
        '''
    )
    
    # Input/Output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='Input WAV file')
    input_group.add_argument('--record', '-r', type=int, metavar='SECONDS',
                           help='Record audio for specified seconds')
    
    parser.add_argument('--output', '-o', required=True,
                       help='Output WAV file')
    
    # Method selection
    parser.add_argument('--method', '-m', required=True,
                       choices=['bandpass', 'bpf', 'notch', 'combined',
                               'spectral', 'sub', 'adaptive_spectral',
                               'wiener', 'adaptive_wiener'],
                       help='Noise reduction method')
    
    # Common parameters
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Sample rate for recording (default: 44100)')
    
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Show before/after comparison plots')
    
    parser.add_argument('--snr', action='store_true',
                       help='Calculate and display SNR improvement')
    
    # Band-pass parameters
    parser.add_argument('--lowcut', type=float,
                       help='Band-pass low cutoff frequency (default: 80 Hz)')
    parser.add_argument('--highcut', type=float,
                       help='Band-pass high cutoff frequency (default: 8000 Hz)')
    parser.add_argument('--order', type=int,
                       help='Filter order (default: 5)')
    
    # Notch parameters
    parser.add_argument('--notch-freqs', type=float, nargs='+',
                       help='Notch filter frequencies (default: 60 120)')
    parser.add_argument('--quality', type=float,
                       help='Notch filter Q factor (default: 30)')
    
    # Spectral subtraction parameters
    parser.add_argument('--alpha', type=float,
                       help='Over-subtraction factor (default: 2.0)')
    parser.add_argument('--beta', type=float,
                       help='Spectral floor factor (default: 0.02)')
    parser.add_argument('--alpha-min', type=float,
                       help='Min alpha for adaptive (default: 1.0)')
    parser.add_argument('--alpha-max', type=float,
                       help='Max alpha for adaptive (default: 4.0)')
    
    # Wiener parameters
    parser.add_argument('--smoothing', type=float,
                       help='Wiener gain smoothing (default: 0.98)')
    parser.add_argument('--noise-floor', type=float,
                       help='Adaptive Wiener noise floor in dB (default: -60)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("  Audio DSP Noise Reduction Tool".center(70))
    print("=" * 70)
    
    # Get input audio
    if args.input:
        print(f"\nLoading: {args.input}")
        audio_data, sample_rate = load_wav(args.input)
        print(f"  Duration: {len(audio_data)/sample_rate:.2f} seconds")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Samples: {len(audio_data)}")
    else:
        print(f"\nRecording {args.record} seconds at {args.sample_rate} Hz...")
        print("Speak into your microphone!")
        audio_data = record_audio(
            duration=args.record,
            sample_rate=args.sample_rate,
            visualize=False
        )
        sample_rate = args.sample_rate
        print("Recording complete!")
    
    # Calculate original SNR if requested
    if args.snr:
        original_snr = estimate_snr_from_audio(audio_data, sample_rate)
        print(f"\nOriginal SNR: {original_snr:.2f} dB")
    
    # Process audio
    filtered_data = process_audio(audio_data, sample_rate, args)
    
    print("Processing complete!")
    
    # Calculate filtered SNR if requested
    if args.snr:
        filtered_snr = estimate_snr_from_audio(filtered_data, sample_rate)
        improvement = filtered_snr - original_snr
        print(f"\nFiltered SNR: {filtered_snr:.2f} dB")
        print(f"Improvement:  {improvement:+.2f} dB")
    
    # Save output
    print(f"\nSaving: {args.output}")
    save_wav(args.output, filtered_data, sample_rate)
    
    # Show comparison if requested
    if args.compare:
        print("\nGenerating comparison plots...")
        compare_audio_filtering(
            audio_data, filtered_data, sample_rate,
            filter_name=args.method.upper()
        )
        plt.show()
    
    print("\n" + "=" * 70)
    print("  Processing Complete!".center(70))
    print("=" * 70)
    print(f"\nOutput saved to: {args.output}")
    
    if args.compare:
        print("Close the plot window to exit.")
    
    print()


if __name__ == "__main__":
    main()

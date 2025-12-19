"""Spectral subtraction noise reduction algorithm."""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def estimate_noise_spectrum(audio_data, sample_rate, energy_threshold=0.02, frame_length_ms=25):
    """
    Estimate noise spectrum from silent/low-energy frames.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    energy_threshold : float
        Normalized energy threshold for detecting silent frames (0-1)
    frame_length_ms : float
        Frame length in milliseconds
        
    Returns:
    --------
    np.ndarray
        Estimated noise magnitude spectrum
    """
    frame_length = int(frame_length_ms * sample_rate / 1000)
    num_frames = len(audio_data) // frame_length
    
    # Compute frame energies
    energies = []
    noise_frames = []
    
    for i in range(num_frames):
        frame = audio_data[i * frame_length:(i + 1) * frame_length]
        energy = np.mean(frame ** 2)
        energies.append(energy)
        
        # Store frame for later
        noise_frames.append(frame)
    
    energies = np.array(energies)
    
    # Normalize energies
    if np.max(energies) > 0:
        max_energy = np.max(energies)
    else:
        max_energy = 1.0
    
    norm_energies = energies / max_energy
    
    # Identify silent frames (noise-only)
    silent_mask = norm_energies < energy_threshold
    
    if not np.any(silent_mask):
        # If no silent frames, use the quietest 20%
        threshold = np.percentile(norm_energies, 20)
        silent_mask = norm_energies < threshold
    
    # Average spectrum of silent frames
    noise_spectra = []
    for i, is_silent in enumerate(silent_mask):
        if is_silent:
            frame = noise_frames[i]
            # Apply window to reduce spectral leakage
            window = np.hanning(len(frame))
            windowed_frame = frame * window
            
            # Compute magnitude spectrum
            spectrum = np.abs(np.fft.rfft(windowed_frame))
            noise_spectra.append(spectrum)
    
    if len(noise_spectra) == 0:
        # Fallback: use first frame
        frame = audio_data[:frame_length]
        window = np.hanning(len(frame))
        return np.abs(np.fft.rfft(frame * window))
    
    # Average noise spectrum
    noise_spectrum = np.mean(noise_spectra, axis=0)
    
    return noise_spectrum


def spectral_subtraction(audio_data, sample_rate, 
                         alpha=2.0, beta=0.02, 
                         frame_length=2048, hop_length=None,
                         noise_spectrum=None):
    """
    Apply spectral subtraction noise reduction.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input noisy audio signal
    sample_rate : int
        Sample rate in Hz
    alpha : float
        Over-subtraction factor (default: 2.0)
        Higher values = more aggressive noise removal
    beta : float
        Spectral floor factor (default: 0.02)
        Prevents over-subtraction artifacts
    frame_length : int
        Frame size for STFT (default: 2048)
    hop_length : int, optional
        Hop size between frames (default: frame_length // 2)
    noise_spectrum : np.ndarray, optional
        Pre-computed noise spectrum. If None, will be estimated.
        
    Returns:
    --------
    np.ndarray
        Denoised audio signal
    """
    if hop_length is None:
        hop_length = frame_length // 2
    
    # Estimate noise spectrum if not provided
    if noise_spectrum is None:
        noise_spectrum = estimate_noise_spectrum(audio_data, sample_rate)
    
    # Ensure noise spectrum matches frame length
    if len(noise_spectrum) != frame_length // 2 + 1:
        # Resample noise spectrum to match frame length
        noise_spectrum_full = estimate_noise_spectrum(audio_data, sample_rate, 
                                                       frame_length_ms=frame_length * 1000 / sample_rate)
    else:
        noise_spectrum_full = noise_spectrum
    
    # Apply Hann window
    window = np.hanning(frame_length)
    
    # Compute STFT
    num_frames = 1 + (len(audio_data) - frame_length) // hop_length
    
    # Storage for processed frames
    output_frames = []
    
    for i in range(num_frames):
        # Extract frame
        start = i * hop_length
        end = start + frame_length
        
        if end > len(audio_data):
            # Pad last frame
            frame = np.zeros(frame_length)
            frame[:len(audio_data) - start] = audio_data[start:]
        else:
            frame = audio_data[start:end]
        
        # Apply window
        windowed_frame = frame * window
        
        # Compute spectrum
        spectrum = np.fft.rfft(windowed_frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Spectral subtraction
        # |X_clean[k]| = max(|Y[k]| - alpha * |N[k]|, beta * |Y[k]|)
        clean_magnitude = np.maximum(
            magnitude - alpha * noise_spectrum_full,
            beta * magnitude
        )
        
        # Reconstruct spectrum with original phase
        clean_spectrum = clean_magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        clean_frame = np.fft.irfft(clean_spectrum)
        
        # Apply window for overlap-add
        clean_frame = clean_frame * window
        
        output_frames.append(clean_frame)
    
    # Overlap-add reconstruction
    output_length = (num_frames - 1) * hop_length + frame_length
    output = np.zeros(output_length)
    
    for i, frame in enumerate(output_frames):
        start = i * hop_length
        output[start:start + frame_length] += frame
    
    # Trim to original length
    output = output[:len(audio_data)]
    
    # Normalize to prevent clipping
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * np.max(np.abs(audio_data))
    
    return output


def adaptive_spectral_subtraction(audio_data, sample_rate,
                                  alpha_min=1.0, alpha_max=4.0,
                                  beta=0.02, frame_length=2048):
    """
    Adaptive spectral subtraction with time-varying over-subtraction factor.
    
    Adjusts alpha based on local SNR estimate.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input noisy audio
    sample_rate : int
        Sample rate in Hz
    alpha_min : float
        Minimum over-subtraction factor for high SNR regions
    alpha_max : float
        Maximum over-subtraction factor for low SNR regions
    beta : float
        Spectral floor factor
    frame_length : int
        Frame size
        
    Returns:
    --------
    np.ndarray
        Denoised audio
    """
    # Estimate global noise spectrum
    noise_spectrum = estimate_noise_spectrum(audio_data, sample_rate,
                                             frame_length_ms=frame_length * 1000 / sample_rate)
    
    hop_length = frame_length // 2
    window = np.hanning(frame_length)
    num_frames = 1 + (len(audio_data) - frame_length) // hop_length
    
    output_frames = []
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        
        if end > len(audio_data):
            frame = np.zeros(frame_length)
            frame[:len(audio_data) - start] = audio_data[start:]
        else:
            frame = audio_data[start:end]
        
        windowed_frame = frame * window
        spectrum = np.fft.rfft(windowed_frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Estimate local SNR
        signal_power = np.mean(magnitude ** 2)
        noise_power = np.mean(noise_spectrum ** 2)
        
        if noise_power > 0:
            local_snr = signal_power / noise_power
        else:
            local_snr = 100  # High SNR if no noise
        
        # Adapt alpha based on SNR (lower SNR = higher alpha)
        # SNR of 1 (0 dB) → alpha_max
        # SNR of 10 (10 dB) → alpha_min
        alpha = alpha_max - (alpha_max - alpha_min) * min(local_snr / 10, 1.0)
        
        # Spectral subtraction with adaptive alpha
        clean_magnitude = np.maximum(
            magnitude - alpha * noise_spectrum,
            beta * magnitude
        )
        
        clean_spectrum = clean_magnitude * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spectrum) * window
        
        output_frames.append(clean_frame)
    
    # Overlap-add
    output_length = (num_frames - 1) * hop_length + frame_length
    output = np.zeros(output_length)
    
    for i, frame in enumerate(output_frames):
        start = i * hop_length
        output[start:start + frame_length] += frame
    
    output = output[:len(audio_data)]
    
    # Normalize
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * np.max(np.abs(audio_data))
    
    return output


if __name__ == "__main__":
    print("Spectral Subtraction Noise Reduction Module")
    print("=" * 60)
    print()
    print("This module provides spectral subtraction algorithms:")
    print("  - Basic spectral subtraction with fixed parameters")
    print("  - Adaptive spectral subtraction with SNR-based tuning")
    print()
    print("Use with test_a2.py to test on real audio.")

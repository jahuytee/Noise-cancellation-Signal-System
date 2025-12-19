"""Wiener filter for optimal noise reduction."""

import numpy as np
from scipy import signal


def estimate_psd(audio_data, sample_rate, frame_length=2048, hop_length=None):
    """
    Estimate Power Spectral Density (PSD) of audio signal.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz  
    frame_length : int
        Frame size for STFT
    hop_length : int, optional
        Hop size (default: frame_length // 2)
        
    Returns:
    --------
    np.ndarray
        Average power spectral density
    """
    if hop_length is None:
        hop_length = frame_length // 2
    
    window = np.hanning(frame_length)
    num_frames = 1 + (len(audio_data) - frame_length) // hop_length
    
    psds = []
    
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
        psd = np.abs(spectrum) ** 2
        psds.append(psd)
    
    # Average PSD across all frames
    average_psd = np.mean(psds, axis=0)
    
    return average_psd


def estimate_noise_psd(audio_data, sample_rate, energy_threshold=0.02,
                       frame_length=2048, hop_length=None):
    """
    Estimate noise PSD from low-energy (silent) frames.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    energy_threshold : float
        Normalized energy threshold for silent frames
    frame_length : int
        Frame size
    hop_length : int, optional
        Hop size
        
    Returns:
    --------
    np.ndarray
        Estimated noise PSD
    """
    if hop_length is None:
        hop_length = frame_length // 2
    
    window = np.hanning(frame_length)
    num_frames = 1 + (len(audio_data) - frame_length) // hop_length
    
    # First pass: compute energies
    energies = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        
        if end > len(audio_data):
            frame = np.zeros(frame_length)
            frame[:len(audio_data) - start] = audio_data[start:]
        else:
            frame = audio_data[start:end]
        
        energy = np.mean(frame ** 2)
        energies.append(energy)
    
    energies = np.array(energies)
    
    # Normalize energies
    if np.max(energies) > 0:
        max_energy = np.max(energies)
    else:
        max_energy = 1.0
    
    norm_energies = energies / max_energy
    
    # Identify silent frames
    silent_mask = norm_energies < energy_threshold
    
    if not np.any(silent_mask):
        # Use quietest 20% if no silent frames
        threshold = np.percentile(norm_energies, 20)
        silent_mask = norm_energies < threshold
    
    # Second pass: compute PSDs for silent frames
    noise_psds = []
    
    for i in range(num_frames):
        if not silent_mask[i]:
            continue
        
        start = i * hop_length
        end = start + frame_length
        
        if end > len(audio_data):
            frame = np.zeros(frame_length)
            frame[:len(audio_data) - start] = audio_data[start:]
        else:
            frame = audio_data[start:end]
        
        windowed_frame = frame * window
        spectrum = np.fft.rfft(windowed_frame)
        psd = np.abs(spectrum) ** 2
        noise_psds.append(psd)
    
    if len(noise_psds) == 0:
        # Fallback: use first frame
        frame = audio_data[:frame_length]
        windowed_frame = frame * window
        spectrum = np.fft.rfft(windowed_frame)
        return np.abs(spectrum) ** 2
    
    # Average noise PSD
    noise_psd = np.mean(noise_psds, axis=0)
    
    return noise_psd


def wiener_filter(audio_data, sample_rate,
                 noise_psd=None, 
                 frame_length=2048, hop_length=None,
                 smoothing_factor=0.98):
    """
    Apply Wiener filter for noise reduction.
    
    The Wiener filter is optimal in the minimum mean-square error sense:
    H[k] = S_xx[k] / (S_xx[k] + S_nn[k])
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input noisy audio
    sample_rate : int
        Sample rate in Hz
    noise_psd : np.ndarray, optional
        Pre-computed noise PSD. If None, will be estimated.
    frame_length : int
        Frame size for STFT
    hop_length : int, optional
        Hop size
    smoothing_factor : float
        Exponential smoothing factor for gain (0-1)
        Higher = smoother gain over time
        
    Returns:
    --------
    np.ndarray
        Filtered audio signal
    """
    if hop_length is None:
        hop_length = frame_length // 2
    
    # Estimate noise PSD if not provided
    if noise_psd is None:
        noise_psd = estimate_noise_psd(audio_data, sample_rate,
                                       frame_length=frame_length,
                                       hop_length=hop_length)
    
    window = np.hanning(frame_length)
    num_frames = 1 + (len(audio_data) - frame_length) // hop_length
    
    output_frames = []
    previous_gain = None
    
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
        
        # Estimate signal PSD for this frame
        # S_yy = |Y|^2 (observed noisy signal PSD)
        noisy_psd = magnitude ** 2
        
        # Estimate clean signal PSD
        # S_xx = S_yy - S_nn (with floor at 0)
        signal_psd = np.maximum(noisy_psd - noise_psd, 1e-10)
        
        # Compute Wiener gain
        # H[k] = S_xx[k] / (S_xx[k] + S_nn[k])
        wiener_gain = signal_psd / (signal_psd + noise_psd)
        
        # Smooth gain over time to reduce musical noise
        if previous_gain is not None:
            wiener_gain = smoothing_factor * previous_gain + (1 - smoothing_factor) * wiener_gain
        
        previous_gain = wiener_gain
        
        # Apply gain
        filtered_magnitude = magnitude * wiener_gain
        
        # Reconstruct spectrum with original phase
        filtered_spectrum = filtered_magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        filtered_frame = np.fft.irfft(filtered_spectrum) * window
        
        output_frames.append(filtered_frame)
    
    # Overlap-add reconstruction
    output_length = (num_frames - 1) * hop_length + frame_length
    output = np.zeros(output_length)
    
    for i, frame in enumerate(output_frames):
        start = i * hop_length
        output[start:start + frame_length] += frame
    
    # Trim to original length
    output = output[:len(audio_data)]
    
    # Normalize
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * np.max(np.abs(audio_data))
    
    return output


def adaptive_wiener_filter(audio_data, sample_rate,
                           frame_length=2048, hop_length=None,
                           noise_floor_db=-60):
    """
    Adaptive Wiener filter with improved noise PSD tracking.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input noisy audio
    sample_rate : int
        Sample rate in Hz
    frame_length : int
        Frame size
    hop_length : int, optional
        Hop size
    noise_floor_db : float
        Minimum gain in dB (prevents over-suppression)
        
    Returns:
    --------
    np.ndarray
        Filtered audio
    """
    if hop_length is None:
        hop_length = frame_length // 2
    
    window = np.hanning(frame_length)
    num_frames = 1 + (len(audio_data) - frame_length) // hop_length
    
    # Initial noise PSD estimate
    noise_psd = estimate_noise_psd(audio_data, sample_rate,
                                    frame_length=frame_length,
                                    hop_length=hop_length)
    
    output_frames = []
    noise_floor_linear = 10 ** (noise_floor_db / 20)
    
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
        
        noisy_psd = magnitude ** 2
        
        # Adaptive noise PSD update (track slowly varying noise)
        # If current PSD is lower than estimate, update noise estimate
        alpha_noise = 0.95  # Smoothing for noise tracking
        noise_psd = alpha_noise * noise_psd + (1 - alpha_noise) * np.minimum(noisy_psd, noise_psd)
        
        # Estimate signal PSD
        signal_psd = np.maximum(noisy_psd - noise_psd, 1e-10)
        
        # Wiener gain with floor
        wiener_gain = signal_psd / (signal_psd + noise_psd)
        wiener_gain = np.maximum(wiener_gain, noise_floor_linear)
        
        # Apply gain
        filtered_magnitude = magnitude * wiener_gain
        filtered_spectrum = filtered_magnitude * np.exp(1j * phase)
        filtered_frame = np.fft.irfft(filtered_spectrum) * window
        
        output_frames.append(filtered_frame)
    
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
    print("Wiener Filter Noise Reduction Module")
    print("=" * 60)
    print()
    print("This module provides Wiener filtering algorithms:")
    print("  - Classic Wiener filter with fixed noise PSD")
    print("  - Adaptive Wiener filter with noise tracking")
    print()
    print("Use with test_a2.py to test on real audio.")

"""Utility functions for audio processing and visualization."""

import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal


def compute_snr(signal_data, noise_data):
    """
    Compute Signal-to-Noise Ratio in dB.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Clean or desired signal
    noise_data : np.ndarray
        Noise signal
        
    Returns:
    --------
    float
        SNR in decibels
    """
    signal_power = np.mean(signal_data ** 2)
    noise_power = np.mean(noise_data ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


def estimate_snr_from_audio(audio_data, sample_rate, energy_threshold=0.02):
    """
    Estimate SNR by detecting signal and noise segments.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    energy_threshold : float
        Threshold for detecting silent frames (normalized energy)
        
    Returns:
    --------
    float
        Estimated SNR in dB
    """
    # Frame parameters
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    
    # Compute frame energies
    num_frames = len(audio_data) // frame_length
    energies = []
    
    for i in range(num_frames):
        frame = audio_data[i * frame_length:(i + 1) * frame_length]
        energy = np.mean(frame ** 2)
        energies.append(energy)
    
    energies = np.array(energies)
    
    # Normalize energies
    if np.max(energies) > 0:
        max_energy = np.max(energies)
    else:
        max_energy = 1.0

    norm_energies = energies / max_energy
    
    # Separate signal and noise frames
    noise_frames = norm_energies < energy_threshold
    signal_frames = ~noise_frames
    
    if not np.any(signal_frames) or not np.any(noise_frames):
        return 0.0
    
    signal_power = np.mean(energies[signal_frames])
    noise_power = np.mean(energies[noise_frames])
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db


def save_wav(filename, audio_data, sample_rate):
    """
    Save audio data to WAV file.
    
    Parameters:
    -----------
    filename : str
        Output WAV file path
    audio_data : np.ndarray
        Audio signal to save
    sample_rate : int
        Sample rate in Hz
    """
    # Normalize to 16-bit range
    audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_normalized.tobytes())
    
    print(f"Saved audio to {filename}")


def plot_spectrum(audio_data, sample_rate, title="Magnitude Spectrum", ax=None):
    """
    Plot magnitude spectrum of audio signal.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Compute FFT

    n = len(audio_data)
    fft_data = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(n, 1/sample_rate)
    magnitude = np.abs(fft_data)
    
    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    ax.plot(freqs, magnitude_db)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, sample_rate / 2)
    
    return ax


def plot_spectrogram(audio_data, sample_rate, title="Spectrogram", ax=None):
    """
    Plot spectrogram of audio signal.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=1024)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Power (dB)')
    
    return ax


def plot_waveform(audio_data, sample_rate, title="Waveform", ax=None):
    """
    Plot time-domain waveform.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    time = np.arange(len(audio_data)) / sample_rate
    
    ax.plot(time, audio_data, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax

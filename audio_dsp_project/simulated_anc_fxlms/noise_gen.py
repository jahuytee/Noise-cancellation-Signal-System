"""Noise signal generation for ANC simulation."""

import numpy as np


def generate_tonal_noise(frequency, duration, sample_rate=44100, amplitude=1.0):
    """
    Generate a pure tone (single frequency).
    
    Parameters:
    -----------
    frequency : float
        Frequency in Hz
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate in Hz
    amplitude : float
        Amplitude (0-1)
        
    Returns:
    --------
    np.ndarray
        Generated noise signal
    """
    t = np.arange(0, duration, 1/sample_rate)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal


def generate_white_noise(duration, sample_rate=44100, amplitude=1.0):
    """
    Generate white noise (flat spectrum).
    
    Parameters:
    -----------
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate in Hz
    amplitude : float
        RMS amplitude
        
    Returns:
    --------
    np.ndarray
        Generated white noise
    """
    n_samples = int(duration * sample_rate)
    noise = np.random.randn(n_samples)
    
    # Normalize to desired RMS amplitude
    rms = np.sqrt(np.mean(noise ** 2))
    noise = noise * (amplitude / rms)
    
    return noise


def generate_pink_noise(duration, sample_rate=44100, amplitude=1.0):
    """
    Generate pink noise (1/f spectrum).
    
    Uses the Voss-McCartney algorithm.
    
    Parameters:
    -----------
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate in Hz
    amplitude : float
        RMS amplitude
        
    Returns:
    --------
    np.ndarray
        Generated pink noise
    """
    n_samples = int(duration * sample_rate)
    
    # Number of random sources
    n_rows = 16
    
    # Generate random values
    array = np.random.randn(n_rows, n_samples)
    
    # Determine when to update each row
    update_pattern = np.zeros((n_rows, n_samples), dtype=bool)
    for i in range(n_rows):
        update_pattern[i, ::2**(i+1)] = True
    
    # Build pink noise
    pink = np.zeros(n_samples)
    values = np.zeros(n_rows)
    
    for i in range(n_samples):
        # Update values that need updating
        for row in range(n_rows):
            if update_pattern[row, i]:
                values[row] = array[row, i]
        
        # Sum all values
        pink[i] = np.sum(values)
    
    # Normalize to desired RMS amplitude
    rms = np.sqrt(np.mean(pink ** 2))
    pink = pink * (amplitude / rms)
    
    return pink


def generate_swept_sine(f_start, f_end, duration, sample_rate=44100, amplitude=1.0):
    """
    Generate a frequency-swept sine wave (chirp).
    
    Parameters:
    -----------
    f_start : float
        Starting frequency in Hz
    f_end : float
        Ending frequency in Hz
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate in Hz
    amplitude : float
        Amplitude (0-1)
        
    Returns:
    --------
    np.ndarray
        Generated chirp signal
    """
    t = np.arange(0, duration, 1/sample_rate)
    
    # Linear frequency sweep
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    
    signal = amplitude * np.sin(phase)
    return signal


def generate_multi_tonal(frequencies, duration, sample_rate=44100, amplitude=1.0):
    """
    Generate multiple tones combined.
    
    Parameters:
    -----------
    frequencies : list of float
        List of frequencies in Hz
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate in Hz
    amplitude : float
        Total amplitude (distributed equally across tones)
        
    Returns:
    --------
    np.ndarray
        Combined multi-tonal signal
    """
    t = np.arange(0, duration, 1/sample_rate)
    signal = np.zeros(len(t))
    
    # Add each frequency component
    amp_per_tone = amplitude / len(frequencies)
    for freq in frequencies:
        signal += amp_per_tone * np.sin(2 * np.pi * freq * t)
    
    return signal


def add_noise_to_signal(signal, noise_type='white', snr_db=10):
    """
    Add noise to a clean signal at specified SNR.
    
    Parameters:
    -----------
    signal : np.ndarray
        Clean signal
    noise_type : str
        Type of noise ('white', 'pink')
    snr_db : float
        Signal-to-noise ratio in dB
        
    Returns:
    --------
    np.ndarray
        Noisy signal
    """
    # Calculate signal power
    signal_power = np.mean(signal ** 2)
    
    # Calculate desired noise power
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    if noise_type == 'white':
        noise = generate_white_noise(
            len(signal) / 44100,  # Assuming 44.1kHz
            amplitude=np.sqrt(noise_power)
        )
    elif noise_type == 'pink':
        noise = generate_pink_noise(
            len(signal) / 44100,
            amplitude=np.sqrt(noise_power)
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Ensure same length
    noise = noise[:len(signal)]
    
    return signal + noise


if __name__ == "__main__":
    print("Noise Generation Module for ANC Simulation")
    print("=" * 60)
    print()
    print("Available noise types:")
    print("  - Tonal (pure sine wave)")
    print("  - White noise (flat spectrum)")
    print("  - Pink noise (1/f spectrum)")
    print("  - Swept sine (chirp)")
    print("  - Multi-tonal (multiple frequencies)")
    print()
    print("Use with test_b.py to test ANC system")

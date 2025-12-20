"""Acoustic path simulation for ANC system."""

import numpy as np
from scipy import signal


def generate_primary_path(length=128, delay=5, decay=0.95, add_noise=True):
    """
    Generate primary acoustic path impulse response P(z).
    
    This represents the path from noise source to error microphone.
    
    Parameters:
    -----------
    length : int
        Length of impulse response in samples
    delay : int
        Initial delay in samples
    decay : float
        Decay factor (0-1), controls how quickly IR decays
    add_noise : bool
        Add random variations to make it more realistic
        
    Returns:
    --------
    np.ndarray
        Primary path impulse response
    """
    h = np.zeros(length)
    
    # Initial impulse
    h[delay] = 1.0
    
    # Exponential decay with random variations
    for i in range(delay + 1, length):
        h[i] = h[i-1] * decay
        
        if add_noise:
            # Add small random variations (±10%)
            h[i] *= (1 + 0.1 * np.random.randn())
    
    # Normalize
    h = h / np.max(np.abs(h))
    
    return h


def generate_secondary_path(length=128, delay=3, decay=0.9, add_noise=True):
    """
    Generate secondary acoustic path impulse response S(z).
    
    This represents the path from speaker (anti-noise) to error microphone.
    
    Parameters:
    -----------
    length : int
        Length of impulse response
    delay : int
        Initial delay (usually shorter than primary)
    decay : float
        Decay factor
    add_noise : bool
        Add random variations
        
    Returns:
    --------
    np.ndarray
        Secondary path impulse response
    """
    h = np.zeros(length)
    
    # Initial impulse
    h[delay] = 1.0
    
    # Exponential decay
    for i in range(delay + 1, length):
        h[i] = h[i-1] * decay
        
        if add_noise:
            h[i] *= (1 + 0.1 * np.random.randn())
    
    # Normalize
    h = h / np.max(np.abs(h))
    
    return h


def generate_realistic_path(length=256, room_size='small'):
    """
    Generate more realistic impulse response based on room acoustics.
    
    Parameters:
    -----------
    length : int
        IR length in samples
    room_size : str
        'small', 'medium', or 'large'
        
    Returns:
    --------
    np.ndarray
        Realistic impulse response
    """
    # Room parameters
    params = {
        'small': {'delay': 5, 'rt60': 0.2, 'early_reflections': 3},
        'medium': {'delay': 10, 'rt60': 0.4, 'early_reflections': 5},
        'large': {'delay': 15, 'rt60': 0.8, 'early_reflections': 8}
    }
    
    p = params[room_size]
    h = np.zeros(length)
    
    # Direct path
    h[p['delay']] = 1.0
    
    # Early reflections
    for i in range(p['early_reflections']):
        reflection_delay = p['delay'] + 10 + i * 5
        reflection_amp = 0.7 * (0.8 ** i)
        if reflection_delay < length:
            h[reflection_delay] += reflection_amp
    
    # Late reverberation (exponential decay)
    sample_rate = 44100
    decay_rate = np.log(1000) / (p['rt60'] * sample_rate)  # T60 decay
    
    reverb_start = p['delay'] + p['early_reflections'] * 5 + 10
    for i in range(reverb_start, length):
        t = (i - reverb_start) / sample_rate
        h[i] = np.exp(-decay_rate * t) * np.random.randn() * 0.1
    
    # Normalize
    h = h / np.max(np.abs(h))
    
    return h


def apply_path(signal, impulse_response):
    """
    Apply acoustic path to signal (convolution).
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    impulse_response : np.ndarray
        Path impulse response
        
    Returns:
    --------
    np.ndarray
        Filtered signal
    """
    # Convolve signal with IR
    output = np.convolve(signal, impulse_response, mode='same')
    return output


def create_path_mismatch(true_ir, mismatch_percent=10):
    """
    Create a mismatched estimate of the impulse response.
    
    Used to test robustness when secondary path estimate is inaccurate.
    
    Parameters:
    -----------
    true_ir : np.ndarray
        True impulse response
    mismatch_percent : float
        Percentage of mismatch (0-100)
        
    Returns:
    --------
    np.ndarray
        Mismatched IR estimate
    """
    # Add random noise proportional to mismatch
    noise_level = mismatch_percent / 100.0
    noise = noise_level * np.random.randn(len(true_ir))
    
    mismatched_ir = true_ir + noise * np.max(np.abs(true_ir))
    
    # Ensure it's still reasonably normalized
    mismatched_ir = mismatched_ir / np.max(np.abs(mismatched_ir)) * np.max(np.abs(true_ir))
    
    return mismatched_ir


def plot_impulse_response(ir, sample_rate=44100, title="Impulse Response"):
    """
    Plot impulse response in time and frequency domain.
    
    Parameters:
    -----------
    ir : np.ndarray
        Impulse response
    sample_rate : int
        Sample rate
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time domain
    t = np.arange(len(ir)) / sample_rate * 1000  # Convert to ms
    ax1.plot(t, ir)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title} - Time Domain')
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain
    freqs = np.fft.rfftfreq(len(ir), 1/sample_rate)
    magnitude = np.abs(np.fft.rfft(ir))
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    ax2.plot(freqs, magnitude_db)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title(f'{title} - Frequency Response')
    ax2.set_xlim([0, sample_rate/2])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Acoustic Path Simulation Module")
    print("=" * 60)
    print()
    print("Available functions:")
    print("  - generate_primary_path(): P(z) from source to error mic")
    print("  - generate_secondary_path(): S(z) from speaker to error mic")
    print("  - apply_path(): Convolve signal with IR")
    print("  - create_path_mismatch(): Simulate modeling errors")
    print()
    print("Use with test_b.py to test ANC system")

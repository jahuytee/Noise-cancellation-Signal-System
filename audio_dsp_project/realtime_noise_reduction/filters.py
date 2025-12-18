"""Baseline audio filters: band-pass and notch filters."""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from .utils import plot_spectrum, plot_spectrogram, estimate_snr_from_audio


def apply_bandpass_filter(audio_data, sample_rate, lowcut=80, highcut=8000, order=5):
    """
    Apply band-pass filter to keep only frequencies in specified range.
    
    This is useful for speech enhancement, keeping only the speech frequency range
    (typically 80-8000 Hz) and removing very low and very high frequency noise.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    lowcut : float
        Low cutoff frequency in Hz (default: 80 Hz)
    highcut : float
        High cutoff frequency in Hz (default: 8000 Hz)
    order : int
        Filter order (default: 5, higher = sharper cutoff)
        
    Returns:
    --------
    np.ndarray
        Filtered audio signal
    """
    # Normalize frequencies to Nyquist frequency (half sample rate)
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth band-pass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter (filtfilt for zero-phase filtering)
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    return filtered_audio


def apply_notch_filter(audio_data, sample_rate, notch_freq=60, quality=30):
    """
    Apply notch filter to remove specific frequency (e.g., 60 Hz power line hum).
    
    A notch filter removes a very narrow band of frequencies around the target
    frequency, useful for removing electrical hum or other tonal interference.
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    notch_freq : float
        Frequency to remove in Hz (default: 60 Hz)
    quality : float
        Quality factor (Q), higher = narrower notch (default: 30)
        
    Returns:
    --------
    np.ndarray
        Filtered audio signal
    """
    # Normalize frequency to Nyquist frequency
    nyquist = 0.5 * sample_rate
    freq = notch_freq / nyquist
    
    # Design notch filter
    b, a = signal.iirnotch(freq, quality)
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    return filtered_audio


def apply_cascaded_notch(audio_data, sample_rate, notch_freqs=[60, 120], quality=30):
    """
    Apply multiple notch filters in cascade.
    
    Useful for removing 60 Hz power line noise and its harmonics (120 Hz, 180 Hz, etc.)
    
    Parameters:
    -----------
    audio_data : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    notch_freqs : list of float
        Frequencies to remove in Hz (default: [60, 120])
    quality : float
        Quality factor (Q) for each notch (default: 30)
        
    Returns:
    --------
    np.ndarray
        Filtered audio signal
    """
    filtered_audio = audio_data.copy()
    
    for notch_freq in notch_freqs:
        filtered_audio = apply_notch_filter(filtered_audio, sample_rate, notch_freq, quality)
    
    return filtered_audio


def compare_audio_filtering(original_audio, filtered_audio, sample_rate, 
                            filter_name="Filtered", save_path=None):
    """
    Create side-by-side comparison plots of original vs filtered audio.
    
    Shows spectrum and spectrogram comparisons to visualize filtering effects.
    
    Parameters:
    -----------
    original_audio : np.ndarray
        Original audio signal
    filtered_audio : np.ndarray
        Filtered audio signal
    sample_rate : int
        Sample rate in Hz
    filter_name : str
        Name of the filter for plot titles (default: "Filtered")
    save_path : str, optional
        Path to save the comparison plot
        
    Returns:
    --------
    tuple
        (figure, axes) matplotlib objects
    """
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original spectrum
    plot_spectrum(original_audio, sample_rate, title="Original - Spectrum", ax=axes[0, 0])
    
    # Filtered spectrum
    plot_spectrum(filtered_audio, sample_rate, title=f"{filter_name} - Spectrum", ax=axes[0, 1])
    
    # Original spectrogram
    plot_spectrogram(original_audio, sample_rate, title="Original - Spectrogram", ax=axes[1, 0])
    
    # Filtered spectrogram
    plot_spectrogram(filtered_audio, sample_rate, title=f"{filter_name} - Spectrogram", ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig, axes


def compute_snr_improvement(original_audio, filtered_audio, sample_rate):
    """
    Compute SNR improvement from filtering.
    
    Parameters:
    -----------
    original_audio : np.ndarray
        Original audio signal
    filtered_audio : np.ndarray
        Filtered audio signal
    sample_rate : int
        Sample rate in Hz
        
    Returns:
    --------
    dict
        Dictionary with 'original_snr', 'filtered_snr', and 'improvement' in dB
    """
    original_snr = estimate_snr_from_audio(original_audio, sample_rate)
    filtered_snr = estimate_snr_from_audio(filtered_audio, sample_rate)
    improvement = filtered_snr - original_snr
    
    return {
        'original_snr': original_snr,
        'filtered_snr': filtered_snr,
        'improvement': improvement
    }


def plot_filter_response(filter_type='bandpass', sample_rate=44100, **kwargs):
    """
    Plot the frequency response of a filter.
    
    Parameters:
    -----------
    filter_type : str
        'bandpass' or 'notch'
    sample_rate : int
        Sample rate in Hz
    **kwargs : dict
        Parameters for the specific filter type
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if filter_type == 'bandpass':
        lowcut = kwargs.get('lowcut', 80)
        highcut = kwargs.get('highcut', 8000)
        order = kwargs.get('order', 5)
        
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        
        title = f'Band-pass Filter Response ({lowcut}-{highcut} Hz, Order {order})'
        
    elif filter_type == 'notch':
        notch_freq = kwargs.get('notch_freq', 60)
        quality = kwargs.get('quality', 30)
        
        nyquist = 0.5 * sample_rate
        freq = notch_freq / nyquist
        b, a = signal.iirnotch(freq, quality)
        
        title = f'Notch Filter Response ({notch_freq} Hz, Q={quality})'
    
    else:
        raise ValueError("filter_type must be 'bandpass' or 'notch'")
    
    # Compute frequency response
    w, h = signal.freqz(b, a, worN=8000)
    freq_hz = w * sample_rate / (2 * np.pi)
    
    # Plot magnitude response in dB
    magnitude_db = 20 * np.log10(abs(h) + 1e-10)
    ax.plot(freq_hz, magnitude_db, linewidth=2)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, sample_rate / 2)
    
    # Add reference lines
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')
    ax.axhline(-20, color='orange', linestyle='--', alpha=0.5, label='-20 dB')
    ax.legend()
    
    plt.tight_layout()
    
    return ax


if __name__ == "__main__":
    # Demo: Show filter responses
    print("Filter Response Demonstrations")
    print("=" * 60)
    
    # Band-pass filter response
    print("\n1. Band-pass Filter (80-8000 Hz)")
    plot_filter_response('bandpass', lowcut=80, highcut=8000, order=5)
    plt.savefig('results/plots/bandpass_response.png', dpi=150, bbox_inches='tight')
    print("   Saved to: results/plots/bandpass_response.png")
    
    # Notch filter response
    print("\n2. Notch Filter (60 Hz)")
    plot_filter_response('notch', notch_freq=60, quality=30)
    plt.savefig('results/plots/notch_response.png', dpi=150, bbox_inches='tight')
    print("   Saved to: results/plots/notch_response.png")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Filter demonstrations complete!")

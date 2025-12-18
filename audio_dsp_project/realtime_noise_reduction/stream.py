"""Real-time audio streaming with FFT visualization."""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import queue
from .utils import save_wav


class AudioStreamer:
    """Real-time audio streaming and visualization class."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, duration=10):
        """
        Initialize audio streamer.
        
        Parameters:
        -----------
        sample_rate : int
            Sample rate in Hz (default: 44100)
        chunk_size : int
            Number of samples per chunk (default: 2048)
        duration : int
            Recording duration in seconds (default: 10)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.duration = duration
        
        # Audio buffer
        self.audio_buffer = []
        self.audio_queue = queue.Queue()
        
        # For real-time plotting
        self.time_data = deque(maxlen=chunk_size)
        self.freq_data = deque(maxlen=chunk_size // 2 + 1)
        
        # Initialize with zeros
        self.time_data.extend([0] * chunk_size)
        self.freq_data.extend([0] * (chunk_size // 2 + 1))
        
        # Compute frequency axis
        self.freqs = np.fft.rfftfreq(chunk_size, 1/sample_rate)
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for sounddevice stream.
        
        Called automatically when new audio data is available.
        """
        if status:
            print(f"Status: {status}")
        
        # Copy data to avoid issues
        audio_chunk = indata[:, 0].copy()
        
        # Add to buffer
        self.audio_buffer.append(audio_chunk)
        
        # Add to queue for plotting
        self.audio_queue.put(audio_chunk)
        
    def start_recording(self):
        """Start audio recording."""
        print(f"Recording for {self.duration} seconds...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk size: {self.chunk_size} samples")
        
        self.is_recording = True
        self.audio_buffer = []
        
        # Start recording stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        ):
            sd.sleep(int(self.duration * 1000))
        
        self.is_recording = False
        print("Recording finished!")
        
    def get_audio_data(self):
        """
        Get recorded audio as numpy array.
        
        Returns:
        --------
        np.ndarray
            Complete recorded audio
        """
        if len(self.audio_buffer) == 0:
            return np.array([])
        
        return np.concatenate(self.audio_buffer)
    
    def update_plot(self, frame):
        """
        Update function for matplotlib animation.
        
        Parameters:
        -----------
        frame : int
            Frame number (unused, required by FuncAnimation)
        """
        # Get latest audio chunk from queue
        try:
            while not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get_nowait()
                
                # Update time-domain data
                self.time_data.extend(audio_chunk)
                
                # Compute FFT
                fft_data = np.fft.rfft(audio_chunk)
                magnitude = np.abs(fft_data)
                
                # Convert to dB
                magnitude_db = 20 * np.log10(magnitude + 1e-10)
                
                # Update frequency-domain data
                self.freq_data = magnitude_db
                
        except queue.Empty:
            pass
        
        # Update waveform plot
        self.line_time.set_ydata(list(self.time_data))
        
        # Update spectrum plot
        self.line_freq.set_ydata(self.freq_data)
        
        return self.line_time, self.line_freq
    
    def record_with_visualization(self, output_file=None):
        """
        Record audio with real-time visualization.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save recorded audio as WAV
        """
        # Create figure with two subplots
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time-domain plot
        time_axis = np.arange(self.chunk_size) / self.sample_rate
        self.line_time, = self.ax_time.plot(time_axis, list(self.time_data))
        self.ax_time.set_ylim(-1, 1)
        self.ax_time.set_xlabel('Time (s)')
        self.ax_time.set_ylabel('Amplitude')
        self.ax_time.set_title('Real-Time Waveform')
        self.ax_time.grid(True, alpha=0.3)
        
        # Frequency-domain plot
        self.line_freq, = self.ax_freq.plot(self.freqs, list(self.freq_data))
        self.ax_freq.set_ylim(-80, 0)
        self.ax_freq.set_xlim(0, self.sample_rate / 2)
        self.ax_freq.set_xlabel('Frequency (Hz)')
        self.ax_freq.set_ylabel('Magnitude (dB)')
        self.ax_freq.set_title('Real-Time Spectrum (FFT)')
        self.ax_freq.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Start recording in separate thread
        self.recording_thread = threading.Thread(target=self.start_recording)
        self.recording_thread.start()
        
        # Start animation
        anim = FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=50,  # Update every 50ms
            blit=True,
            cache_frame_data=False
        )
        
        plt.show()
        
        # Wait for recording to finish
        self.recording_thread.join()
        
        # Get the complete audio
        audio_data = self.get_audio_data()
        
        # Save if requested
        if output_file:
            save_wav(output_file, audio_data, self.sample_rate)
        
        return audio_data


def record_audio(duration=10, sample_rate=44100, visualize=True, output_file=None):
    """
    Convenience function to record audio.
    
    Parameters:
    -----------
    duration : int
        Recording duration in seconds
    sample_rate : int
        Sample rate in Hz
    visualize : bool
        Whether to show real-time visualization
    output_file : str, optional
        Path to save WAV file
        
    Returns:
    --------
    np.ndarray
        Recorded audio data
    """
    streamer = AudioStreamer(sample_rate=sample_rate, duration=duration)
    
    if visualize:
        audio_data = streamer.record_with_visualization(output_file)
    else:
        streamer.start_recording()
        audio_data = streamer.get_audio_data()
        if output_file:
            save_wav(output_file, audio_data, sample_rate)
    
    return audio_data


if __name__ == "__main__":
    # Test the streamer
    print("Testing audio streamer...")
    print("Speak into your microphone!")
    
    audio = record_audio(
        duration=5,
        sample_rate=44100,
        visualize=True,
        output_file="test_recording.wav"
    )
    
    print(f"\nRecorded {len(audio)} samples ({len(audio)/44100:.2f} seconds)")

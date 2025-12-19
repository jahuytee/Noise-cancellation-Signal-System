# Audio DSP Project: Real-Time Noise Reduction + Simulated ANC

A comprehensive audio digital signal processing system implementing real-time noise reduction and simulated Active Noise Cancellation using the FxLMS algorithm.

## 🎯 Project Overview

This project demonstrates practical DSP techniques with two main components:

### Phase A: Real-Time Noise Reduction
- **A0**: Mic streaming with real-time FFT visualization
- **A1**: Baseline filters (band-pass, notch)
- **A2**: Advanced noise reduction (spectral subtraction, Wiener filtering)
- **A3**: CLI integration

### Phase B: Simulated Active Noise Cancellation
- **B0**: ANC simulation environment
- **B1**: FxLMS adaptive algorithm implementation
- **B2**: Parameter sweep experiments

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Working microphone

### Installation

```bash
# Navigate to project directory
cd audio_dsp_project

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

Test your microphone and see real-time visualization:

```bash
# From the audio_dsp_project directory
python -m realtime_noise_reduction.stream
```

This will:
- Record 5 seconds of audio from your microphone
- Show real-time waveform and FFT spectrum
- Save output as `test_recording.wav`

## 📁 Project Structure

```
audio_dsp_project/
├── realtime_noise_reduction/
│   ├── stream.py              # Mic streaming + visualization ✅
│   ├── utils.py               # SNR, plotting, WAV utilities ✅
│   ├── filters.py             # Band-pass, notch filters ✅
│   ├── spectral_subtraction.py # Spectral subtraction ✅
│   └── wiener.py              # Wiener filtering ✅
├── simulated_anc_fxlms/
│   ├── simulate.py            # (coming soon)
│   ├── paths.py               # (coming soon)
│   ├── fxlms.py               # (coming soon)
│   └── noise_gen.py           # (coming soon)
├── results/                   # Output audio and plots
├── requirements.txt
├── test_a0.py                # Phase A0 verification ✅
├── test_a1.py                # Phase A1 verification ✅
├── test_a2.py                # Phase A2 verification ✅
└── README.md
```

## 📊 Current Status

✅ **Phase A0 Complete**
- [x] Mic audio streaming
- [x] Real-time waveform plotting
- [x] Real-time FFT spectrum plotting
- [x] WAV file saving
- [x] Utility functions (SNR, spectrograms)

✅ **Phase A1 Complete**
- [x] Band-pass filter (80-8000 Hz)
- [x] Notch filters (60 Hz, 120 Hz)
- [x] Before/after comparisons
- [x] SNR improvement metrics
- [x] Filter response visualization

✅ **Phase A2 Complete**
- [x] Spectral subtraction (basic + adaptive)
- [x] Wiener filter (basic + adaptive)
- [x] Noise PSD estimation
- [x] Method comparison tools
- [x] Advanced SNR metrics

🔄 **In Progress**
- [ ] Phase A3: CLI integration
- [ ] Phase B: FxLMS simulation

## 🛠️ Technologies

- **numpy**: Array operations and FFT
- **scipy**: Signal processing (filters, windowing)
- **sounddevice**: Real-time audio I/O
- **matplotlib**: Visualization and spectrograms

## 📝 Usage Examples

### Record audio with visualization
```python
from realtime_noise_reduction.stream import record_audio

# Record 10 seconds with real-time plots
audio = record_audio(
    duration=10,
    sample_rate=44100,
    visualize=True,
    output_file="results/my_recording.wav"
)
```

### Compute SNR
```python
from realtime_noise_reduction.utils import estimate_snr_from_audio

snr_db = estimate_snr_from_audio(audio, sample_rate=44100)
print(f"Estimated SNR: {snr_db:.2f} dB")
```

### Plot spectrum
```python
from realtime_noise_reduction.utils import plot_spectrum
import matplotlib.pyplot as plt

plot_spectrum(audio, sample_rate=44100, title="My Audio Spectrum")
plt.show()
```

## 🎓 Learning Objectives

- Real-time audio processing
- Frequency-domain analysis (FFT)
- Adaptive filtering techniques
- Noise estimation and reduction
- Audio signal metrics (SNR)

## 📈 Coming Soon

- Band-pass and notch filtering
- Spectral subtraction algorithm
- Wiener filtering
- FxLMS active noise cancellation
- Command-line interface
- Comprehensive results and demos

## 📄 License

This project is for educational and portfolio purposes.

---

**Author**: Jason  
**Project Type**: Audio DSP / Signal Processing / EE Fundamentals

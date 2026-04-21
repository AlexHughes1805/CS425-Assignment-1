import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import soundfile as sf
import os

from audio_io import load_audio, quantize, compute_snr, BIT_DEPTH
from fourier_analysis import compute_fft, compute_dft, apply_time_shift, apply_phase_shift, TIME_SHIFT_MS
from stft_analysis import compute_stft
from effects import apply_clipping, downsample, CLIPPING_THRESHOLD, DOWNSAMPLE_FACTOR
import time


# Create output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility: side-by-side plotting
def compare_plots(sig1, sig2, title1, title2, suptitle):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(sig1)
    plt.title(title1)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(sig2)
    plt.title(title2)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

# Utility: frequency spectrum comparison
def compare_spectra(sig1, sig2, sr, title1, title2, suptitle):
    spec1 = compute_fft(sig1)
    spec2 = compute_fft(sig2)

    freqs1 = np.fft.fftfreq(len(spec1), 1/sr)
    freqs2 = np.fft.fftfreq(len(spec2), 1/sr)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(freqs1[:len(freqs1)//2], np.abs(spec1)[:len(spec1)//2])
    plt.title(title1)
    plt.xlabel("Frequency (Hz)")

    plt.subplot(1, 2, 2)
    plt.plot(freqs2[:len(freqs2)//2], np.abs(spec2)[:len(spec2)//2])
    plt.title(title2)
    plt.xlabel("Frequency (Hz)")

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

# Load audio
signal, sr = load_audio(r'hotAndHeavy.wav')

# --- ORIGINAL ---
sf.write(os.path.join(OUTPUT_DIR, "original.wav"), signal, sr)

plt.figure()
plt.plot(signal)
plt.title("Original Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# --- QUANTISATION ---
quantized = quantize(signal)
snr = compute_snr(signal, quantized)

sf.write(os.path.join(OUTPUT_DIR, f"quantized_{BIT_DEPTH}bit.wav"), quantized, sr)

compare_plots(signal, quantized,
              "Original", f"Quantized ({BIT_DEPTH}-bit)",
              f"Quantisation Effect | SNR={snr:.2f} dB")

compare_spectra(signal, quantized, sr,
                "Original Spectrum", "Quantized Spectrum",
                "Quantisation Effect")

# --- TIME SHIFT ---

shifted = apply_time_shift(signal, sr)

sf.write(os.path.join(OUTPUT_DIR, f"time_shifted_{TIME_SHIFT_MS}ms.wav"), shifted, sr)

compare_plots(signal, shifted,
              "Original", "Time-Shifted",
              "Time Shift Effect")

compare_spectra(signal, shifted, sr,
                "Original Spectrum", "Shifted Spectrum",
                "Time Shift vs Spectrum")


# --- PHASE SHIFT (FREQUENCY DOMAIN) ---

# FFT
spectrum = compute_fft(signal)

# Apply phase shift in frequency domain
shifted_spectrum = apply_phase_shift(spectrum, sr)

# Inverse FFT to get time-domain signal
shifted_phase = np.fft.ifft(shifted_spectrum).real

# Save output
sf.write(os.path.join(OUTPUT_DIR, f"phase_shifted_{TIME_SHIFT_MS}ms.wav"),
         shifted_phase, sr)

# Compare with original
compare_plots(signal, shifted_phase,
              "Original", f"Phase Shifted ({TIME_SHIFT_MS} ms)",
              "Phase Shift via Frequency Domain")

compare_spectra(signal, shifted_phase, sr,
                "Original Spectrum", "Phase Shifted Spectrum",
                "Phase Shift Effect")

# --- STFT ---
stft = compute_stft(signal)

plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(abs(stft)),
                         sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram (Original)")
plt.show()



# --- DFT vs FFT ---
small_signal = signal[:1024]  # keep it small (DFT is slow)

# DFT timing
start = time.time()
dft_result = compute_dft(small_signal)
dft_time = time.time() - start

# FFT timing
start = time.time()
fft_result = compute_fft(small_signal)
fft_time = time.time() - start

print(f"DFT Time: {dft_time:.4f} seconds")
print(f"FFT Time: {fft_time:.6f} seconds")


# Plot comparison
N = len(small_signal)
freqs = np.fft.fftfreq(N, d=1/sr)

plt.figure(figsize=(10, 4))

plt.plot(freqs[:N//2], np.abs(dft_result)[:N//2], label="DFT")
plt.plot(freqs[:N//2], np.abs(fft_result)[:N//2], linestyle='--', label="FFT")

plt.title("DFT vs FFT (Magnitude Spectrum)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude |X(f)|")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- CLIPPING ---

clipped_hard = apply_clipping(signal, mode = 'hard')

clipped_soft = apply_clipping(signal, mode = 'soft')

sf.write(f"outputs/clipped_hard{CLIPPING_THRESHOLD}.wav", clipped_hard, sr)
sf.write(f"outputs/clipped_soft{CLIPPING_THRESHOLD}.wav", clipped_soft, sr)

compare_plots(signal, clipped_hard,
              "Original", "Hard Clipping",
              "Hard Clipping Effect")

compare_spectra(signal, clipped_hard, sr,
                "Original Spectrum", "Hard Clipping Spectrum",
                "Hard Clipping Effect")

compare_plots(signal, clipped_soft,
              "Original", "Soft Clipping",
              "Hard Clipping Effect")

compare_spectra(signal, clipped_soft, sr,
                "Original Spectrum", "Soft Clipping Spectrum",
                "Soft Clipping Effect")

# --- ALIASING ---

down = downsample(signal)
new_sr = sr // DOWNSAMPLE_FACTOR

sf.write(os.path.join(OUTPUT_DIR, f"downsampled_aliasing{DOWNSAMPLE_FACTOR}.wav"), down, new_sr)

compare_plots(signal, down,
              "Original", "Downsampled",
              "Aliasing Effect (Time Domain)")

compare_spectra(signal, down, new_sr,
                "Original Spectrum", "Downsampled Spectrum",
                "Aliasing Effect (Frequency Domain)")
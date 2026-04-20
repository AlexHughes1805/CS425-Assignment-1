import numpy as np
import librosa

TARGET_SR = 44100
BIT_DEPTH = 16

def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=TARGET_SR)
    return signal, sr


def quantize(signal):
    levels = 2 ** BIT_DEPTH
    signal_q = np.round(signal * levels) / levels
    return signal_q


def compute_snr(original, quantized):
    noise = original - quantized
    snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr
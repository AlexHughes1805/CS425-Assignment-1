import numpy as np

TIME_SHIFT_MS = 0


def compute_fft(signal):
    return np.fft.fft(signal)



def apply_time_shift(signal, sr):
    """
    Apply time shift to a signal using zero-padding.

    Parameters:
    - signal: input audio signal
    - shift_ms: time shift in milliseconds (positive = delay, negative = advance)
    - sr: sampling rate (Hz)

    Returns:
    - shifted signal
    """
    
    # Convert milliseconds to samples
    shift_samples = int((TIME_SHIFT_MS / 1000.0) * sr)

    if shift_samples > 0:
        # Delay: add zeros at the beginning
        shifted = np.concatenate((np.zeros(shift_samples), signal))
    
    elif shift_samples < 0:
        # Advance: remove from beginning, pad zeros at end
        shift_samples = abs(shift_samples)
        shifted = np.concatenate((signal[shift_samples:], np.zeros(shift_samples)))
    
    else:
        shifted = signal.copy()

    return shifted


def apply_phase_shift(spectrum, sr):
    """
    Apply a phase shift equivalent to a time shift.

    Parameters:
    - spectrum: FFT of signal
    - shift_ms: time shift in milliseconds
    - sr: sampling rate

    Returns:
    - phase-shifted spectrum
    """

    N = len(spectrum)

    # Convert ms → seconds
    tau = TIME_SHIFT_MS / 1000.0

    # Frequency bins (angular frequency ω)
    freqs = np.fft.fftfreq(N, d=1/sr)
    omega = 2 * np.pi * freqs

    # Apply phase shift: exp(-j ω τ)
    phase_shift = np.exp(-1j * omega * tau)

    return spectrum * phase_shift


def compute_dft(signal):
    """
    Compute the Discrete Fourier Transform (DFT) manually.

    Parameters:
    - signal: input signal (1D array)

    Returns:
    - DFT of the signal
    """
    N = len(signal)
    X = np.zeros(N, dtype=complex)

    for k in range(N):  # frequency bins
        for n in range(N):  # time samples
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

    return X
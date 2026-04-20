import librosa

FRAME_SIZE = 1024
HOP_LENGTH = 512
WINDOW_TYPE = 'hann'
# A list of possible window types can be find in the following link:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window


def compute_stft(signal):
    return librosa.stft(signal, n_fft=FRAME_SIZE,
                        hop_length=HOP_LENGTH,
                        window=WINDOW_TYPE)


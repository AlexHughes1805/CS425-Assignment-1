import numpy as np

CLIPPING_THRESHOLD = 0.8
DOWNSAMPLE_FACTOR = 2


import numpy as np

def apply_clipping(signal, mode='hard'):
    """
    Apply clipping to an audio signal.

    Parameters:
    - signal: input audio signal
    - threshold: clipping threshold (0 to 1)
    - mode: 'hard' or 'soft'

    Returns:
    - clipped signal
    """

    if mode == 'hard':
        # Hard clipping: abrupt cutoff
        clipped = np.clip(signal, -CLIPPING_THRESHOLD, CLIPPING_THRESHOLD)

    elif mode == 'soft':
        # Soft clipping: smooth non-linear compression
        # Using tanh for smooth saturation
        clipped = np.tanh(signal / CLIPPING_THRESHOLD) * CLIPPING_THRESHOLD

    else:
        raise ValueError("mode must be 'hard' or 'soft'")

    return clipped


def downsample(signal):
    return signal[::DOWNSAMPLE_FACTOR]
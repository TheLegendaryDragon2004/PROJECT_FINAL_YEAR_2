import librosa
import numpy as np
import random

def extract_features(
    audio_path,
    sr=16000,
    n_mfcc=40,
    max_len=250,
    augment=True
):
    y, _ = librosa.load(audio_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=30)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=160
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.stack([mfcc, delta, delta2], axis=0)  # (3, F, T)
    features = np.transpose(features, (2, 0, 1))       # (T, 3, F)

    # CMVN
    mean = features.mean(axis=(0,2), keepdims=True)
    std = features.std(axis=(0,2), keepdims=True) + 1e-6
    features = (features - mean) / std

    # SpecAugment (TRAIN ONLY)
    if augment:
        T, C, F = features.shape
        t_mask = random.randint(0, int(0.08 * T))
        t0 = random.randint(0, max(1, T - t_mask))
        features[t0:t0+t_mask] = 0

        f_mask = random.randint(0, int(0.08 * F))
        f0 = random.randint(0, max(1, F - f_mask))
        features[:, :, f0:f0+f_mask] = 0

    # Pad / truncate
    if features.shape[0] > max_len:
        features = features[:max_len]
    else:
        pad = max_len - features.shape[0]
        features = np.pad(features, ((0,pad),(0,0),(0,0)))

    return features.astype(np.float32)

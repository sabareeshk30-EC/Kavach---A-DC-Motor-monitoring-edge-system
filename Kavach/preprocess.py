import numpy as np
import librosa

from config import (
    SAMPLE_RATE,
    CLIP_DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)


def load_audio(path):
    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    target_length = SAMPLE_RATE * CLIP_DURATION

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
        )
    else:
        audio = audio[:target_length]

    return audio.astype(np.float32)


def audio_to_logmel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mean = logmel.mean()
    std = logmel.std() + 1e-8

    logmel = (logmel - mean) / std

    return logmel.astype(np.float32)


def preprocess_audio(path):
    audio = load_audio(path)
    logmel = audio_to_logmel(audio)

    # CNN input:
    # batch x channel x frequency x time

    tensor = logmel[np.newaxis, np.newaxis, :, :]

    return tensor.astype(np.float32)

# File: 'src/controllerUtils/plot.py'
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display

# Make `src/` importable for the project layout
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from controllerUtils.energy_scorer import (
    detect_energy
)

def _soft_spectral_gate_denoise(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    vad_mode: int,
    hop_ms: int,
    noise_reduction_db: float = 12.0,
    mask_slope_db: float = 6.0,
) -> np.ndarray:
    """
    Soft spectral gating using a noise profile estimated from VAD non-speech frames.
    Falls back to a global low-percentile noise estimate if non-speech frames are scarce.
    """
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    phase = np.exp(1j * np.angle(D))

    # VAD mask aligned to hop_ms
    vad_mask, _, vad_hop = webrtc_speech_mask(y, sr, mode=vad_mode, frame_ms=10, hop_ms=hop_ms)
    # Align lengths
    L = min(mag.shape[1], len(vad_mask))
    mag = mag[:, :L]
    phase = phase[:, :L]
    vad_mask = vad_mask[:L]
    non_speech = ~vad_mask

    # Noise profile
    eps = 1e-10
    if np.sum(non_speech) >= int(0.05 * L):  # enough non-speech frames
        noise_mag = np.median(mag[:, non_speech], axis=1, keepdims=True)
    else:
        # Fallback: robust floor across all frames
        noise_mag = np.percentile(mag, 10, axis=1, keepdims=True)

    # Mask in dB space: sigmoid around (noise + reduction)
    S_db = librosa.amplitude_to_db(mag + eps, ref=1.0)
    N_db = librosa.amplitude_to_db(noise_mag + eps, ref=1.0)
    thr_db = N_db + noise_reduction_db
    mask = 1.0 / (1.0 + np.exp(-(S_db - thr_db) / max(1e-6, mask_slope_db)))
    S_clean = mag * mask

    # Reconstruct
    D_clean = S_clean * phase
    y_clean = librosa.istft(D_clean, hop_length=hop_length, win_length=win_length, length=len(y))
    return y_clean.astype(np.float32)

def _preprocess_for_plot(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    vad_mode: int,
    hop_ms: int,
    preemph_coef: float = 0.97,
    noise_reduction_db: float = 12.0,
    mask_slope_db: float = 6.0,
    rms_target_dbfs: float = -23.0,
) -> np.ndarray:
    """
    DC removal -> pre-emphasis (high-pass) -> soft spectral gating -> RMS normalize.
    """
    if len(y) == 0:
        return y

    # DC removal
    y = y - float(np.mean(y))

    # Pre-emphasis (simple high-pass)
    # y[n] = y[n] - a * y[n-1]
    y = np.append(y[0], y[1:] - preemph_coef * y[:-1])

    # Spectral denoise
    y = _soft_spectral_gate_denoise(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        vad_mode=vad_mode,
        hop_ms=hop_ms,
        noise_reduction_db=noise_reduction_db,
        mask_slope_db=mask_slope_db,
    )

    # RMS normalization to target dBFS
    eps = 1e-9
    rms = float(np.sqrt(np.mean(np.square(y)) + eps))
    target_amp = 10.0 ** (rms_target_dbfs / 20.0)
    if rms > 0:
        gain = min(1.0, target_amp / rms)
        y = np.clip(y * gain, -1.0, 1.0)

    return y.astype(np.float32)


def plot_speech_vs_noise_spectrogram(
    wav_path: str,
    out_path: str = "spectrogram_overlay.png",
    # Time/frequency resolution:
    sr: int = 16000,
    n_fft: int = 1024,       # increase for better freq resolution (e.g., 2048)
    hop_ms: int = 10,        # 10 ms hop; increase for faster/rougher view
    win_ms: int = 25,        # 25 ms window; 20–30 ms is typical for speech
    # Display dynamic range:
    top_db: float = 80.0,    # clamp low-energy floor (bigger => more background visible)
    # VAD + noise detection:
    vad_mode: int = 3,       # 0..3 (3 is most aggressive)
    speech_overlap_th: float = 0.30,  # event fraction overlapping speech to tag as masking
    # Frequency view:
    fmax: int = 8000,  # show up to 8 kHz for wide-band speech
    n_mels: int = 128,       # mel bins
    cmap: str = "magma",
    # Cleaning toggles:
    denoise: bool = True,
    preemph_coef: float = 0.97,
    noise_reduction_db: float = 12.0,
    mask_slope_db: float = 6.0,
    rms_target_dbfs: float = -23.0,
):
    # 1) Load audio
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    # 2) Optional cleanup before graphing
    hop_length = int(sr * hop_ms / 1000)
    win_length = int(sr * win_ms / 1000)
    if denoise:
        y = _preprocess_for_plot(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            vad_mode=vad_mode,
            hop_ms=hop_ms,
            preemph_coef=preemph_coef,
            noise_reduction_db=noise_reduction_db,
            mask_slope_db=mask_slope_db,
            rms_target_dbfs=rms_target_dbfs,
        )

    # 3) Compute log-mel spectrogram from cleaned audio
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=0,
        fmax=fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.maximum(S_db, np.max(S_db) - top_db)  # clamp floor

    energy_segments = detect_energy(wav_path)

    # 6) Plot spectrogram + overlays
    t_axis = np.arange(S_db.shape[1]) * (hop_length / sr)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=120)
    img = librosa.display.specshow(
        S_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        fmax=fmax,
        cmap=cmap,
        ax=ax,
    )
    cbar = fig.colorbar(img, ax=ax, format="%+0.0f dB")
    cbar.set_label("dB")

    ax.set_ylim(0, float(fmax))

    y0, y1 = ax.get_ylim()
    for ev in energy_segments:
        s, e = float(ev["start_time"]), float(ev["end_time"])
        label = ev.get("label", "normal_speech")
        if label == "loud_noise":
            ax.add_patch(
                patches.Rectangle(
                    (s, y0),
                    width=max(1e-6, e - s),
                    height=(y1 - y0),
                    linewidth=1.0,
                    linestyle="--",
                    edgecolor=(0.1, 0.9, 0.2, 1.0),
                    facecolor=(0.1, 0.9, 0.2, 0.25),
                )
            )
        lbl = f"{ev.get('label_detail', 'noise')}:{ev.get('score', 0):.2f}"
        ax.text(s, y1, lbl, va="bottom", ha="left", fontsize=8, color=(0.1, 0.6, 0.9, 1.0))

    ax.set_xlim(0, t_axis[-1] if len(t_axis) else (len(y) / sr))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path
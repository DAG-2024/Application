# File: 'src/controllerUtils/plot.py'
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display
import webrtcvad

def _webrtc_speech_mask(
    y: np.ndarray,
    sr: int,
    mode: int = 3,
    frame_ms: int = 10,
    hop_ms: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Voice Activity Detection (VAD) using WebRTC VAD.
    Returns a boolean mask aligned to the STFT frames.
    """
    vad = webrtcvad.Vad(mode)
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    # Pad signal to fit into frames
    num_frames = 1 + (len(y) - frame_length) // hop_length
    pad_len = (num_frames * hop_length + frame_length) - len(y)
    y_padded = np.pad(y, (0, pad_len), mode='constant')

    # Frame the signal
    frames = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length).T

    # VAD decision for each frame
    vad_mask = np.array([vad.is_speech((frame * 32768).astype(np.int16).tobytes(), sr) for frame in frames])

    return vad_mask, frames, hop_length

def _soft_spectral_gate_denoise(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    hop_ms: int,
    vad_mode: int = 3,
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
    vad_mask, _, vad_hop = _webrtc_speech_mask(y, sr, mode=vad_mode, frame_ms=10, hop_ms=hop_ms)
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

def plot_speech_spectrogram(
    wav_path: str,
    out_path: str = "spectrogram_overlay.png",
    # Add overlay of segments:
    energy_segments = None,
    anomaly_segments = None,
    low_conf_segments = None,
    conf_threshold: float = 0.5,
    # Time/frequency resolution:
    sr: int = 16000,
    n_fft: int = 1024,       # increase for better freq resolution (e.g., 2048)
    hop_ms: int = 10,        # 10 ms hop; increase for faster/rougher view
    win_ms: int = 25,        # 25 ms window; 20–30 ms is typical for speech
    # Display dynamic range:
    top_db: float = 80.0,    # clamp low-energy floor (bigger => more background visible)
    # Frequency view:
    fmax: int = 8000,  # show up to 8 kHz for wide-band speech
    n_mels: int = 128,       # mel bins
    cmap: str = "magma",
    indent_increment: float = 750.0,  # vertical offset increment for text labels
):
    # 1) Load audio
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    # 2) Optional cleanup before graphing
    hop_length = int(sr * hop_ms / 1000)
    win_length = int(sr * win_ms / 1000)

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
    indent = 0
    # Overlay energy-based segments
    if energy_segments is not None:
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
                lbl = f"{ev.get('label', 'noise')}:[{ev.get('start_time', 0):.2f}-{ev.get('end_time', 0):.2f}]"
                ax.text(s, y1, lbl, va="bottom", ha="left", fontsize=8, color=(0.1, 0.9, 0.2, 1.0))
        indent += indent_increment

    if anomaly_segments is not None:
        for ev in anomaly_segments:
            s, e = float(ev.start), float(ev.end)
            ax.add_patch(
                patches.Rectangle(
                    (s, y0),
                    width=max(1e-6, e - s),
                    height=(y1 - y0),
                    linewidth=1.0,
                    linestyle="--",
                    edgecolor=(0.1, 0.1, 0.9, 1.0),
                    facecolor=(0.1, 0.1, 0.9, 0.25),
                )
            )
            lbl = f"anomaly:[{s:.2f}-{e:.2f}]"
            ax.text(s, y1 + indent, lbl, va="bottom", ha="left", fontsize=8, color=(0.1, 0.1, 0.9, 1.0))
        indent += indent_increment

    if low_conf_segments is not None:
        for ev in low_conf_segments:
            s, e = float(ev.start), float(ev.end)
            conf = float(ev.confidence)
            if conf < conf_threshold:
                ax.add_patch(
                    patches.Rectangle(
                        (s, y0),
                        width=max(1e-6, e - s),
                        height=(y1 - y0),
                        linewidth=1.0,
                        linestyle="--",
                        edgecolor=(0.9, 0.5, 0.2, 1.0),
                        facecolor=(0.9, 0.5, 0.2, 0.25),
                    )
                )
                lbl = f"low_conf({conf:.2f}):[{s:.2f}-{e:.2f}]"
                ax.text(s, y1 + indent, lbl, va="bottom", ha="left", fontsize=8, color=(0.9, 0.5, 0.2, 1.0))

    ax.set_xlim(0, t_axis[-1] if len(t_axis) else (len(y) / sr))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path
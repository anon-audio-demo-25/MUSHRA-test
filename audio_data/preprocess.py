# preprocess.py
# Align (exact same sample count) + make stereo for MUSHRA trials.
# Usage:
#   python preprocess.py                      # uses ./audio_data and 48kHz
#   python preprocess.py --root audio_data --sr 48000 --replace

import argparse, os, sys
from pathlib import Path
import numpy as np

# These two are widely available in ML stacks
import librosa
import soundfile as sf

TARGET_FILENAMES = ["ref.wav", "ctx.wav", "wg6.wav", "bddm19.wav", "gen.wav"]

def load_audio_force_sr(path: Path, sr: int) -> np.ndarray:
    """Load as float32, target SR, keep channels (mono->(1,T)). Returns (C,T)."""
    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, T)
    return y.astype(np.float32)

def to_stereo(y: np.ndarray) -> np.ndarray:
    """Ensure (2,T). If mono, duplicate; if >2, keep first two."""
    if y.ndim != 2:
        raise ValueError("Audio array must be (C,T)")
    C, T = y.shape
    if C == 1:
        return np.vstack([y, y])
    if C >= 2:
        return y[:2, :]
    raise ValueError("Unsupported channel count")

def pad_or_trim_to(y: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim to exact target_len (samples)."""
    C, T = y.shape
    if T == target_len:
        return y
    if T > target_len:
        return y[:, :target_len]
    # pad at end
    out = np.zeros((C, target_len), dtype=y.dtype)
    out[:, :T] = y
    return out

def write_wav(path: Path, y: np.ndarray, sr: int):
    """Write PCM16 stereo WAV. y should be (C,T)."""
    # soundfile expects (T,C) for multi-channel
    sf.write(str(path), y.T, sr, subtype="PCM_16")

def find_trial_dirs(root: Path):
    """
    Yield directories that contain at least one of TARGET_FILENAMES.
    Your structure is audio_data/<Pair>/<Take>/...
    """
    for p in root.rglob("*"):
        if p.is_dir():
            if any((p / name).exists() for name in TARGET_FILENAMES):
                yield p

def choose_template_len(trial_dir: Path, sr: int) -> int:
    """Prefer ref.wav; otherwise choose the longest present file (after resample)."""
    ref = trial_dir / "ref.wav"
    candidates = [trial_dir / n for n in TARGET_FILENAMES if (trial_dir / n).exists()]
    if not candidates:
        return 0
    if ref.exists():
        y = load_audio_force_sr(ref, sr)
        return y.shape[1]
    # else pick the longest by samples
    max_len = -1
    for c in candidates:
        y = load_audio_force_sr(c, sr)
        max_len = max(max_len, y.shape[1])
    return max_len

def process_trial(trial_dir: Path, sr: int, replace: bool):
    target_len = choose_template_len(trial_dir, sr)
    if target_len == 0:
        return False, "no audio files"

    print(f"\n→ {trial_dir}")
    print(f"   target sample rate: {sr} Hz")
    print(f"   target length:      {target_len} samples")

    for name in TARGET_FILENAMES:
        in_path = trial_dir / name
        if not in_path.exists():
            continue
        try:
            y = load_audio_force_sr(in_path, sr)
            y = to_stereo(y)
            y = pad_or_trim_to(y, target_len)
        except Exception as e:
            print(f"   [SKIP] {name}: {e}")
            continue

        if replace:
            out_path = in_path  # overwrite
        else:
            out_path = trial_dir / f"{in_path.stem}_aligned.wav"

        write_wav(out_path, y, sr)
        dur = y.shape[1] / sr
        print(f"   [OK]  {name} -> {out_path.name} | stereo @ {sr}Hz | {y.shape[1]} samples ({dur:.2f}s)")
    return True, "done"

def main():
    ap = argparse.ArgumentParser(description="Align length + make stereo for MUSHRA folders.")
    ap.add_argument("--root", type=str, default="./", help="root folder containing trials")
    ap.add_argument("--sr", type=int, default=48000, help="target sample rate")
    ap.add_argument("--replace", action="store_true", help="overwrite originals instead of writing *_aligned.wav")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}")
        sys.exit(1)

    any_done = False
    for trial in find_trial_dirs(root):
        ok, _ = process_trial(trial, args.sr, args.replace)
        any_done = any_done or ok

    if not any_done:
        print("No matching trial folders found.")
    else:
        print("\nAll done. Point your YAML at *_aligned.wav (or use --replace next time).")

if __name__ == "__main__":
    main()

# # from the folder that contains audio_data/ and preprocess.py
# python preprocess.py
# # or, to overwrite originals (after you’ve checked *_aligned.wav sound right)
# python preprocess.py --replace
# # custom SR (if you prefer 44100)
# python preprocess.py --sr 44100
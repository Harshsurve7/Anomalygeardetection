from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DAY2_DIR = ROOT / "data" / "raw" / "day_2test"
DAY3_DIR = ROOT / "data" / "raw" / "day_3test"
RESULT_DIR = ROOT / "results" / "event_plots"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------
# IMPORTANT:
# These are the five recordings we want to inspect first.
# ---------------------------------------------------------
FILES = [
   (DAY2_DIR, "UMIK2299_32.wav"),

]
# ---------------------------------------------------------
# AUDIO
# ---------------------------------------------------------
SR = 16000
# ---------------------------------------------------------
# DETECT CANDIDATES
# ---------------------------------------------------------
def detect_candidates(y, sr):
   rms = librosa.feature.rms(
       y=y,
       frame_length=2048,
       hop_length=512
   )[0]
   rms_db = librosa.amplitude_to_db(
       rms,
       ref=np.max
   )
   rms_norm = (
       rms_db - rms_db.min()
   ) / (
       rms_db.max() - rms_db.min() + 1e-12
   )
   drop = -np.gradient(rms_norm)
   kernel = np.ones(9) / 9.0
   drop_smooth = np.convolve(
       drop,
       kernel,
       mode="same"
   )
   from scipy.signal import find_peaks
   distance = int(
       3.0 * sr / 512
   )
   peaks, _ = find_peaks(
       drop_smooth,
       distance=distance,
       prominence=0.015
   )
   times = librosa.frames_to_time(
       peaks,
       sr=sr,
       hop_length=512
   )
   duration = len(y) / sr
   valid = (
       (times >= 3.0) &
       (times <= duration - 2.0)
   )
   return list(times[valid])

# ---------------------------------------------------------
# ACOUSTIC STRENGTH
# ---------------------------------------------------------
def candidate_strength(y, sr, t):
   center = int(t * sr)
   before_start = max(
       0,
       center - int(0.8 * sr)
   )
   before_end = max(
       0,
       center - int(0.15 * sr)
   )
   after_start = min(
       len(y),
       center + int(0.15 * sr)
   )
   after_end = min(
       len(y),
       center + int(0.8 * sr)
   )
   if (
       before_end <= before_start
       or after_end <= after_start
   ):
       return 0.0
   before = y[
       before_start:before_end
   ]
   after = y[
       after_start:after_end
   ]
   before_rms = np.sqrt(
       np.mean(before ** 2) + 1e-12
   )
   after_rms = np.sqrt(
       np.mean(after ** 2) + 1e-12
   )
   before_db = 20 * np.log10(
       before_rms + 1e-12
   )
   after_db = 20 * np.log10(
       after_rms + 1e-12
   )
   return abs(
       after_db - before_db
   )

# ---------------------------------------------------------
# REUSE THE SAME SEQUENCE OPTIMIZER
# ---------------------------------------------------------
import sys
sys.path.insert(
   0,
   str(ROOT / "src")
)
from test_day2_day3_optimizer import (
   find_best_7_sequence
)

# ---------------------------------------------------------
# PLOT ONE RECORDING
# ---------------------------------------------------------
def plot_file(directory, filename):
   path = directory / filename
   if not path.exists():
       print(
           f"NOT FOUND: {path}"
       )
       return
   print()
   print("=" * 70)
   print(filename)
   print("=" * 70)
   y, sr = librosa.load(
       path,
       sr=SR,
       mono=True
   )
   duration = len(y) / sr
   candidates = detect_candidates(
       y,
       sr
   )
   strengths = [
       candidate_strength(
           y,
           sr,
           t
       )
       for t in candidates
   ]
   sequence, score = find_best_7_sequence(
       candidates,
       strengths
   )
   print(
       "Selected events:"
   )
   if sequence:
       for i, t in enumerate(
           sequence,
           start=1
       ):
           print(
               f"  Event {i}: {t:.2f}s"
           )
   print(
       f"Score: {score:.4f}"
   )
   # -----------------------------------------------------
   # SPECTROGRAM
   # -----------------------------------------------------
   D = librosa.stft(
       y,
       n_fft=2048,
       hop_length=512
   )
   S_db = librosa.amplitude_to_db(
       np.abs(D),
       ref=np.max
   )
   times = librosa.times_like(
       S_db,
       sr=sr,
       hop_length=512
   )
   frequencies = librosa.fft_frequencies(
       sr=sr,
       n_fft=2048
   )
   plt.figure(
       figsize=(18, 8)
   )
   plt.pcolormesh(
       times,
       frequencies,
       S_db,
       shading="auto"
   )
   plt.colorbar(
       label="dB"
   )
   # Keep the same frequency range as
   # the screenshot you showed.
   plt.ylim(
       0,
       20000
   )
   plt.xlabel(
       "Time (seconds)"
   )
   plt.ylabel(
       "Frequency (Hz)"
   )
   plt.title(
       f"Detected Gear Events - {filename}"
   )
   # -----------------------------------------------------
   # SELECTED EVENTS
   # -----------------------------------------------------
   if sequence:
       for i, event_time in enumerate(
           sequence,
           start=1
       ):
           plt.axvline(
               event_time,
               linewidth=2
           )
           plt.text(
               event_time,
               19000,
               str(i),
               rotation=90,
               verticalalignment="top",
               horizontalalignment="right",
               fontsize=12
           )
   # -----------------------------------------------------
   # ALL CANDIDATES
   # -----------------------------------------------------
   for candidate in candidates:
       plt.axvline(
           candidate,
           linewidth=0.5,
           alpha=0.35
       )
   plt.tight_layout()
   output = (
       RESULT_DIR /
       filename.replace(
           ".wav",
           "_detected.png"
       )
   )
   plt.savefig(
       output,
       dpi=150
   )
   plt.close()
   print(
       f"Saved: {output}"
   )

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
   for directory, filename in FILES:
       plot_file(
           directory,
           filename
       )
   print()
   print("=" * 70)
   print("DONE")
   print("=" * 70)
   print(
       f"Plots saved in:\n{RESULT_DIR}"
   )

if __name__ == "__main__":
   main()
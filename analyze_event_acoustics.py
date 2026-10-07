from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =========================================================
# PATHS
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
DAY2_DIR = ROOT / "data" / "raw" / "day_2test"
FILE_NAME = "UMIK2299_32.wav"
OUTPUT_DIR = (
   ROOT /
   "results" /
   "event_acoustic_analysis"
)
OUTPUT_DIR.mkdir(
   parents=True,
   exist_ok=True
)

# =========================================================
# AUDIO PARAMETERS
# =========================================================
SR = 16000
HOP = 512

# =========================================================
# LOCAL VALLEY PARAMETERS
# =========================================================
# Search around each candidate for the actual valley.
SEARCH_BEFORE = 0.80
SEARCH_AFTER = 0.80
# Windows used to calculate baseline sound levels.
BASELINE_WINDOW = 0.50
# Avoid measuring immediately next to the valley.
BASELINE_GAP = 0.15

# =========================================================
# DETECT CANDIDATES
# =========================================================
def detect_candidates(y, sr):
   rms = librosa.feature.rms(
       y=y,
       frame_length=2048,
       hop_length=HOP
   )[0]
   rms_db = librosa.amplitude_to_db(
       rms,
       ref=np.max
   )
   rms_times = librosa.times_like(
       rms_db,
       sr=sr,
       hop_length=HOP
   )
   # Normalize
   rms_norm = (
       rms_db - rms_db.min()
   ) / (
       rms_db.max()
       - rms_db.min()
       + 1e-12
   )
   # Gear change tends to produce a sudden decrease.
   drop = -np.gradient(rms_norm)
   # Smooth candidate signal.
   kernel = np.ones(9) / 9.0
   drop_smooth = np.convolve(
       drop,
       kernel,
       mode="same"
   )
   distance = int(
       3.0 * sr / HOP
   )
   peaks, _ = find_peaks(
       drop_smooth,
       distance=distance,
       prominence=0.015
   )
   candidate_times = librosa.frames_to_time(
       peaks,
       sr=sr,
       hop_length=HOP
   )
   duration = len(y) / sr
   valid = (
       (candidate_times >= 3.0)
&
       (candidate_times <= duration - 2.0)
   )
   candidate_times = (
       candidate_times[valid]
   )
   return (
       candidate_times,
       rms_db,
       rms_times,
       drop_smooth
   )

# =========================================================
# HELPER: RMS FROM dB ARRAY
# =========================================================
def median_region(
   rms_db,
   rms_times,
   start_time,
   end_time
):
   mask = (
       (rms_times >= start_time)
&
       (rms_times <= end_time)
   )
   values = rms_db[mask]
   if len(values) == 0:
       return None
   return float(
       np.median(values)
   )

# =========================================================
# FIND ACTUAL LOCAL VALLEY
# =========================================================
def find_local_valley(
   rms_db,
   rms_times,
   candidate_time
):
   search_start = (
       candidate_time
       - SEARCH_BEFORE
   )
   search_end = (
       candidate_time
       + SEARCH_AFTER
   )
   search_mask = (
       (rms_times >= search_start)
&
       (rms_times <= search_end)
   )
   indices = np.where(
       search_mask
   )[0]
   if len(indices) == 0:
       return None
   # Find the lowest RMS point in the local window.
   local_index = indices[
       np.argmin(
           rms_db[indices]
       )
   ]
   return {
       "index": int(local_index),
       "time": float(
           rms_times[local_index]
       ),
       "value": float(
           rms_db[local_index]
       )
   }

# =========================================================
# ANALYZE ONE CANDIDATE
# =========================================================
def analyze_event(
   rms_db,
   rms_times,
   candidate_time
):
   valley = find_local_valley(
       rms_db,
       rms_times,
       candidate_time
   )
   if valley is None:
       return None
   valley_time = valley["time"]
   valley_db = valley["value"]
   # -----------------------------------------------------
   # BEFORE BASELINE
   # -----------------------------------------------------
   before_end = (
       valley_time
       - BASELINE_GAP
   )
   before_start = (
       before_end
       - BASELINE_WINDOW
   )
   before_db = median_region(
       rms_db,
       rms_times,
       before_start,
       before_end
   )
   # -----------------------------------------------------
   # AFTER BASELINE
   # -----------------------------------------------------
   after_start = (
       valley_time
       + BASELINE_GAP
   )
   after_end = (
       after_start
       + BASELINE_WINDOW
   )
   after_db = median_region(
       rms_db,
       rms_times,
       after_start,
       after_end
   )
   if (
       before_db is None
       or after_db is None
   ):
       return None
   # -----------------------------------------------------
   # ACOUSTIC MEASUREMENTS
   # -----------------------------------------------------
   drop_db = (
       before_db
       - valley_db
   )
   recovery_db = (
       after_db
       - valley_db
   )
   # A genuine transition should have:
   #
   # BEFORE  = relatively high
   # VALLEY  = lower
   # AFTER   = rises again
   #
   # Therefore both drop and recovery should
   # ideally be positive.
   if (
       drop_db > 0
       and recovery_db > 0
   ):
       event_score = min(
           drop_db,
           recovery_db
       )
   else:
       event_score = min(
           drop_db,
           recovery_db
       )
   return {
       "candidate_time": float(
           candidate_time
       ),
       "valley_time": float(
           valley_time
       ),
       "valley_offset": float(
           valley_time
           - candidate_time
       ),
       "before_db": float(
           before_db
       ),
       "valley_db": float(
           valley_db
       ),
       "after_db": float(
           after_db
       ),
       "drop_db": float(
           drop_db
       ),
       "recovery_db": float(
           recovery_db
       ),
       "event_score": float(
           event_score
       )
   }

# =========================================================
# MAIN
# =========================================================
def main():
   path = (
       DAY2_DIR /
       FILE_NAME
   )
   if not path.exists():
       print()
       print(
           "FILE NOT FOUND:"
       )
       print(path)
       return
   print()
   print("=" * 100)
   print(
       f"LOCAL VALLEY ANALYSIS: {FILE_NAME}"
   )
   print("=" * 100)
   # -----------------------------------------------------
   # LOAD
   # -----------------------------------------------------
   y, sr = librosa.load(
       path,
       sr=SR,
       mono=True
   )
   duration = (
       len(y) / sr
   )
   print(
       f"Duration: {duration:.2f} seconds"
   )
   # -----------------------------------------------------
   # CANDIDATES
   # -----------------------------------------------------
   (
       candidates,
       rms_db,
       rms_times,
       drop_smooth
   ) = detect_candidates(
       y,
       sr
   )
   print(
       f"Candidates: {len(candidates)}"
   )
   print()
   # -----------------------------------------------------
   # ANALYZE CANDIDATES
   # -----------------------------------------------------
   analyses = []
   for candidate in candidates:
       result = analyze_event(
           rms_db,
           rms_times,
           float(candidate)
       )
       if result is not None:
           analyses.append(
               result
           )
   # -----------------------------------------------------
   # PRINT RESULTS
   # -----------------------------------------------------
   print("-" * 120)
   print(
       f"{'#':>3} "
       f"{'Candidate':>11} "
       f"{'Valley':>11} "
       f"{'Offset':>9} "
       f"{'Before':>10} "
       f"{'Valley':>10} "
       f"{'After':>10} "
       f"{'Drop':>9} "
       f"{'Recovery':>10} "
       f"{'Score':>9}"
   )
   print("-" * 120)
   for number, a in enumerate(
       analyses,
       start=1
   ):
       print(
           f"{number:3d} "
           f"{a['candidate_time']:11.2f} "
           f"{a['valley_time']:11.2f} "
           f"{a['valley_offset']:9.2f} "
           f"{a['before_db']:10.2f} "
           f"{a['valley_db']:10.2f} "
           f"{a['after_db']:10.2f} "
           f"{a['drop_db']:9.2f} "
           f"{a['recovery_db']:10.2f} "
           f"{a['event_score']:9.2f}"
       )
   print("-" * 120)
   # =====================================================
   # PLOT 1 — FULL RMS
   # =====================================================
   plt.figure(
       figsize=(18, 7)
   )
   plt.plot(
       rms_times,
       rms_db
   )
   for number, a in enumerate(
       analyses,
       start=1
   ):
       # Candidate = dashed-looking thin line
       plt.axvline(
           a["candidate_time"],
           linewidth=0.7,
           alpha=0.35
       )
       # Actual valley = stronger line
       plt.axvline(
           a["valley_time"],
           linewidth=1.8
       )
       plt.text(
           a["valley_time"],
           np.max(rms_db),
           str(number),
           rotation=90,
           verticalalignment="top"
       )
   plt.xlabel(
       "Time (seconds)"
   )
   plt.ylabel(
       "RMS (dB)"
   )
   plt.title(
       "UMIK2299_32 - Local Valley Detection"
   )
   plt.grid(
       alpha=0.25
   )
   plt.tight_layout()
   output1 = (
       OUTPUT_DIR /
       "UMIK2299_32_local_valleys.png"
   )
   plt.savefig(
       output1,
       dpi=150
   )
   plt.close()
   print()
   print(
       f"Saved: {output1}"
   )
   # =====================================================
   # PLOT 2 — 65 TO 80 SECONDS
   # =====================================================
   start_time = 65.0
   end_time = 80.0
   mask = (
       (rms_times >= start_time)
&
       (rms_times <= end_time)
   )
   plt.figure(
       figsize=(18, 7)
   )
   plt.plot(
       rms_times[mask],
       rms_db[mask]
   )
   for number, a in enumerate(
       analyses,
       start=1
   ):
       if (
           start_time
<= a["valley_time"]
<= end_time
       ):
           plt.axvline(
               a["candidate_time"],
               linewidth=0.7,
               alpha=0.35
           )
           plt.axvline(
               a["valley_time"],
               linewidth=2
           )
           plt.text(
               a["valley_time"],
               np.max(
                   rms_db[mask]
               ),
               f"Event {number}",
               rotation=90,
               verticalalignment="top"
           )
   plt.xlabel(
       "Time (seconds)"
   )
   plt.ylabel(
       "RMS (dB)"
   )
   plt.title(
       "UMIK2299_32 - Local Valley Analysis 65–80 Seconds"
   )
   plt.grid(
       alpha=0.25
   )
   plt.tight_layout()
   output2 = (
       OUTPUT_DIR /
       "UMIK2299_32_local_valley_zoom.png"
   )
   plt.savefig(
       output2,
       dpi=150
   )
   plt.close()
   print(
       f"Saved: {output2}"
   )
   # =====================================================
   # PLOT 3 — SPECTROGRAM
   # =====================================================
   zoom_start = int(
       start_time * sr
   )
   zoom_end = int(
       end_time * sr
   )
   y_zoom = y[
       zoom_start:zoom_end
   ]
   D = librosa.stft(
       y_zoom,
       n_fft=2048,
       hop_length=HOP
   )
   S_db = librosa.amplitude_to_db(
       np.abs(D),
       ref=np.max
   )
   spec_times = (
       librosa.times_like(
           S_db,
           sr=sr,
           hop_length=HOP
       )
       + start_time
   )
   frequencies = (
       librosa.fft_frequencies(
           sr=sr,
           n_fft=2048
       )
   )
   plt.figure(
       figsize=(18, 8)
   )
   plt.pcolormesh(
       spec_times,
       frequencies,
       S_db,
       shading="auto"
   )
   plt.colorbar(
       label="dB"
   )
   plt.ylim(
       0,
       8000
   )
   for number, a in enumerate(
       analyses,
       start=1
   ):
       if (
           start_time
<= a["valley_time"]
<= end_time
       ):
           plt.axvline(
               a["valley_time"],
               linewidth=2
           )
           plt.text(
               a["valley_time"],
               7600,
               f"Event {number}",
               rotation=90,
               verticalalignment="top"
           )
   plt.xlabel(
       "Time (seconds)"
   )
   plt.ylabel(
       "Frequency (Hz)"
   )
   plt.title(
       "UMIK2299_32 - Spectrogram 65–80 Seconds"
   )
   plt.tight_layout()
   output3 = (
       OUTPUT_DIR /
       "UMIK2299_32_local_valley_spectrogram.png"
   )
   plt.savefig(
       output3,
       dpi=150
   )
   plt.close()
   print(
       f"Saved: {output3}"
   )
   # =====================================================
   # FINISH
   # =====================================================
   print()
   print("=" * 100)
   print(
       "LOCAL VALLEY ANALYSIS COMPLETE"
   )
   print("=" * 100)

if __name__ == "__main__":
   main()
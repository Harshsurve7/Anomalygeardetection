from pathlib import Path
from itertools import combinations
import sys
import json
import numpy as np
import librosa
import torch
import torch.nn as nn

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SRC_DIR))
import sequence_optimizer as so

# ============================================================
# SETTINGS
# ============================================================
SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 220
PRE_EVENT = 1.5
POST_EVENT = 2.0
MODEL_PATH = (
   PROJECT_ROOT
   / "models"
   / "gear_autoencoder.pt"
)
OUTPUT_DIR = (
   PROJECT_ROOT
   / "results"
   / "metrics"
)
OUTPUT_DIR.mkdir(
   parents=True,
   exist_ok=True
)

# ============================================================
# GEAR SEQUENCE
# ============================================================
GEARS = [
   "N_R",
   "R_1",
   "1_2",
   "2_3",
   "3_4",
   "4_5",
   "5_6",
]

# ============================================================
# V5.1 TIMING
# ============================================================
EXPECTED_INTERVALS = np.array(
   [15.0, 15.0, 14.0, 16.0, 17.0, 13.0],
   dtype=float,
)
INTERVAL_SCALE = np.array(
   [6.0, 6.0, 6.0, 6.0, 6.0, 5.0],
   dtype=float,
)

# ============================================================
# AUTOENCODER
# ============================================================
class GearAutoencoder(nn.Module):
   def __init__(self):
       super().__init__()
       self.encoder = nn.Sequential(
           nn.Conv2d(
               1,
               16,
               kernel_size=3,
               stride=2,
               padding=1,
           ),
           nn.ReLU(),
           nn.Conv2d(
               16,
               32,
               kernel_size=3,
               stride=2,
               padding=1,
           ),
           nn.ReLU(),
           nn.Conv2d(
               32,
               64,
               kernel_size=3,
               stride=2,
               padding=1,
           ),
           nn.ReLU(),
       )
       self.decoder = nn.Sequential(
           nn.ConvTranspose2d(
               64,
               32,
               kernel_size=3,
               stride=2,
               padding=1,
               output_padding=1,
           ),
           nn.ReLU(),
           nn.ConvTranspose2d(
               32,
               16,
               kernel_size=3,
               stride=2,
               padding=1,
               output_padding=1,
           ),
           nn.ReLU(),
           nn.ConvTranspose2d(
               16,
               1,
               kernel_size=3,
               stride=2,
               padding=1,
               output_padding=1,
           ),
           nn.Sigmoid(),
       )
   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x

# ============================================================
# LOAD MODEL
# ============================================================
if not MODEL_PATH.exists():
   raise FileNotFoundError(
       f"Model not found:\n{MODEL_PATH}"
   )

DEVICE = torch.device(
   "cuda"
   if torch.cuda.is_available()
   else "cpu"
)

MODEL = GearAutoencoder().to(DEVICE)

CHECKPOINT = torch.load(
   MODEL_PATH,
   map_location=DEVICE,
)

if isinstance(CHECKPOINT, dict):
   if "model_state_dict" in CHECKPOINT:
       STATE_DICT = CHECKPOINT[
           "model_state_dict"
       ]
   elif "state_dict" in CHECKPOINT:
       STATE_DICT = CHECKPOINT[
           "state_dict"
       ]
   else:
       STATE_DICT = CHECKPOINT
else:
   STATE_DICT = CHECKPOINT

MODEL.load_state_dict(STATE_DICT)
MODEL.eval()

# ============================================================
# FIND ONLY KNOWN-OK RECORDINGS
# ============================================================
def find_ok_recordings():
   folders = [
       PROJECT_ROOT
       / "data"
       / "raw"
       / "day_2test",
       PROJECT_ROOT
       / "data"
       / "raw"
       / "day_3test",
   ]
   files = []
   for folder in folders:
       if folder.exists():
           files.extend(
               folder.rglob("*.wav")
           )
   unique = []
   seen = set()
   for file in files:
       key = str(
           file.resolve()
       )
       if key not in seen:
           seen.add(key)
           unique.append(file)
   return sorted(
       unique,
       key=lambda x: str(x).lower()
   )

# ============================================================
# ACOUSTIC SCORE
# ============================================================
def acoustic_score(candidate):
   strength = float(
       candidate["strength"]
   )
   shape = float(
       candidate["shape_score"]
   )
   balance = float(
       candidate["balance"]
   )
   strength_score = 1.0 / (
       1.0
       + np.exp(
           -(strength - 4.0) / 5.0
       )
   )
   score = (
       0.55 * strength_score
       + 0.30 * shape
       + 0.15 * balance
   )
   return float(
       np.clip(
           score,
           0.0,
           1.0
       )
   )

# ============================================================
# TIMING SCORE
# ============================================================
def timing_score(intervals):
   intervals = np.asarray(
       intervals,
       dtype=float
   )
   if len(intervals) != 6:
       return 0.0
   values = np.exp(
       -0.5
       * (
           (
               intervals
               - EXPECTED_INTERVALS
           )
           / INTERVAL_SCALE
       )
       ** 2
   )
   score = float(
       np.mean(values)
   )
   penalty = 1.0
   for interval in intervals:
       if interval < 5.0:
           penalty *= 0.55
       elif interval < 7.0:
           penalty *= 0.80
       elif interval < 8.0:
           penalty *= 0.92
   return score * penalty

# ============================================================
# FIND 7 EVENTS
# ============================================================
def find_best_sequence(audio):
   candidates = so.detect_candidates(
       audio
   )
   candidates = so.prepare_candidates(
       candidates
   )
   if len(candidates) < 7:
       return None
   best = None
   for sequence in combinations(
       candidates,
       7
   ):
       times = np.array(
           [
               item["time"]
               for item in sequence
           ],
           dtype=float
       )
       intervals = np.diff(times)
       if np.any(
           intervals < 3.0
       ):
           continue
       if np.any(
           intervals > 26.0
       ):
           continue
       acoustic_values = [
           acoustic_score(item)
           for item in sequence
       ]
       acoustic = float(
           np.mean(
               acoustic_values
           )
       )
       timing = timing_score(
           intervals
       )
       position = 1.0
       if times[0] < 6.0:
           position = 0.80
       score = (
           0.50 * acoustic
           + 0.40 * timing
           + 0.10 * position
       )
       if (
           best is None
           or score > best["score"]
       ):
           best = {
               "score": float(score),
               "times": times,
           }
   return best

# ============================================================
# MEL
# ============================================================
def audio_to_mel(audio):
   mel = librosa.feature.melspectrogram(
       y=audio,
       sr=SR,
       n_fft=N_FFT,
       hop_length=HOP_LENGTH,
       n_mels=N_MELS,
       fmin=50,
       fmax=8000,
   )
   mel_db = librosa.power_to_db(
       mel,
       ref=np.max
   )
   if (
       mel_db.shape[1]
< TARGET_FRAMES
   ):
       padding = (
           TARGET_FRAMES
           - mel_db.shape[1]
       )
       mel_db = np.pad(
           mel_db,
           (
               (0, 0),
               (0, padding)
           ),
           mode="constant",
           constant_values=-80,
       )
   else:
       mel_db = mel_db[
           :,
           :TARGET_FRAMES
       ]
   mel_db = (
       mel_db + 80.0
   ) / 80.0
   return mel_db.astype(
       np.float32
   )

# ============================================================
# RECONSTRUCTION ERROR
# ============================================================
def reconstruction_error(audio):
   mel = audio_to_mel(
       audio
   )
   tensor = torch.from_numpy(
       mel
   )
   tensor = tensor.unsqueeze(
       0
   )
   tensor = tensor.unsqueeze(
       0
   )
   tensor = tensor.to(
       DEVICE
   )
   with torch.no_grad():
       reconstruction = MODEL(
           tensor
       )
   height = min(
       tensor.shape[2],
       reconstruction.shape[2]
   )
   width = min(
       tensor.shape[3],
       reconstruction.shape[3]
   )
   tensor = tensor[
       :,
       :,
       :height,
       :width
   ]
   reconstruction = reconstruction[
       :,
       :,
       :height,
       :width
   ]
   error = torch.mean(
       (
           tensor
           - reconstruction
       ) ** 2
   )
   return float(
       error.item()
   )

# ============================================================
# MAIN
# ============================================================
def main():
   print("=" * 70)
   print(
       "NORMAL DATA "
       "ANOMALY THRESHOLD CALIBRATION"
   )
   print("=" * 70)
   recordings = find_ok_recordings()
   print(
       f"OK recordings: "
       f"{len(recordings)}"
   )
   if not recordings:
       print(
           "ERROR: No OK recordings found."
       )
       return

   # --------------------------------------------------------
   # Store errors by gear
   # --------------------------------------------------------
   errors_by_gear = {
       gear: []
       for gear in GEARS
   }

   sequence_failures = 0

   # --------------------------------------------------------
   # Process recordings
   # --------------------------------------------------------
   for index, path in enumerate(
       recordings,
       start=1
   ):
       try:
           audio, _ = so.load_audio(
               path
           )
           sequence = find_best_sequence(
               audio
           )
           if sequence is None:
               sequence_failures += 1
               print(
                   f"[{index:02d}/"
                   f"{len(recordings)}] "
                   f"{path.name} -> "
                   f"SEQUENCE FAIL"
               )
               continue

           for gear, time in zip(
               GEARS,
               sequence["times"]
           ):
               start = max(
                   0.0,
                   time - PRE_EVENT
               )
               end = min(
                   len(audio) / SR,
                   time + POST_EVENT
               )
               start_sample = int(
                   start * SR
               )
               end_sample = int(
                   end * SR
               )
               clip = audio[
                   start_sample:end_sample
               ]
               error = reconstruction_error(
                   clip
               )
               errors_by_gear[
                   gear
               ].append(
                   error
               )

           print(
               f"[{index:02d}/"
               f"{len(recordings)}] "
               f"{path.name} -> OK"
           )

       except Exception as exc:
           print(
               f"[{index:02d}/"
               f"{len(recordings)}] "
               f"{path.name} -> ERROR: "
               f"{exc}"
           )

   # ========================================================
   # STATISTICS
   # ========================================================
   print("\n" + "=" * 70)
   print(
       "NORMAL RECONSTRUCTION-ERROR "
       "DISTRIBUTIONS"
   )
   print("=" * 70)
   thresholds = {}
   for gear in GEARS:
       values = np.asarray(
           errors_by_gear[gear],
           dtype=float
       )
       if len(values) == 0:
           print(
               f"{gear}: NO DATA"
           )
           continue

       p95 = float(
           np.percentile(
               values,
               95
           )
       )
       p99 = float(
           np.percentile(
               values,
               99
           )
       )
       maximum = float(
           np.max(values)
       )
       median = float(
           np.median(values)
       )
       mean = float(
           np.mean(values)
       )

       # Proposed threshold:
       # use the 99th percentile of known-normal
       # data, with a small safety margin.
       proposed = float(
           p99 * 1.05
       )
       thresholds[gear] = proposed

       print(
           f"\n{gear}"
       )
       print(
           f"  Samples : {len(values)}"
       )
       print(
           f"  Mean    : {mean:.6f}"
       )
       print(
           f"  Median  : {median:.6f}"
       )
       print(
           f"  P95     : {p95:.6f}"
       )
       print(
           f"  P99     : {p99:.6f}"
       )
       print(
           f"  Max     : {maximum:.6f}"
       )
       print(
           f"  Proposed: {proposed:.6f}"
       )

   # ========================================================
   # SAVE THRESHOLDS
   # ========================================================
   output_file = (
       OUTPUT_DIR
       / "calibrated_thresholds.json"
   )
   with open(
       output_file,
       "w",
       encoding="utf-8"
   ) as file:
       json.dump(
           thresholds,
           file,
           indent=4
       )

   print("\n" + "=" * 70)
   print(
       "THRESHOLDS SAVED"
   )
   print(
       output_file
   )
   print(
       f"Sequence failures: "
       f"{sequence_failures}"
   )
   print("=" * 70)

if __name__ == "__main__":
   main()
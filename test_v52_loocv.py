from pathlib import Path
from itertools import combinations
import sys
import numpy as np

# ============================================================
# IMPORT SEQUENCE OPTIMIZER
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))
import sequence_optimizer as so

# ============================================================
# PROJECT ROOT
# ============================================================
def find_project_root():
   here = Path(__file__).resolve()
   for p in [here] + list(here.parents):
       if (p / "data").exists() and (p / "src").exists():
           return p
   raise RuntimeError(
       "Could not find TRANSAXLE_AI project root."
   )

ROOT = find_project_root()

# ============================================================
# CONFIG
# ============================================================
EXPECTED_EVENTS = 7
MIN_INTERVAL = 3.0
MAX_INTERVAL = 26.0

# ============================================================
# FIND RECORDINGS
# ============================================================
def find_recordings():
   folders = [
       ROOT / "data" / "raw" / "day_2test",
       ROOT / "data" / "raw" / "day_3test",
       ROOT / "data" / "day_2test",
       ROOT / "data" / "day_3test",
   ]
   files = []
   for folder in folders:
       if not folder.exists():
           continue
       files.extend(folder.rglob("*.wav"))
   unique = []
   seen = set()
   for p in files:
       key = str(p.resolve())
       if key not in seen:
           seen.add(key)
           unique.append(p)
   return sorted(
       unique,
       key=lambda x: str(x).lower()
   )

# ============================================================
# LOAD CANDIDATES
# ============================================================
def get_candidates(path):
   y, sr = so.load_audio(path)
   candidates = so.detect_candidates(y)
   candidates = so.prepare_candidates(
       candidates
   )
   return candidates

# ============================================================
# RAW ACOUSTIC SCORE
# ============================================================
def candidate_acoustic_score(c):
   strength = float(c["strength"])
   shape = float(c["shape_score"])
   balance = float(c["balance"])
   strength_score = 1.0 / (
       1.0 +
       np.exp(
           -(strength - 4.0) / 5.0
       )
   )
   return float(
       np.clip(
           0.55 * strength_score +
           0.30 * shape +
           0.15 * balance,
           0.0,
           1.0
       )
   )

# ============================================================
# TRAINING-TIME INTERVAL PROFILE
# ============================================================
def build_interval_profile(training_results):
   all_intervals = []
   for result in training_results:
       if result is None:
           continue
       times = result["times"]
       if len(times) != 7:
           continue
       intervals = np.diff(times)
       if len(intervals) == 6:
           all_intervals.append(intervals)
   if not all_intervals:
       return (
           np.array([
               15.0,
               15.0,
               14.0,
               16.0,
               17.0,
               13.0
           ]),
           np.array([
               6.0,
               6.0,
               6.0,
               6.0,
               6.0,
               5.0
           ])
       )
   arr = np.asarray(
       all_intervals,
       dtype=float
   )
   median = np.median(
       arr,
       axis=0
   )
   # Robust spread.
   mad = np.median(
       np.abs(arr - median),
       axis=0
   )
   scale = np.maximum(
       1.4826 * mad,
       4.0
   )
   return median, scale

# ============================================================
# TIMING SCORE
# ============================================================
def timing_score(
   intervals,
   expected,
   scale
):
   intervals = np.asarray(
       intervals,
       dtype=float
   )
   if len(intervals) != 6:
       return 0.0
   score = np.exp(
       -0.5 *
       (
           (intervals - expected)
           / scale
       ) ** 2
   )
   score = float(
       np.mean(score)
   )
   # Short intervals remain allowed.
   penalty = 1.0
   for x in intervals:
       if x < 5.0:
           penalty *= 0.55
       elif x < 7.0:
           penalty *= 0.80
       elif x < 8.0:
           penalty *= 0.92
   return score * penalty

# ============================================================
# SCORE SEQUENCE
# ============================================================
def score_sequence(
   sequence,
   expected,
   scale
):
   times = np.array(
       [c["time"] for c in sequence],
       dtype=float
   )
   intervals = np.diff(times)
   if np.any(
       intervals < MIN_INTERVAL
   ):
       return None
   if np.any(
       intervals > MAX_INTERVAL
   ):
       return None
   acoustic_values = [
       candidate_acoustic_score(c)
       for c in sequence
   ]
   acoustic = float(
       np.mean(acoustic_values)
   )
   timing = timing_score(
       intervals,
       expected,
       scale
   )
   position = 1.0
   if times[0] < 6.0:
       position *= 0.80
   final_score = (
       0.50 * acoustic +
       0.40 * timing +
       0.10 * position
   )
   return {
       "score": float(final_score),
       "acoustic": acoustic,
       "timing": timing,
       "times": times,
       "intervals": intervals,
       "acoustic_values": acoustic_values
   }

# ============================================================
# BEST 7-EVENT SEQUENCE
# ============================================================
def find_best_sequence(
   candidates,
   expected,
   scale
):
   if len(candidates) < 7:
       return None
   best = None
   for combo in combinations(
       candidates,
       7
   ):
       result = score_sequence(
           combo,
           expected,
           scale
       )
       if result is None:
           continue
       if (
           best is None
           or result["score"] > best["score"]
       ):
           best = result
   return best

# ============================================================
# CLASSIFICATION
# ============================================================
def classify(result):
   if result is None:
       return "REJECTED"
   acoustic = result["acoustic"]
   score = result["score"]
   weak = sum(
       x < 0.35
       for x in result["acoustic_values"]
   )
   short = sum(
       x < 8.0
       for x in result["intervals"]
   )
   if (
       score >= 0.68
       and acoustic >= 0.55
       and weak <= 1
       and short <= 2
   ):
       return "RELIABLE"
   if (
       score >= 0.55
       and acoustic >= 0.45
       and weak <= 2
       and short <= 3
   ):
       return "QUESTIONABLE"
   return "REJECTED"

# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("TRANSAXLE 6-GEAR V5.2 LOOCV")
print("=" * 70)
recordings = find_recordings()
print(f"Project root: {ROOT}")
print(f"Total recordings: {len(recordings)}")
if len(recordings) != 33:
   print(
       "\nWARNING:"
       f" Expected 33 recordings but found "
       f"{len(recordings)}."
   )

# ------------------------------------------------------------
# Precompute candidates
# ------------------------------------------------------------
candidate_cache = {}
errors = []
for path in recordings:
   try:
       candidate_cache[str(path)] = \
           get_candidates(path)
   except Exception as e:
       errors.append(
           (path, str(e))
       )

# ------------------------------------------------------------
# First pass:
# obtain V5.1-style sequences
# ------------------------------------------------------------
DEFAULT_EXPECTED = np.array([
   15.0,
   15.0,
   14.0,
   16.0,
   17.0,
   13.0
])
DEFAULT_SCALE = np.array([
   6.0,
   6.0,
   6.0,
   6.0,
   6.0,
   5.0
])
initial_results = {}
for path in recordings:
   candidates = candidate_cache.get(
       str(path)
   )
   if candidates is None:
       continue
   result = find_best_sequence(
       candidates,
       DEFAULT_EXPECTED,
       DEFAULT_SCALE
   )
   initial_results[str(path)] = result

# ============================================================
# LOOCV
# ============================================================
results = []
for i, held_out in enumerate(
   recordings,
   start=1
):
   training_results = []
   for path in recordings:
       if path == held_out:
           continue
       result = initial_results.get(
           str(path)
       )
       if result is not None:
           training_results.append(result)
   expected, scale = build_interval_profile(
       training_results
   )
   candidates = candidate_cache.get(
       str(held_out)
   )
   if candidates is None:
       results.append({
           "path": held_out,
           "status": "ERROR"
       })
       continue
   best = find_best_sequence(
       candidates,
       expected,
       scale
   )
   status = classify(best)
   if best is None:
       results.append({
           "path": held_out,
           "status": status
       })
       continue
   weak = sum(
       x < 0.35
       for x in best["acoustic_values"]
   )
   short = sum(
       x < 8.0
       for x in best["intervals"]
   )
   results.append({
       "path": held_out,
       "status": status,
       "score": best["score"],
       "acoustic": best["acoustic"],
       "timing": best["timing"],
       "weak": weak,
       "short": short
   })

# ============================================================
# SUMMARY
# ============================================================
reliable = [
   r for r in results
   if r["status"] == "RELIABLE"
]
questionable = [
   r for r in results
   if r["status"] == "QUESTIONABLE"
]
rejected = [
   r for r in results
   if r["status"] == "REJECTED"
]
loocv_errors = [
   r for r in results
   if r["status"] == "ERROR"
]

print("\n" + "=" * 70)
print("V5.2 LOOCV SUMMARY")
print("=" * 70)
print(
   f"Total recordings: {len(results)}"
)
print(
   f"RELIABLE:     "
   f"{len(reliable)}/{len(results)}"
)
print(
   f"QUESTIONABLE: "
   f"{len(questionable)}/{len(results)}"
)
print(
   f"REJECTED:     "
   f"{len(rejected)}/{len(results)}"
)
print(
   f"ERRORS:       "
   f"{len(loocv_errors)}/{len(results)}"
)

# ============================================================
# PROBLEMS ONLY
# ============================================================
problems = (
   questionable +
   rejected +
   loocv_errors
)
if problems:
   print("\n" + "=" * 70)
   print("LOOCV PROBLEMS")
   print("=" * 70)
   for r in problems:
       print(
           f"\n{r['path'].name}"
       )
       print(
           f"Status: {r['status']}"
       )
       if "score" in r:
           print(
               f"Score:    {r['score']:.3f}"
           )
           print(
               f"Acoustic: {r['acoustic']:.3f}"
           )
           print(
               f"Timing:   {r['timing']:.3f}"
           )
           print(
               f"Weak:     {r['weak']}"
           )
           print(
               f"Short:    {r['short']}"
           )
else:
   print(
       "\nNo LOOCV problems."
   )

# ============================================================
# FINAL
# ============================================================
print("\n" + "=" * 70)
print("V5.2 LOOCV COMPLETE")
print("=" * 70)
from pathlib import Path
from itertools import combinations
import numpy as np
import sequence_optimizer as so

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DAY2 = ROOT / "data" / "raw" / "day_2test"
DAY3 = ROOT / "data" / "raw" / "day_3test"

# ============================================================
# EXPECTED 6-GEAR SEQUENCE
# ============================================================
TRANSITIONS = [
   "N→R",
   "R→1",
   "1→2",
   "2→3",
   "3→4",
   "4→5",
   "5→6",
]

# ============================================================
# TIMING MODEL
# ============================================================
# Typical interval between consecutive genuine gear changes.
#
# This is deliberately broad. We do NOT want to force
# every recording to have identical timing.
EXPECTED_INTERVALS = np.array([
   15.0,
   15.0,
   14.0,
   16.0,
   17.0,
   13.0,
])
INTERVAL_SCALE = np.array([
   6.0,
   6.0,
   6.0,
   6.0,
   6.0,
   5.0,
])

ABSOLUTE_MIN_INTERVAL = 3.0
MAX_INTERVAL = 26.0

# ============================================================
# ACOUSTIC SCORE
# ============================================================
def raw_acoustic_score(c):
   strength = c["strength"]
   shape = c["shape_score"]
   balance = c["balance"]
   # Strength is the strongest signal.
   strength_score = np.clip(
       (strength + 2.0) / 30.0,
       0.0,
       1.0
   )
   return float(
       0.55 * strength_score
       +
       0.30 * shape
       +
       0.15 * balance
   )

# ============================================================
# TIMING SCORE
# ============================================================
def interval_score(
   interval,
   index
):
   if interval < ABSOLUTE_MIN_INTERVAL:
       return 0.0
   if interval > MAX_INTERVAL:
       return 0.0
   expected = EXPECTED_INTERVALS[index]
   scale = INTERVAL_SCALE[index]
   z = (
       interval - expected
   ) / scale
   score = np.exp(
       -0.5 * z * z
   )
   # Keep short intervals possible.
   #
   # We already know from the diagnostics that a genuine
   # transition can occasionally occur around 5 seconds.
   if interval < 5.0:
       score *= 0.35
   elif interval < 7.0:
       score *= 0.65
   elif interval < 8.0:
       score *= 0.85
   return float(score)

# ============================================================
# POSITION SCORE
# ============================================================
def position_score(
   time,
   duration,
   index
):
   normalized = (
       time /
       max(duration, 1e-6)
   )
   expected = so.POSITION_TEMPLATE[index]
   distance = abs(
       normalized - expected
   )
   # Very soft positional preference.
   #
   # Timing between events is much more important.
   return float(
       np.exp(
           -(
               distance / 0.22
           ) ** 2
       )
   )

# ============================================================
# SCORE ONE SEQUENCE
# ============================================================
def score_sequence(
   sequence,
   duration
):
   times = np.array([
       c["time"]
       for c in sequence
   ])
   intervals = np.diff(
       times
   )
   if len(intervals) != 6:
       return -999.0
   # --------------------------------------------------------
   # Acoustic
   # --------------------------------------------------------
   acoustic_scores = np.array([
       raw_acoustic_score(c)
       for c in sequence
   ])
   acoustic = float(
       np.mean(
           acoustic_scores
       )
   )
   # --------------------------------------------------------
   # Timing
   # --------------------------------------------------------
   timing_scores = np.array([
       interval_score(
           interval,
           i
       )
       for i, interval
       in enumerate(intervals)
   ])
   timing = float(
       np.mean(
           timing_scores
       )
   )
   # --------------------------------------------------------
   # Position
   # --------------------------------------------------------
   position_scores = np.array([
       position_score(
           c["time"],
           duration,
           i
       )
       for i, c
       in enumerate(sequence)
   ])
   position = float(
       np.mean(
           position_scores
       )
   )
   # --------------------------------------------------------
   # Final score
   # --------------------------------------------------------
   #
   # Acoustic evidence:
   #       50%
   #
   # Timing:
   #       40%
   #
   # Position:
   #       10%
   #
   # This intentionally removes the overly strong
   # transition-specific profile influence that caused
   # some obvious sequences to be rejected.
   #
   final_score = (
       0.50 * acoustic
       +
       0.40 * timing
       +
       0.10 * position
   )
   return float(
       final_score
   )

# ============================================================
# FIND BEST 7-EVENT SEQUENCE
# ============================================================
def find_best_v5(
   candidates,
   duration
):
   if len(candidates) < 7:
       return None
   best_sequence = None
   best_score = -999.0
   # --------------------------------------------------------
   # Exhaustively evaluate possible 7-event combinations.
   #
   # Maximum candidate count in our current data is small
   # enough that this is completely practical.
   # --------------------------------------------------------
   for indices in combinations(
       range(len(candidates)),
       7
   ):
       sequence = [
           candidates[i]
           for i in indices
       ]
       times = [
           c["time"]
           for c in sequence
       ]
       intervals = np.diff(
           times
       )
       # ----------------------------------------------------
       # Hard physical constraints
       # ----------------------------------------------------
       if np.any(
           intervals
<
           ABSOLUTE_MIN_INTERVAL
       ):
           continue
       if np.any(
           intervals
>
           MAX_INTERVAL
       ):
           continue
       score = score_sequence(
           sequence,
           duration
       )
       if score > best_score:
           best_score = score
           best_sequence = sequence
   if best_sequence is None:
       return None
   return {
       "sequence": best_sequence,
       "score": best_score,
   }

# ============================================================
# PROCESS ONE RECORDING
# ============================================================
def process_file(path):
   y, sr = so.load_audio(
       path
   )
   duration = len(y) / sr
   candidates = so.detect_candidates(
       y
   )
   if len(candidates) < 7:
       return {
           "file": path.name,
           "status": "REJECTED",
           "candidates": len(candidates),
       }
   result = find_best_v5(
       candidates,
       duration
   )
   if result is None:
       return {
           "file": path.name,
           "status": "REJECTED",
           "candidates": len(candidates),
       }
   sequence = result[
       "sequence"
   ]
   times = [
       c["time"]
       for c in sequence
   ]
   intervals = np.diff(
       times
   )
   strengths = [
       c["strength"]
       for c in sequence
   ]
   acoustic = [
       raw_acoustic_score(c)
       for c in sequence
   ]
   min_acoustic = min(
       acoustic
   )
   # --------------------------------------------------------
   # Classification
   # --------------------------------------------------------
   weak = sum(
       x < 0.30
       for x in acoustic
   )
   very_weak = sum(
       x < 0.20
       for x in acoustic
   )
   short_intervals = sum(
       x < 8.0
       for x in intervals
   )
   if very_weak >= 2:
       status = "QUESTIONABLE"
   elif weak >= 3:
       status = "QUESTIONABLE"
   elif short_intervals >= 2:
       status = "QUESTIONABLE"
   else:
       status = "RELIABLE"
   return {
       "file": path.name,
       "status": status,
       "candidates": len(candidates),
       "score": result["score"],
       "times": times,
       "intervals": intervals.tolist(),
       "strengths": strengths,
       "acoustic": acoustic,
       "min_acoustic": min_acoustic,
       "weak": weak,
       "very_weak": very_weak,
       "short_intervals": short_intervals,
   }

# ============================================================
# MAIN
# ============================================================
def main():
   print()
   print("=" * 90)
   print(
       "DAY 2 + DAY 3 "
       "6-GEAR SEQUENCE OPTIMIZER V5"
   )
   print("=" * 90)
   files = sorted(
       list(DAY2.glob("*.wav"))
       +
       list(DAY3.glob("*.wav"))
   )
   print(
       f"Total recordings: {len(files)}"
   )
   results = []
   for path in files:
       try:
           result = process_file(
               path
           )
           results.append(
               result
           )
       except Exception as e:
           result = {
               "file": path.name,
               "status": "ERROR",
               "error": str(e),
           }
           results.append(
               result
           )
   # ========================================================
   # SUMMARY
   # ========================================================
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
   errors = [
       r for r in results
       if r["status"] == "ERROR"
   ]
   print()
   print("=" * 90)
   print("V5 SUMMARY")
   print("=" * 90)
   print(
       f"Total recordings: "
       f"{len(results)}"
   )
   print(
       f"RELIABLE: "
       f"{len(reliable)}/{len(results)}"
   )
   print(
       f"QUESTIONABLE: "
       f"{len(questionable)}/{len(results)}"
   )
   print(
       f"REJECTED: "
       f"{len(rejected)}/{len(results)}"
   )
   print(
       f"ERRORS: "
       f"{len(errors)}/{len(results)}"
   )
   print()
   print(
       "REJECTED / QUESTIONABLE"
   )
   print("-" * 90)
   for r in (
       rejected +
       questionable
   ):
       print(
           f"{r['file']:<32} "
           f"{r['status']:<12} "
           f"score={r.get('score', 0):.3f} "
           f"minA={r.get('min_acoustic', 0):.3f} "
           f"weak={r.get('weak', 0)} "
           f"short={r.get('short_intervals', 0)}"
       )
   print()
   print(
       "BEST SEQUENCES FOR PROBLEM RECORDINGS"
   )
   print("-" * 90)
   problem_files = (
       rejected +
       questionable
   )
   for r in problem_files:
       if "times" not in r:
           continue
       print()
       print(
           r["file"]
       )
       print(
           "Times:",
           ", ".join(
               f"{x:.2f}"
               for x in r["times"]
           )
       )
       print(
           "Intervals:",
           ", ".join(
               f"{x:.2f}"
               for x in r["intervals"]
           )
       )
       print(
           "Strength:",
           ", ".join(
               f"{x:.2f}"
               for x in r["strengths"]
           )
       )
       print(
           "Acoustic:",
           ", ".join(
               f"{x:.3f}"
               for x in r["acoustic"]
           )
       )
   print()
   print("=" * 90)
   print("V5 COMPLETE")
   print("=" * 90)

if __name__ == "__main__":
   main()
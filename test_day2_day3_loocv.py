from pathlib import Path
import json
import numpy as np
import sequence_optimizer as so

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DAY2 = ROOT / "data" / "raw" / "day_2test"
DAY3 = ROOT / "data" / "raw" / "day_3test"
TRANSITIONS = so.TRANSITIONS

# ============================================================
# CONFIG
# ============================================================
# We use the existing optimizer to obtain candidate events
# for the training recordings.
#
# The held-out recording is NEVER included in the profile
# that is applied to it.
PROFILE_FIELDS = [
   "drop",
   "recovery",
   "strength",
   "balance",
]

# ============================================================
# ROBUST STATISTICS
# ============================================================
def robust_stats(values):
   values = np.asarray(
       values,
       dtype=float
   )
   values = values[
       np.isfinite(values)
   ]
   if len(values) == 0:
       return {
           "median": 0.0,
           "mad": 1.0,
           "p10": 0.0,
           "p90": 0.0,
       }
   median = float(
       np.median(values)
   )
   mad = float(
       np.median(
           np.abs(
               values - median
           )
       )
   )
   # Prevent zero scale.
   mad = max(
       mad,
       1e-3
   )
   return {
       "median": median,
       "mad": mad,
       "p10": float(
           np.percentile(
               values,
               10
           )
       ),
       "p90": float(
           np.percentile(
               values,
               90
           )
       ),
   }

# ============================================================
# SIMPLE ACOUSTIC-ONLY TRAINING SEQUENCE
# ============================================================
def acoustic_seed_sequence(
   candidates,
   duration
):
   """
   Creates a preliminary 7-event sequence without using
   the transition profile.
   This is used ONLY to build the training profile.
   The held-out recording is not used here.
   """
   if len(candidates) < 7:
       return None
   candidates = sorted(
       candidates,
       key=lambda x: x["time"]
   )
   # --------------------------------------------------------
   # Give every candidate a profile-independent score.
   # --------------------------------------------------------
   scored = []
   for c in candidates:
       strength_score = np.clip(
           (
               c["strength"] + 5.0
           ) / 35.0,
           0,
           1
       )
       shape_score = c[
           "shape_score"
       ]
       balance_score = c[
           "balance"
       ]
       score = (
           0.55 * strength_score
           +
           0.30 * shape_score
           +
           0.15 * balance_score
       )
       scored.append(
           (
               score,
               c
           )
       )
   # --------------------------------------------------------
   # Dynamic programming.
   #
   # Same basic physical constraints as the real optimizer,
   # but NO transition profile.
   # --------------------------------------------------------
   n = len(candidates)
   dp = np.full(
       (7, n),
       -np.inf
   )
   parent = np.full(
       (7, n),
       -1,
       dtype=int
   )
   for i, (_, c) in enumerate(
       scored
   ):
       position = (
           c["time"] /
           max(duration, 1e-6)
       )
       expected = so.POSITION_TEMPLATE[0]
       position_score = np.exp(
           -(
               abs(
                   position -
                   expected
               )
               /
               0.22
           ) ** 2
       )
       dp[0, i] = (
           0.85 * scored[i][0]
           +
           0.15 * position_score
       )
   for k in range(
       1,
       7
   ):
       expected_position = (
           so.POSITION_TEMPLATE[k]
       )
       for i in range(n):
           current = candidates[i]
           position = (
               current["time"] /
               max(duration, 1e-6)
           )
           position_score = np.exp(
               -(
                   abs(
                       position -
                       expected_position
                   )
                   /
                   0.22
               ) ** 2
           )
           current_score = (
               0.70 * scored[i][0]
               +
               0.15 * position_score
           )
           for j in range(i):
               if not np.isfinite(
                   dp[k - 1, j]
               ):
                   continue
               previous = candidates[j]
               interval = (
                   current["time"]
                   -
                   previous["time"]
               )
               if (
                   interval
<
                   so.ABSOLUTE_MIN_INTERVAL
               ):
                   continue
               if (
                   interval
>
                   so.MAX_INTERVAL
               ):
                   continue
               # Soft timing preference.
               timing = so.timing_score(
                   interval
               )
               total = (
                   dp[k - 1, j]
                   +
                   current_score
                   +
                   0.15 * timing
               )
               if total > dp[k, i]:
                   dp[k, i] = total
                   parent[k, i] = j
   if not np.any(
       np.isfinite(
           dp[6]
       )
   ):
       return None
   current = int(
       np.nanargmax(
           dp[6]
       )
   )
   indices = []
   for k in range(
       6,
       -1,
       -1
   ):
       indices.append(
           current
       )
       if k > 0:
           current = parent[
               k,
               current
           ]
           if current < 0:
               return None
   indices.reverse()
   return [
       candidates[i]
       for i in indices
   ]

# ============================================================
# BUILD PROFILE FROM TRAINING RECORDINGS
# ============================================================
def build_training_profile(
   training_files
):
   values = {
       transition: {
           field: []
           for field in PROFILE_FIELDS
       }
       for transition in TRANSITIONS
   }
   used = 0
   failed = 0
   for path in training_files:
       try:
           y, sr = so.load_audio(
               path
           )
           duration = (
               len(y)
               /
               sr
           )
           candidates = (
               so.detect_candidates(
                   y
               )
           )
           if len(candidates) < 7:
               failed += 1
               continue
           sequence = (
               acoustic_seed_sequence(
                   candidates,
                   duration
               )
           )
           if sequence is None:
               failed += 1
               continue
           if len(sequence) != 7:
               failed += 1
               continue
           used += 1
           for k, candidate in enumerate(
               sequence
           ):
               transition = (
                   TRANSITIONS[k]
               )
               for field in PROFILE_FIELDS:
                   values[
                       transition
                   ][field].append(
                       candidate[field]
                   )
       except Exception:
           failed += 1
   # --------------------------------------------------------
   # Convert raw values to profile statistics.
   # --------------------------------------------------------
   profile = {}
   for transition in TRANSITIONS:
       profile[
           transition
       ] = {}
       for field in PROFILE_FIELDS:
           profile[
               transition
           ][field] = robust_stats(
               values[
                   transition
               ][field]
           )
   return (
       profile,
       used,
       failed
   )

# ============================================================
# TEST ONE HELD-OUT RECORDING
# ============================================================
def test_held_out(
   path,
   profile
):
   # IMPORTANT:
   #
   # sequence_optimizer.calculate_acoustic_score()
   # reads the module-level PROFILE.
   #
   # Temporarily replace it with the profile created from
   # the OTHER recordings.
   old_profile = so.PROFILE
   so.PROFILE = profile
   try:
       y, sr = so.load_audio(
           path
       )
       duration = (
           len(y)
           /
           sr
       )
       candidates = (
           so.detect_candidates(
               y
           )
       )
       if len(candidates) < 7:
           return {
               "status": "REJECTED",
               "events": len(candidates),
               "score": 0.0,
               "weak": 0,
               "min_acoustic": 0.0,
               "min_quality": 0.0,
           }
       candidates = (
           so.prepare_candidates(
               candidates
           )
       )
       result = (
           so.find_best_sequence(
               candidates,
               duration
           )
       )
       if result is None:
           return {
               "status": "REJECTED",
               "events": 0,
               "score": 0.0,
               "weak": 0,
               "min_acoustic": 0.0,
               "min_quality": 0.0,
           }
       validation = (
           so.validate_sequence(
               result
           )
       )
       selected = result[
           "selected"
       ]
       acoustics = []
       qualities = []
       for k, candidate in enumerate(
           selected
       ):
           transition = (
               TRANSITIONS[k]
           )
           acoustic = candidate[
               "acoustic_by_transition"
           ][transition]
           quality = so.event_quality(
               candidate,
               acoustic
           )
           acoustics.append(
               acoustic
           )
           qualities.append(
               quality
           )
       return {
           "status": validation[
               "status"
           ],
           "events": len(selected),
           "score": float(
               result["score"]
           ),
           "weak": int(
               result["weak_events"]
           ),
           "min_acoustic": float(
               min(acoustics)
           ),
           "min_quality": float(
               min(qualities)
           ),
       }
   finally:
       # ALWAYS restore original profile.
       so.PROFILE = old_profile

# ============================================================
# MAIN
# ============================================================
def main():
   print()
   print("=" * 80)
   print(
       "DAY 2 + DAY 3 "
       "V4 LEAVE-ONE-RECORDING-OUT TEST"
   )
   print("=" * 80)
   files = sorted(
       list(
           DAY2.glob("*.wav")
       )
       +
       list(
           DAY3.glob("*.wav")
       )
   )
   print(
       f"Total recordings: {len(files)}"
   )
   if len(files) == 0:
       print(
           "ERROR: No WAV files found."
       )
       return
   results = []
   # ========================================================
   # LOOCV
   # ========================================================
   for i, held_out in enumerate(
       files,
       1
   ):
       training_files = [
           p
           for p in files
           if p != held_out
       ]
       # ----------------------------------------------------
       # Build profile from OTHER recordings only.
       # ----------------------------------------------------
       profile, used, failed = (
           build_training_profile(
               training_files
           )
       )
       # ----------------------------------------------------
       # Test held-out recording.
       # ----------------------------------------------------
       result = test_held_out(
           held_out,
           profile
       )
       results.append(
           {
               "file": held_out.name,
               **result,
           }
       )
       print(
           f"[{i:02d}/{len(files)}] "
           f"{held_out.name:<32} "
           f"{result['status']:<12} "
           f"events={result['events']} "
           f"score={result['score']:.3f} "
           f"minA={result['min_acoustic']:.3f} "
           f"minQ={result['min_quality']:.3f}"
       )
   # ========================================================
   # SUMMARY
   # ========================================================
   total = len(results)
   reliable = sum(
       r["status"] == "RELIABLE"
       for r in results
   )
   questionable = sum(
       r["status"] == "QUESTIONABLE"
       for r in results
   )
   rejected = sum(
       r["status"] == "REJECTED"
       for r in results
   )
   errors = sum(
       r["status"] == "ERROR"
       for r in results
   )
   paths = sum(
       r["events"] == 7
       for r in results
   )
   print()
   print("=" * 80)
   print("V4 LOOCV SUMMARY")
   print("=" * 80)
   print(
       f"Total recordings:       {total}"
   )
   print(
       f"7-event paths found:    "
       f"{paths}/{total} "
       f"({100 * paths / total:.1f}%)"
   )
   print(
       f"RELIABLE:               "
       f"{reliable}/{total} "
       f"({100 * reliable / total:.1f}%)"
   )
   print(
       f"QUESTIONABLE:           "
       f"{questionable}/{total} "
       f"({100 * questionable / total:.1f}%)"
   )
   print(
       f"REJECTED:               "
       f"{rejected}/{total} "
       f"({100 * rejected / total:.1f}%)"
   )
   print(
       f"ERRORS:                 "
       f"{errors}/{total}"
   )
   # ========================================================
   # WORST RECORDINGS
   # ========================================================
   print()
   print("LOWEST QUALITY HELD-OUT RECORDINGS")
   print("-" * 80)
   for r in sorted(
       results,
       key=lambda x: (
           x["min_quality"]
       )
   )[:10]:
       print(
           f"{r['file']:<32} "
           f"{r['status']:<12} "
           f"minQ={r['min_quality']:.3f} "
           f"minA={r['min_acoustic']:.3f} "
           f"weak={r['weak']} "
           f"score={r['score']:.3f}"
       )
   # ========================================================
   # SAVE
   # ========================================================
   output_dir = (
       ROOT
       / "results"
       / "metrics"
   )
   output_dir.mkdir(
       parents=True,
       exist_ok=True
   )
   output_path = (
       output_dir
       /
       "day2_day3_v4_loocv_summary.json"
   )
   with open(
       output_path,
       "w",
       encoding="utf-8"
   ) as f:
       json.dump(
           {
               "total": total,
               "paths_found": paths,
               "reliable": reliable,
               "questionable": questionable,
               "rejected": rejected,
               "errors": errors,
               "results": results,
           },
           f,
           indent=2
       )
   print()
   print(
       "Saved:"
   )
   print(
       output_path
   )
   print()
   print("=" * 80)
   print("V4 COMPLETE")
   print("=" * 80)

if __name__ == "__main__":
   main()
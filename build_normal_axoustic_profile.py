from pathlib import Path
import json
import numpy as np
from sequence_optimizer import (
   load_audio,
   optimize_events,
   EXPECTED_EVENTS,
   GEAR_SEQUENCE,
)

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DAY2_DIR = ROOT / "data" / "raw" / "day_2test"
DAY3_DIR = ROOT / "data" / "raw" / "day_3test"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(
   parents=True,
   exist_ok=True
)
OUTPUT_FILE = (
   RESULTS_DIR
   / "normal_acoustic_profile.json"
)

# ============================================================
# SETTINGS
# ============================================================
EXPECTED = EXPECTED_EVENTS
# Robust percentile limits.
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90

# ============================================================
# COLLECT FILES
# ============================================================
files = []
if DAY2_DIR.exists():
   files.extend(
       sorted(
           DAY2_DIR.glob("*.wav")
       )
   )
if DAY3_DIR.exists():
   files.extend(
       sorted(
           DAY3_DIR.glob("*.wav")
       )
   )

print()
print("=" * 80)
print("BUILDING NORMAL ACOUSTIC PROFILE")
print("=" * 80)
print(
   f"Total recordings: {len(files)}"
)
print()

# ============================================================
# STORAGE
#
# One list for every gear transition.
# ============================================================
measurements = {}
for transition in GEAR_SEQUENCE:
   measurements[transition] = {
       "drop_db": [],
       "recovery_db": [],
       "strength": [],
       "balance": [],
       "acoustic_confidence": [],
       "time": [],
   }

# ============================================================
# PROCESS RECORDINGS
# ============================================================
successful = 0
failed = 0

for index, path in enumerate(
   files,
   start=1
):
   print(
       f"[{index}/{len(files)}] "
       f"{path.name}"
   )
   try:
       y, sr = load_audio(
           path
       )
       result = optimize_events(
           y,
           sr,
           EXPECTED
       )
       selected = result[
           "selected"
       ]
       if selected is None:
           print(
               "   FAILED: no 7-event sequence"
           )
           failed += 1
           continue
       if len(selected) != EXPECTED:
           print(
               "   FAILED: incorrect event count"
           )
           failed += 1
           continue
       successful += 1
       # ----------------------------------------------------
       # Store each event according to its known sequence
       # position.
       # ----------------------------------------------------
       for i, event in enumerate(
           selected
       ):
           transition = (
               GEAR_SEQUENCE[i]
           )
           measurements[
               transition
           ]["drop_db"].append(
               float(
                   event["drop_db"]
               )
           )
           measurements[
               transition
           ]["recovery_db"].append(
               float(
                   event["recovery_db"]
               )
           )
           measurements[
               transition
           ]["strength"].append(
               float(
                   event["strength"]
               )
           )
           measurements[
               transition
           ]["balance"].append(
               float(
                   event["balance"]
               )
           )
           measurements[
               transition
           ]["acoustic_confidence"].append(
               float(
                   event[
                       "acoustic_confidence"
                   ]
               )
           )
           measurements[
               transition
           ]["time"].append(
               float(
                   event["time"]
               )
           )
       print(
           "   OK"
       )
   except Exception as e:
       print(
           f"   ERROR: {e}"
       )
       failed += 1

# ============================================================
# ROBUST STATISTICS
# ============================================================
def robust_stats(values):
   values = np.asarray(
       values,
       dtype=float
   )
   if len(values) == 0:
       return {
           "count": 0,
           "median": None,
           "mean": None,
           "std": None,
           "mad": None,
           "p10": None,
           "p25": None,
           "p75": None,
           "p90": None,
           "min": None,
           "max": None,
       }
   median = float(
       np.median(values)
   )
   mad = float(
       np.median(
           np.abs(
               values
               - median
           )
       )
   )
   return {
       "count": int(
           len(values)
       ),
       "median": median,
       "mean": float(
           np.mean(values)
       ),
       "std": float(
           np.std(values)
       ),
       "mad": mad,
       "p10": float(
           np.percentile(
               values,
               10
           )
       ),
       "p25": float(
           np.percentile(
               values,
               25
           )
       ),
       "p75": float(
           np.percentile(
               values,
               75
           )
       ),
       "p90": float(
           np.percentile(
               values,
               90
           )
       ),
       "min": float(
           np.min(values)
       ),
       "max": float(
           np.max(values)
       ),
   }

# ============================================================
# BUILD PROFILE
# ============================================================
profile = {
   "metadata": {
       "total_recordings": len(files),
       "successful_recordings": successful,
       "failed_recordings": failed,
       "expected_events": EXPECTED,
       "sequence": GEAR_SEQUENCE,
       "source": (
           "Day 2 + Day 3 known "
           "6-forward-gear recordings"
       ),
   },
   "transitions": {}
}

for transition in GEAR_SEQUENCE:
   data = measurements[
       transition
   ]
   profile[
       "transitions"
   ][transition] = {
       "drop_db": robust_stats(
           data["drop_db"]
       ),
       "recovery_db": robust_stats(
           data["recovery_db"]
       ),
       "strength": robust_stats(
           data["strength"]
       ),
       "balance": robust_stats(
           data["balance"]
       ),
       "acoustic_confidence": robust_stats(
           data[
               "acoustic_confidence"
           ]
       ),
       "time": robust_stats(
           data["time"]
       ),
   }

# ============================================================
# SAVE JSON
# ============================================================
with open(
   OUTPUT_FILE,
   "w",
   encoding="utf-8"
) as f:
   json.dump(
       profile,
       f,
       indent=4
   )

# ============================================================
# PRINT PROFILE
# ============================================================
print()
print()
print("=" * 80)
print("NORMAL ACOUSTIC PROFILE")
print("=" * 80)
print(
   f"Recordings used: "
   f"{successful}/{len(files)}"
)
print()
for transition in GEAR_SEQUENCE:
   data = profile[
       "transitions"
   ][transition]
   print()
   print(
       "-" * 80
   )
   print(
       transition
   )
   print(
       "-" * 80
   )
   strength = data[
       "strength"
   ]
   drop = data[
       "drop_db"
   ]
   recovery = data[
       "recovery_db"
   ]
   balance = data[
       "balance"
   ]
   confidence = data[
       "acoustic_confidence"
   ]
   timing = data[
       "time"
   ]
   print(
       f"Drop       "
       f"median={drop['median']:.2f} dB  "
       f"P10={drop['p10']:.2f}  "
       f"P90={drop['p90']:.2f}"
   )
   print(
       f"Recovery   "
       f"median={recovery['median']:.2f} dB  "
       f"P10={recovery['p10']:.2f}  "
       f"P90={recovery['p90']:.2f}"
   )
   print(
       f"Strength   "
       f"median={strength['median']:.2f} dB  "
       f"P10={strength['p10']:.2f}  "
       f"P90={strength['p90']:.2f}"
   )
   print(
       f"Balance    "
       f"median={balance['median']:.3f}  "
       f"P10={balance['p10']:.3f}  "
       f"P90={balance['p90']:.3f}"
   )
   print(
       f"Acoustic   "
       f"median={confidence['median']:.3f}  "
       f"P10={confidence['p10']:.3f}  "
       f"P90={confidence['p90']:.3f}"
   )
   print(
       f"Time       "
       f"median={timing['median']:.2f}s  "
       f"P10={timing['p10']:.2f}s  "
       f"P90={timing['p90']:.2f}s"
   )

# ============================================================
# FINAL
# ============================================================
print()
print()
print("=" * 80)
print("PROFILE COMPLETE")
print("=" * 80)
print(
   "Saved:"
)
print(
   OUTPUT_FILE
)
print()
print(
   "This profile will be used in the next"
)
print(
   "sequence-optimizer version."
)
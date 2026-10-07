from pathlib import Path
import sequence_optimizer as so

ROOT = Path(__file__).resolve().parents[1]
DAY2 = ROOT / "data" / "raw" / "day_2test"
DAY3 = ROOT / "data" / "raw" / "day_3test"
TARGETS = {
   "UMIK2299_32.wav",
   "UMIK2299_34.wav",
   "UMIK2299_3.wav",
   "UMIK2299_4.wav",
   "UMIK2299_36.wav",
   "UMIK2299_35.wav",
}

def inspect_file(path):
   print()
   print("=" * 100)
   print(path.name)
   print("=" * 100)
   y, sr = so.load_audio(path)
   duration = len(y) / sr
   candidates = so.detect_candidates(y)
   print(f"Duration: {duration:.2f}s")
   print(f"Candidates: {len(candidates)}")
   if len(candidates) == 0:
       print("NO CANDIDATES")
       return
   candidates = so.prepare_candidates(candidates)
   print()
   print(
       f"{'#':<4}"
       f"{'Time':>9}"
       f"{'Drop':>9}"
       f"{'Recovery':>11}"
       f"{'Strength':>11}"
       f"{'Shape':>9}"
   )
   print("-" * 60)
   for i, c in enumerate(
       candidates,
       1
   ):
       print(
           f"{i:<4}"
           f"{c['time']:>9.2f}"
           f"{c['drop']:>9.2f}"
           f"{c['recovery']:>11.2f}"
           f"{c['strength']:>11.2f}"
           f"{c['shape_score']:>9.3f}"
       )
   # --------------------------------------------------------
   # Try the normal 7-event optimizer.
   # --------------------------------------------------------
   result = so.find_best_sequence(
       candidates,
       duration
   )
   print()
   if result is None:
       print(
           "BEST 7-EVENT PATH: NONE"
       )
       print(
           "This means the failure occurs "
           "before final sequence validation."
       )
       return
   selected = result[
       "selected"
   ]
   print(
       f"BEST 7-EVENT PATH: "
       f"{len(selected)} events"
   )
   print(
       f"Sequence score: "
       f"{result['score']:.4f}"
   )
   print(
       f"Weak events: "
       f"{result['weak_events']}"
   )
   print()
   print(
       f"{'Transition':<12}"
       f"{'Time':>9}"
       f"{'Drop':>9}"
       f"{'Recovery':>11}"
       f"{'Strength':>11}"
       f"{'Acoustic':>11}"
       f"{'Quality':>10}"
   )
   print("-" * 75)
   for k, c in enumerate(
       selected
   ):
       transition = (
           so.TRANSITIONS[k]
       )
       acoustic = c[
           "acoustic_by_transition"
       ][transition]
       quality = so.event_quality(
           c,
           acoustic
       )
       print(
           f"{transition:<12}"
           f"{c['time']:>9.2f}"
           f"{c['drop']:>9.2f}"
           f"{c['recovery']:>11.2f}"
           f"{c['strength']:>11.2f}"
           f"{acoustic:>11.3f}"
           f"{quality:>10.3f}"
       )
   print()
   print(
       "Intervals:"
   )
   times = [
       c["time"]
       for c in selected
   ]
   intervals = [
       times[i] - times[i - 1]
       for i in range(
           1,
           len(times)
       )
   ]
   print(
       "  "
       +
       ", ".join(
           f"{x:.2f}s"
           for x in intervals
       )
   )

def main():
   files = sorted(
       list(DAY2.glob("*.wav"))
       +
       list(DAY3.glob("*.wav"))
   )
   targets = [
       p
       for p in files
       if p.name in TARGETS
   ]
   print("=" * 100)
   print("6-GEAR FAILURE DIAGNOSTIC")
   print("=" * 100)
   print(
       f"Target recordings: {len(targets)}"
   )
   for path in targets:
       try:
           inspect_file(path)
       except Exception as e:
           print()
           print(
               f"ERROR in {path.name}:"
           )
           print(e)

if __name__ == "__main__":
   main()
from pathlib import Path
from sequence_optimizer import process_file

ROOT = Path(__file__).resolve().parents[1]
DAY2 = ROOT / "data" / "raw" / "day_2test"
DAY3 = ROOT / "data" / "raw" / "day_3test"

def main():
   print()
   print("=" * 100)
   print("DAY 2 + DAY 3 ACOUSTIC SEQUENCE OPTIMIZER V2 TEST")
   print("=" * 100)
   print(f"Day 2 folder: {DAY2}")
   print(f"Day 3 folder: {DAY3}")
   files = sorted(
       list(DAY2.glob("*.wav")) +
       list(DAY3.glob("*.wav"))
   )
   print(f"Total files: {len(files)}")
   results = []
   for i, path in enumerate(files, 1):
       print()
       print("=" * 100)
       print(f"[{i}/{len(files)}] {path.name}")
       print("=" * 100)
       result = process_file(path)
       results.append(result)
   # ========================================================
   # SUMMARY
   # ========================================================
   reliable = sum(
       r.get("status") == "RELIABLE"
       for r in results
   )
   questionable = sum(
       r.get("status") == "QUESTIONABLE"
       for r in results
   )
   rejected = sum(
       r.get("status") == "REJECTED"
       for r in results
   )
   errors = sum(
       r.get("status") == "ERROR"
       for r in results
   )
   print()
   print("=" * 100)
   print("FINAL SUMMARY")
   print("=" * 100)
   print(f"Total recordings: {len(results)}")
   print(f"RELIABLE: {reliable}")
   print(f"QUESTIONABLE: {questionable}")
   print(f"REJECTED: {rejected}")
   print(f"ERRORS: {errors}")
   if len(results) > 0:
       usable = reliable + questionable
       print()
       print(
           f"7-event paths found: "
           f"{usable}/{len(results)} "
           f"({100 * usable / len(results):.1f}%)"
       )
       print(
           f"Reliable sequences: "
           f"{reliable}/{len(results)} "
           f"({100 * reliable / len(results):.1f}%)"
       )

if __name__ == "__main__":
   main()
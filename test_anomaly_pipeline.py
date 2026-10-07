from pathlib import Path
from itertools import combinations
import sys

import numpy as np
import librosa
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SRC_DIR))

import sequence_optimizer as so

SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 220
PRE_EVENT = 1.5
POST_EVENT = 2.0
MODEL_PATH = PROJECT_ROOT / "models" / "gear_autoencoder.pt"

GEARS = ["N_R", "R_1", "1_2", "2_3", "3_4", "4_5", "5_6"]

# ============================================================
# CALIBRATED GEAR-SPECIFIC THRESHOLDS
# ============================================================
THRESHOLD_PATH = (
   PROJECT_ROOT
   / "results"
   / "metrics"
   / "calibrated_thresholds.json"
)
if not THRESHOLD_PATH.exists():
   raise FileNotFoundError(
       f"Calibrated threshold file not found:\n"
       f"{THRESHOLD_PATH}\n\n"
       f"Run calibrate_anomaly_thresholds.py first."
   )
# ============================================================
# CALIBRATED GEAR-SPECIFIC THRESHOLDS
# ============================================================
import json
THRESHOLD_PATH = (
   PROJECT_ROOT
   / "results"
   / "metrics"
   / "calibrated_thresholds.json"
)
if not THRESHOLD_PATH.exists():
   raise FileNotFoundError(
       f"Calibrated threshold file not found:\n"
       f"{THRESHOLD_PATH}\n\n"
       f"Run calibrate_anomaly_thresholds.py first."
   )
with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
   threshold_data = json.load(f)

def extract_threshold(data, gear):
   """
   Supports the possible JSON structures produced by
   the calibration script.
   """
   # Structure:
   # {"thresholds": {"N→R": 0.002812, ...}}
   if "thresholds" in data:
       value = data["thresholds"][gear]
       if isinstance(value, dict):
           return float(value["threshold"])
       return float(value)
   # Structure:
   # {"N→R": {"threshold": 0.002812}, ...}
   if gear in data:
       value = data[gear]
       if isinstance(value, dict):
           if "threshold" in value:
               return float(value["threshold"])
           if "proposed" in value:
               return float(value["proposed"])
       return float(value)
   # Try ASCII gear names such as N_R, R_1, etc.
   gear_key = (
       gear
       .replace("→", "_")
   )
   if gear_key in data:
       value = data[gear_key]
       if isinstance(value, dict):
           if "threshold" in value:
               return float(value["threshold"])
           if "proposed" in value:
               return float(value["proposed"])
       return float(value)
   raise KeyError(
       f"Could not find threshold for {gear} "
       f"in {THRESHOLD_PATH}"
   )

THRESHOLDS = {
   gear: extract_threshold(
       threshold_data,
       gear,
   )
   for gear in GEARS
}

print("\nLoaded calibrated anomaly thresholds:")
for gear in GEARS:
   print(
       f"  {gear}: "
       f"{THRESHOLDS[gear]:.6f}"
   )
EXPECTED_INTERVALS = np.array([15.0, 15.0, 14.0, 16.0, 17.0, 13.0])
INTERVAL_SCALE = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 5.0])


class GearAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Autoencoder model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GearAutoencoder().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()
    return model, device


# ============================================================
# FIND NOT-OK WAV FILES
# ============================================================
def find_recordings():
   not_ok_folder = (
       PROJECT_ROOT
       / "data"
       / "raw"
       / "not ok"
   )
   if not not_ok_folder.exists():
       raise FileNotFoundError(
           f"NOT-OK folder not found:\n"
           f"{not_ok_folder}"
       )
   files = list(
       not_ok_folder.rglob("*.wav")
   )
   return sorted(
       files,
       key=lambda x: str(x).lower(),
   )


def acoustic_score(candidate):
    strength = float(candidate["strength"])
    shape = float(candidate["shape_score"])
    balance = float(candidate["balance"])
    strength_score = 1.0 / (1.0 + np.exp(-(strength - 4.0) / 5.0))
    return float(np.clip(
        0.55 * strength_score + 0.30 * shape + 0.15 * balance,
        0.0,
        1.0,
    ))


def timing_score(intervals):
    intervals = np.asarray(intervals, dtype=float)
    if len(intervals) != 6:
        return 0.0

    values = np.exp(
        -0.5 * ((intervals - EXPECTED_INTERVALS) / INTERVAL_SCALE) ** 2
    )
    score = float(np.mean(values))
    penalty = 1.0

    for value in intervals:
        if value < 5.0:
            penalty *= 0.55
        elif value < 7.0:
            penalty *= 0.80
        elif value < 8.0:
            penalty *= 0.92

    return score * penalty


def find_best_sequence(audio):
    candidates = so.prepare_candidates(so.detect_candidates(audio))
    if len(candidates) < 7:
        return None

    best = None

    for sequence in combinations(candidates, 7):
        times = np.array([item["time"] for item in sequence], dtype=float)
        intervals = np.diff(times)

        if np.any(intervals < 3.0) or np.any(intervals > 26.0):
            continue

        acoustic = float(np.mean([acoustic_score(item) for item in sequence]))
        timing = timing_score(intervals)
        position = 0.80 if times[0] < 6.0 else 1.0
        score = 0.50 * acoustic + 0.40 * timing + 0.10 * position

        if best is None or score > best["score"]:
            best = {"score": float(score), "times": times}

    return best


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
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=-80,
        )
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]

    return ((mel_db + 80.0) / 80.0).astype(np.float32)


def reconstruction_error(audio, model, device):
    mel = audio_to_mel(audio)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstruction = model(x)

    height = min(x.shape[2], reconstruction.shape[2])
    width = min(x.shape[3], reconstruction.shape[3])
    x = x[:, :, :height, :width]
    reconstruction = reconstruction[:, :, :height, :width]

    return float(torch.mean((x - reconstruction) ** 2).item())


def process_recording(path, model, device):
    audio, _ = so.load_audio(path)
    sequence = find_best_sequence(audio)

    if sequence is None:
        return {"status": "SEQUENCE_FAIL", "events": []}

    events = []
    for gear, time in zip(GEARS, sequence["times"]):
        start = max(0.0, float(time) - PRE_EVENT)
        end = min(len(audio) / SR, float(time) + POST_EVENT)
        clip = audio[int(start * SR):int(end * SR)]
        error = reconstruction_error(clip, model, device)
        threshold = THRESHOLDS[gear]
        status = "ANOMALY" if error > threshold else "NORMAL"
        events.append({
            "gear": gear,
            "time": float(time),
            "error": error,
            "threshold": threshold,
            "status": status,
        })

    recording_status = "ANOMALY" if any(e["status"] == "ANOMALY" for e in events) else "NORMAL"
    return {"status": recording_status, "events": events, "score": sequence["score"]}


def main():
    print("=" * 70)
    print("TRANSAXLE 6-GEAR ANOMALY TEST")
    print("=" * 70)

    model, device = load_model()
    recordings = find_recordings()

    print(f"Project:    {PROJECT_ROOT}")
    print(f"Model:      {MODEL_PATH}")
    print(f"Device:     {device}")
    print(f"Recordings: {len(recordings)}")

    total_events = 0
    normal_events = 0
    anomaly_events = 0
    sequence_failures = 0
    errors = 0
    anomalies = []

    for index, path in enumerate(recordings, start=1):
        try:
            result = process_recording(path, model, device)

            if result["status"] == "SEQUENCE_FAIL":
                sequence_failures += 1
                print(f"[{index:02d}/{len(recordings)}] {path.name} -> SEQUENCE_FAIL")
                continue

            for event in result["events"]:
                total_events += 1
                if event["status"] == "ANOMALY":
                    anomaly_events += 1
                else:
                    normal_events += 1

            if result["status"] == "ANOMALY":
                anomalies.append((path, result))

            print(f"[{index:02d}/{len(recordings)}] {path.name} -> {result['status']}")

        except Exception as exc:
            errors += 1
            print(f"[{index:02d}/{len(recordings)}] {path.name} -> ERROR: {exc}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Recordings:        {len(recordings)}")
    print(f"Sequence failures: {sequence_failures}")
    print(f"Errors:            {errors}")
    print(f"Events evaluated:  {total_events}")
    print(f"Normal events:     {normal_events}")
    print(f"Anomaly events:    {anomaly_events}")

    print("\n" + "=" * 70)
    print("ANOMALY DETAILS")
    print("=" * 70)

    if not anomalies:
        print("No anomalies detected.")
    else:
        for path, result in anomalies:
            print(f"\n{path.name}")
            for event in result["events"]:
                if event["status"] == "ANOMALY":
                    print(
                        f"  {event['gear']}  "
                        f"time={event['time']:.2f}s  "
                        f"error={event['error']:.6f}  "
                        f"threshold={event['threshold']:.6f}"
                    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

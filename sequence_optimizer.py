from pathlib import Path
import json
import math

import numpy as np
import librosa
from scipy.signal import find_peaks


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = (
    ROOT
    / "results"
    / "metrics"
    / "transition_profile.json"
)


# ============================================================
# AUDIO
# ============================================================

SR = 16000

FRAME_LENGTH = 2048
HOP_LENGTH = 512


# ============================================================
# EXPECTED TRANSITION SEQUENCE
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

N_EVENTS = len(TRANSITIONS)


# ============================================================
# TIMING
# ============================================================

TIME_MEDIAN = 13.70
TIME_SCALE = 6.58

# Absolute physical minimum.
ABSOLUTE_MIN_INTERVAL = 3.0

# Intervals below this are suspicious.
SOFT_MIN_INTERVAL = 8.0

MAX_INTERVAL = 26.0


# ============================================================
# EVENT DETECTION
# ============================================================

MIN_PEAK_SPACING = 2.5

MIN_PROMINENCE = 0.012

# Minimum amount of drop/recovery needed to make a
# candidate worth considering.
MIN_DROP_DB = 2.0
MIN_RECOVERY_DB = 1.0

# Window sizes in seconds.
DROP_LOOKBACK = 1.5
TROUGH_WINDOW = 0.45
RECOVERY_START = 0.35
RECOVERY_END = 2.0


# ============================================================
# CANDIDATE MERGING
# ============================================================

# If two candidate events are extremely close,
# they are probably the same physical transient.
MERGE_DISTANCE = 2.5


# ============================================================
# TRANSITION-SPECIFIC STRENGTH
# ============================================================

MIN_STRENGTH = {
    "N→R": -5.0,
    "R→1": 2.0,
    "1→2": 1.0,
    "2→3": 0.5,
    "3→4": -1.5,
    "4→5": -1.5,
    "5→6": -2.5,
}


# ============================================================
# WEAK EVENT SETTINGS
# ============================================================

WEAK_ACOUSTIC_THRESHOLD = 0.35

MAX_WEAK_EVENTS = 2


# ============================================================
# POSITION TEMPLATE
# ============================================================

POSITION_TEMPLATE = np.array([
    0.10,
    0.23,
    0.36,
    0.49,
    0.63,
    0.78,
    0.90,
])


# ============================================================
# PROFILE
# ============================================================

def load_profile():

    if not PROFILE_PATH.exists():

        print()
        print(
            "WARNING: transition profile not found:"
        )
        print(PROFILE_PATH)
        print(
            "Running without profile."
        )
        print()

        return {}

    try:

        with open(
            PROFILE_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            profile = json.load(f)

        print(
            f"Loaded transition profile: "
            f"{PROFILE_PATH}"
        )

        return profile

    except Exception as e:

        print(
            "WARNING: could not load profile:"
        )

        print(e)

        return {}


PROFILE = load_profile()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_signal(x):

    x = np.asarray(
        x,
        dtype=float
    )

    if len(x) == 0:
        return x

    lo = np.percentile(
        x,
        5
    )

    hi = np.percentile(
        x,
        95
    )

    if hi - lo < 1e-9:

        return np.zeros_like(x)

    result = (
        x - lo
    ) / (
        hi - lo
    )

    return np.clip(
        result,
        0,
        1
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    return y, sr


# ============================================================
# RMS
# ============================================================

def calculate_rms(y):

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    times = librosa.frames_to_time(
        np.arange(
            len(rms)
        ),
        sr=SR,
        hop_length=HOP_LENGTH
    )

    return rms, times


# ============================================================
# DB CONVERSION
# ============================================================

def to_db(x):

    return 20.0 * np.log10(
        np.maximum(
            x,
            1e-10
        )
    )


# ============================================================
# CANDIDATE FEATURE EXTRACTION
# ============================================================

def calculate_candidate_features(
    rms,
    times,
    peak_index
):

    frames_per_second = (
        SR /
        HOP_LENGTH
    )

    lookback_frames = max(
        1,
        int(
            DROP_LOOKBACK
            *
            frames_per_second
        )
    )

    trough_frames = max(
        1,
        int(
            TROUGH_WINDOW
            *
            frames_per_second
        )
    )

    recovery_start_frames = max(
        1,
        int(
            RECOVERY_START
            *
            frames_per_second
        )
    )

    recovery_end_frames = max(
        recovery_start_frames + 1,
        int(
            RECOVERY_END
            *
            frames_per_second
        )
    )

    # --------------------------------------------------------
    # BEFORE EVENT
    # --------------------------------------------------------

    before_start = max(
        0,
        peak_index -
        lookback_frames
    )

    before = rms[
        before_start:
        peak_index
    ]

    if len(before) < 3:
        return None

    baseline = np.percentile(
        before,
        75
    )

    # --------------------------------------------------------
    # TROUGHS
    # --------------------------------------------------------

    trough_end = min(
        len(rms),
        peak_index +
        trough_frames
    )

    trough_region = rms[
        peak_index:
        trough_end
    ]

    if len(trough_region) == 0:
        return None

    local_trough_offset = int(
        np.argmin(
            trough_region
        )
    )

    trough_index = (
        peak_index
        +
        local_trough_offset
    )

    trough_level = rms[
        trough_index
    ]

    # --------------------------------------------------------
    # DROP
    # --------------------------------------------------------

    drop_db = (
        to_db(
            baseline
        )
        -
        to_db(
            trough_level
        )
    )

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    recovery_start = min(
        len(rms) - 1,
        trough_index +
        recovery_start_frames
    )

    recovery_end = min(
        len(rms),
        trough_index +
        recovery_end_frames
    )

    if recovery_end <= recovery_start:

        return None

    recovery_region = rms[
        recovery_start:
        recovery_end
    ]

    if len(recovery_region) == 0:
        return None

    recovery_level = np.percentile(
        recovery_region,
        75
    )

    recovery_db = (
        to_db(
            recovery_level
        )
        -
        to_db(
            trough_level
        )
    )

    # --------------------------------------------------------
    # STRENGTH
    # --------------------------------------------------------

    strength = min(
        drop_db,
        recovery_db
    )

    # --------------------------------------------------------
    # BALANCE
    # --------------------------------------------------------

    abs_drop = abs(
        drop_db
    )

    abs_recovery = abs(
        recovery_db
    )

    if max(
        abs_drop,
        abs_recovery
    ) < 1e-9:

        balance = 0.0

    else:

        balance = (
            min(
                abs_drop,
                abs_recovery
            )
            /
            max(
                abs_drop,
                abs_recovery
            )
        )

    # --------------------------------------------------------
    # EVENT SHAPE
    # --------------------------------------------------------

    # We want:
    #
    #   high level
    #       ↓
    #     trough
    #       ↓
    #    recovery
    #
    # A candidate with good drop but no recovery
    # should be penalized.

    recovery_ratio = (
        recovery_db /
        max(
            abs(drop_db),
            1e-6
        )
    )

    recovery_ratio = float(
        np.clip(
            recovery_ratio,
            0,
            1
        )
    )

    # --------------------------------------------------------
    # SHAPE SCORE
    # --------------------------------------------------------

    drop_score = np.clip(
        drop_db / 30.0,
        0,
        1
    )

    recovery_score = np.clip(
        recovery_db / 25.0,
        0,
        1
    )

    shape_score = (
        0.45 * drop_score
        +
        0.40 * recovery_score
        +
        0.15 * balance
    )

    return {
        "index": int(
            peak_index
        ),

        "time": float(
            times[peak_index]
        ),

        "trough_time": float(
            times[trough_index]
        ),

        "drop": float(
            drop_db
        ),

        "recovery": float(
            recovery_db
        ),

        "strength": float(
            strength
        ),

        "balance": float(
            balance
        ),

        "recovery_ratio": float(
            recovery_ratio
        ),

        "shape_score": float(
            np.clip(
                shape_score,
                0,
                1
            )
        ),
    }


# ============================================================
# MERGE CLOSE CANDIDATES
# ============================================================

def merge_close_candidates(
    candidates
):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: x["time"]
    )

    merged = []

    current = candidates[0]

    for candidate in candidates[1:]:

        distance = (
            candidate["time"]
            -
            current["time"]
        )

        if distance < MERGE_DISTANCE:

            # Keep the candidate with the
            # stronger physical event shape.

            current_score = (
                current["shape_score"]
                *
                max(
                    current["strength"],
                    0
                )
            )

            candidate_score = (
                candidate["shape_score"]
                *
                max(
                    candidate["strength"],
                    0
                )
            )

            if candidate_score > current_score:

                current = candidate

        else:

            merged.append(
                current
            )

            current = candidate

    merged.append(
        current
    )

    return merged


# ============================================================
# CANDIDATE DETECTION
# ============================================================

def detect_candidates(y):

    rms, times = calculate_rms(
        y
    )

    log_rms = np.log(
        rms + 1e-10
    )

    # Downward acoustic movement.
    drop_signal = -np.gradient(
        log_rms
    )

    # Smooth only mildly.
    kernel = np.ones(7) / 7.0

    smooth_drop = np.convolve(
        drop_signal,
        kernel,
        mode="same"
    )

    peaks, properties = find_peaks(
        smooth_drop,
        distance=int(
            MIN_PEAK_SPACING
            *
            SR
            /
            HOP_LENGTH
        ),
        prominence=MIN_PROMINENCE
    )

    candidates = []

    duration = len(y) / SR

    for peak_index in peaks:

        time = float(
            times[peak_index]
        )

        if time < 3.0:
            continue

        if time > duration - 2.5:
            continue

        features = (
            calculate_candidate_features(
                rms,
                times,
                peak_index
            )
        )

        if features is None:
            continue

        # ----------------------------------------------------
        # Basic physical plausibility
        # ----------------------------------------------------

        if (
            features["drop"]
            < MIN_DROP_DB
        ):
            continue

        if (
            features["recovery"]
            < MIN_RECOVERY_DB
        ):
            continue

        # ----------------------------------------------------
        # Shape gate
        # ----------------------------------------------------

        if (
            features["shape_score"]
            < 0.12
        ):
            continue

        candidates.append(
            features
        )

    # Remove duplicate detections of
    # the same physical transient.
    candidates = merge_close_candidates(
        candidates
    )

    return candidates


# ============================================================
# PROFILE HELPERS
# ============================================================

def get_transition_profile(
    transition
):

    if not PROFILE:
        return None

    if transition in PROFILE:
        return PROFILE[
            transition
        ]

    if (
        "transitions"
        in PROFILE
    ):

        transitions = PROFILE[
            "transitions"
        ]

        if transition in transitions:
            return transitions[
                transition
            ]

    return None


# ============================================================
# NORMALIZE VALUE
# ============================================================

def normalize_value(
    value,
    low,
    high
):

    if high <= low:
        return 0.0

    return float(
        np.clip(
            (
                value - low
            )
            /
            (
                high - low
            ),
            0,
            1
        )
    )


# ============================================================
# ROBUST PROFILE SCORE
# ============================================================

def robust_normal_score(
    value,
    median,
    scale,
    p10=None,
    p90=None
):

    scale = max(
        abs(scale),
        1e-6
    )

    z = (
        value - median
    ) / scale

    score = math.exp(
        -0.5 * z * z
    )

    if p10 is not None:

        if value < p10:

            distance = (
                p10 - value
            ) / scale

            score *= math.exp(
                -0.30 * distance
            )

    if p90 is not None:

        if value > p90:

            distance = (
                value - p90
            ) / scale

            score *= math.exp(
                -0.30 * distance
            )

    return float(
        np.clip(
            score,
            0,
            1
        )
    )


# ============================================================
# STRENGTH PLAUSIBILITY
# ============================================================

def strength_plausibility(
    transition,
    strength
):

    minimum = MIN_STRENGTH[
        transition
    ]

    if strength >= (
        minimum + 5.0
    ):

        return 1.0

    if strength >= minimum:

        return 0.80

    if strength >= (
        minimum - 3.0
    ):

        return 0.40

    return 0.10


# ============================================================
# ACOUSTIC SCORE
# ============================================================

def calculate_acoustic_score(
    candidate,
    transition
):

    strength = candidate[
        "strength"
    ]

    drop = candidate[
        "drop"
    ]

    recovery = candidate[
        "recovery"
    ]

    balance = candidate[
        "balance"
    ]

    shape = candidate[
        "shape_score"
    ]

    plausibility = (
        strength_plausibility(
            transition,
            strength
        )
    )

    profile = (
        get_transition_profile(
            transition
        )
    )

    # --------------------------------------------------------
    # No profile
    # --------------------------------------------------------

    if profile is None:

        score = (
            0.30
            *
            normalize_value(
                strength,
                -5,
                35
            )
            +
            0.20
            *
            normalize_value(
                drop,
                0,
                40
            )
            +
            0.20
            *
            normalize_value(
                recovery,
                0,
                35
            )
            +
            0.10
            *
            balance
            +
            0.20
            *
            shape
        )

        return float(
            np.clip(
                score
                *
                plausibility,
                0,
                1
            )
        )

    # --------------------------------------------------------
    # Profile extraction
    # --------------------------------------------------------

    def extract(
        key,
        default_median,
        default_scale
    ):

        obj = profile.get(
            key,
            {}
        )

        if not isinstance(
            obj,
            dict
        ):

            return (
                default_median,
                default_scale,
                None,
                None
            )

        median = obj.get(
            "median",
            default_median
        )

        scale = obj.get(
            "mad",
            obj.get(
                "scale",
                default_scale
            )
        )

        p10 = obj.get(
            "p10",
            None
        )

        p90 = obj.get(
            "p90",
            None
        )

        return (
            median,
            scale,
            p10,
            p90
        )

    s_med, s_scale, s_p10, s_p90 = (
        extract(
            "strength",
            8.0,
            5.0
        )
    )

    d_med, d_scale, d_p10, d_p90 = (
        extract(
            "drop",
            15.0,
            8.0
        )
    )

    r_med, r_scale, r_p10, r_p90 = (
        extract(
            "recovery",
            8.0,
            8.0
        )
    )

    b_med, b_scale, b_p10, b_p90 = (
        extract(
            "balance",
            0.6,
            0.25
        )
    )

    strength_score = (
        robust_normal_score(
            strength,
            s_med,
            s_scale,
            s_p10,
            s_p90
        )
    )

    drop_score = (
        robust_normal_score(
            drop,
            d_med,
            d_scale,
            d_p10,
            d_p90
        )
    )

    recovery_score = (
        robust_normal_score(
            recovery,
            r_med,
            r_scale,
            r_p10,
            r_p90
        )
    )

    balance_score = (
        robust_normal_score(
            balance,
            b_med,
            b_scale,
            b_p10,
            b_p90
        )
    )

    # Shape is deliberately included.
    score = (
        0.30 * strength_score
        +
        0.20 * drop_score
        +
        0.15 * recovery_score
        +
        0.10 * balance_score
        +
        0.25 * shape
    )

    return float(
        np.clip(
            score
            *
            plausibility,
            0,
            1
        )
    )


# ============================================================
# TIMING SCORE
# ============================================================

def timing_score(
    interval
):

    if (
        interval
        <
        ABSOLUTE_MIN_INTERVAL
    ):

        return 0.0

    if (
        interval
        >
        MAX_INTERVAL
    ):

        return 0.0

    # --------------------------------------------------------
    # Strong penalty for very short intervals.
    # --------------------------------------------------------

    if interval < 5.0:

        return 0.10

    if interval < 6.0:

        return 0.25

    if interval < 7.0:

        return 0.40

    if interval < 8.0:

        return 0.60

    # --------------------------------------------------------
    # Normal timing model.
    # --------------------------------------------------------

    z = (
        interval
        -
        TIME_MEDIAN
    ) / max(
        TIME_SCALE,
        1e-6
    )

    score = math.exp(
        -0.5 * z * z
    )

    return float(
        np.clip(
            score,
            0,
            1
        )
    )


# ============================================================
# POSITION SCORE
# ============================================================

def position_score(
    time,
    duration,
    transition_index
):

    if duration <= 0:
        return 0.0

    normalized_time = (
        time /
        duration
    )

    expected = (
        POSITION_TEMPLATE[
            transition_index
        ]
    )

    distance = abs(
        normalized_time
        -
        expected
    )

    # Very soft positional preference.
    score = math.exp(
        -(
            distance /
            0.18
        ) ** 2
    )

    return float(
        np.clip(
            score,
            0,
            1
        )
    )


# ============================================================
# EVENT QUALITY
# ============================================================

def event_quality(
    candidate,
    acoustic
):

    strength = candidate[
        "strength"
    ]

    balance = candidate[
        "balance"
    ]

    shape = candidate[
        "shape_score"
    ]

    strength_component = (
        normalize_value(
            strength,
            -5,
            30
        )
    )

    quality = (
        0.55 * acoustic
        +
        0.25 * shape
        +
        0.10 * strength_component
        +
        0.10 * balance
    )

    return float(
        np.clip(
            quality,
            0,
            1
        )
    )


# ============================================================
# PREPARE CANDIDATES
# ============================================================

def prepare_candidates(
    candidates
):

    for candidate in candidates:

        candidate[
            "acoustic_by_transition"
        ] = {}

        for transition in TRANSITIONS:

            candidate[
                "acoustic_by_transition"
            ][transition] = (
                calculate_acoustic_score(
                    candidate,
                    transition
                )
            )

    return candidates


# ============================================================
# SEQUENCE OPTIMIZER
# ============================================================

def find_best_sequence(
    candidates,
    duration
):

    if len(candidates) < N_EVENTS:

        return None

    n = len(candidates)

    dp = np.full(
        (
            N_EVENTS,
            n
        ),
        -np.inf
    )

    parent = np.full(
        (
            N_EVENTS,
            n
        ),
        -1,
        dtype=int
    )

    weak_count = np.full(
        (
            N_EVENTS,
            n
        ),
        999,
        dtype=int
    )

    # ========================================================
    # FIRST EVENT
    # ========================================================

    transition = TRANSITIONS[0]

    for i, candidate in enumerate(
        candidates
    ):

        acoustic = candidate[
            "acoustic_by_transition"
        ][transition]

        quality = event_quality(
            candidate,
            acoustic
        )

        position = position_score(
            candidate["time"],
            duration,
            0
        )

        score = (
            0.75 * quality
            +
            0.25 * position
        )

        dp[0, i] = score

        weak_count[
            0,
            i
        ] = int(
            acoustic
            <
            WEAK_ACOUSTIC_THRESHOLD
        )

    # ========================================================
    # REMAINING EVENTS
    # ========================================================

    for k in range(
        1,
        N_EVENTS
    ):

        transition = (
            TRANSITIONS[k]
        )

        for i, candidate in enumerate(
            candidates
        ):

            acoustic = candidate[
                "acoustic_by_transition"
            ][transition]

            quality = event_quality(
                candidate,
                acoustic
            )

            position = position_score(
                candidate["time"],
                duration,
                k
            )

            for j in range(i):

                previous_score = dp[
                    k - 1,
                    j
                ]

                if not np.isfinite(
                    previous_score
                ):

                    continue

                previous = (
                    candidates[j]
                )

                interval = (
                    candidate["time"]
                    -
                    previous["time"]
                )

                # Absolute impossible interval.
                if (
                    interval
                    <
                    ABSOLUTE_MIN_INTERVAL
                ):

                    continue

                if (
                    interval
                    >
                    MAX_INTERVAL
                ):

                    continue

                tscore = (
                    timing_score(
                        interval
                    )
                )

                is_weak = (
                    acoustic
                    <
                    WEAK_ACOUSTIC_THRESHOLD
                )

                new_weak = (
                    weak_count[
                        k - 1,
                        j
                    ]
                    +
                    int(is_weak)
                )

                if (
                    new_weak
                    >
                    MAX_WEAK_EVENTS
                ):

                    continue

                # ------------------------------------------------
                # LOCAL SCORE
                # ------------------------------------------------

                local_score = (
                    0.55 * quality
                    +
                    0.25 * tscore
                    +
                    0.10 * position
                    +
                    0.10 * acoustic
                )

                # ------------------------------------------------
                # Short-interval penalty
                # ------------------------------------------------

                if interval < 8.0:

                    if interval < 5.0:

                        local_score -= 0.45

                    elif interval < 6.0:

                        local_score -= 0.30

                    elif interval < 7.0:

                        local_score -= 0.20

                    else:

                        local_score -= 0.10

                # ------------------------------------------------
                # Weak-event penalty
                # ------------------------------------------------

                if is_weak:

                    local_score -= 0.18

                total = (
                    previous_score
                    +
                    local_score
                )

                if (
                    total
                    >
                    dp[k, i]
                ):

                    dp[
                        k,
                        i
                    ] = total

                    parent[
                        k,
                        i
                    ] = j

                    weak_count[
                        k,
                        i
                    ] = new_weak

    # ========================================================
    # FINAL STATE
    # ========================================================

    final_scores = dp[
        N_EVENTS - 1
    ]

    if not np.any(
        np.isfinite(
            final_scores
        )
    ):

        return None

    best_index = int(
        np.nanargmax(
            final_scores
        )
    )

    best_score = float(
        final_scores[
            best_index
        ]
    )

    # ========================================================
    # BACKTRACK
    # ========================================================

    selected_indices = []

    current = best_index

    for k in range(
        N_EVENTS - 1,
        -1,
        -1
    ):

        selected_indices.append(
            current
        )

        current = parent[
            k,
            current
        ]

        if (
            k > 0
            and
            current < 0
        ):

            return None

    selected_indices.reverse()

    selected = [
        candidates[i]
        for i in selected_indices
    ]

    return {
        "selected": selected,
        "score": best_score,
        "weak_events": weak_count[
            N_EVENTS - 1,
            best_index
        ],
    }


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_sequence(
    result
):

    if result is None:

        return {
            "status": "REJECTED",
            "reason": (
                "No valid 7-event path"
            )
        }

    selected = result[
        "selected"
    ]

    weak_events = result[
        "weak_events"
    ]

    acoustic_scores = []

    intervals = []

    for k, candidate in enumerate(
        selected
    ):

        transition = (
            TRANSITIONS[k]
        )

        acoustic = candidate[
            "acoustic_by_transition"
        ][transition]

        acoustic_scores.append(
            acoustic
        )

    for i in range(
        1,
        len(selected)
    ):

        intervals.append(
            selected[i]["time"]
            -
            selected[i - 1]["time"]
        )

    acoustic_scores = np.array(
        acoustic_scores
    )

    mean_acoustic = float(
        np.mean(
            acoustic_scores
        )
    )

    min_acoustic = float(
        np.min(
            acoustic_scores
        )
    )

    # --------------------------------------------------------
    # Short intervals
    # --------------------------------------------------------

    very_short = sum(
        x < 5.0
        for x in intervals
    )

    short = sum(
        x < 8.0
        for x in intervals
    )

    # --------------------------------------------------------
    # Reject
    # --------------------------------------------------------

    if very_short >= 2:

        return {
            "status": "REJECTED",
            "reason": (
                "Multiple extremely short "
                "transition intervals"
            )
        }

    if (
        min_acoustic
        <
        0.15
    ):

        return {
            "status": "REJECTED",
            "reason": (
                "At least one event has "
                "extremely weak acoustic evidence"
            )
        }

    if (
        weak_events
        >
        MAX_WEAK_EVENTS
    ):

        return {
            "status": "REJECTED",
            "reason": (
                "Too many weak events"
            )
        }

    # --------------------------------------------------------
    # Questionable
    # --------------------------------------------------------

    if (
        weak_events >= 2
        or
        short >= 2
        or
        mean_acoustic < 0.50
        or
        min_acoustic < 0.25
    ):

        return {
            "status": "QUESTIONABLE",
            "reason": (
                "Sequence exists but "
                "contains weak acoustic "
                "or timing evidence"
            )
        }

    # --------------------------------------------------------
    # Reliable
    # --------------------------------------------------------

    return {
        "status": "RELIABLE",
        "reason": (
            "Seven-event sequence has "
            "sufficient acoustic and "
            "timing support"
        )
    }


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(path):

    print()
    print("=" * 100)
    print(path.name)
    print("=" * 100)

    try:

        y, sr = load_audio(
            path
        )

        duration = (
            len(y)
            /
            sr
        )

        candidates = (
            detect_candidates(
                y
            )
        )

        print(
            f"Duration: "
            f"{duration:.2f}s"
        )

        print(
            f"Candidates detected: "
            f"{len(candidates)}"
        )

        if len(candidates) < N_EVENTS:

            print()
            print(
                "RESULT: REJECTED"
            )

            print(
                "Reason: fewer than "
                "7 valid candidates"
            )

            return {
                "file": path.name,
                "status": "REJECTED",
            }

        candidates = (
            prepare_candidates(
                candidates
            )
        )

        result = (
            find_best_sequence(
                candidates,
                duration
            )
        )

        if result is None:

            print()
            print(
                "RESULT: REJECTED"
            )

            print(
                "Reason: NO VALID "
                "7-EVENT SEQUENCE"
            )

            return {
                "file": path.name,
                "status": "REJECTED",
            }

        validation = (
            validate_sequence(
                result
            )
        )

        selected = result[
            "selected"
        ]

        print(
            f"Selected: "
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
            "Transition        "
            "Time      "
            "Drop    "
            "Recovery    "
            "Strength   "
            "Shape    "
            "Acoustic   "
            "Quality"
        )

        print(
            "-" * 110
        )

        for k, candidate in enumerate(
            selected
        ):

            transition = (
                TRANSITIONS[k]
            )

            acoustic = candidate[
                "acoustic_by_transition"
            ][transition]

            quality = event_quality(
                candidate,
                acoustic
            )

            print(
                f"{transition:<16}"
                f"{candidate['time']:>7.2f}s   "
                f"{candidate['drop']:>7.2f}   "
                f"{candidate['recovery']:>9.2f}   "
                f"{candidate['strength']:>9.2f}   "
                f"{candidate['shape_score']:>7.3f}   "
                f"{acoustic:>8.3f}   "
                f"{quality:>7.3f}"
            )

        # ====================================================
        # INTERVALS
        # ====================================================

        intervals = []

        for i in range(
            1,
            len(selected)
        ):

            intervals.append(
                selected[i]["time"]
                -
                selected[i - 1]["time"]
            )

        print()

        print(
            "Intervals:",
            ", ".join(
                f"{x:.2f}s"
                for x in intervals
            )
        )

        print(
            "Strengths:",
            ", ".join(
                f"{x['strength']:.2f}"
                for x in selected
            )
        )

        print(
            "Acoustic:",
            ", ".join(
                f"{x['acoustic_by_transition'][TRANSITIONS[i]]:.3f}"
                for i, x in enumerate(
                    selected
                )
            )
        )

        print()

        print(
            f"RESULT: "
            f"{validation['status']}"
        )

        print(
            f"Reason: "
            f"{validation['reason']}"
        )

        return {
            "file": path.name,
            "status": validation[
                "status"
            ],
            "score": result[
                "score"
            ],
            "weak_events": result[
                "weak_events"
            ],
            "selected_times": [
                x["time"]
                for x in selected
            ],
            "strengths": [
                x["strength"]
                for x in selected
            ],
            "acoustic": [
                x[
                    "acoustic_by_transition"
                ][TRANSITIONS[i]]
                for i, x in enumerate(
                    selected
                )
            ],
        }

    except Exception as e:

        print()
        print(
            "ERROR:",
            str(e)
        )

        return {
            "file": path.name,
            "status": "ERROR",
            "error": str(e)
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DAY2 = (
        ROOT
        / "data"
        / "raw"
        / "day_2test"
    )

    DAY3 = (
        ROOT
        / "data"
        / "raw"
        / "day_3test"
    )

    print()
    print("=" * 100)
    print(
        "DAY 2 + DAY 3 "
        "ACOUSTIC SEQUENCE OPTIMIZER V3"
    )
    print("=" * 100)

    print(
        f"Day 2 folder: "
        f"{DAY2}"
    )

    print(
        f"Day 3 folder: "
        f"{DAY3}"
    )

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
        f"Total files: "
        f"{len(files)}"
    )

    results = []

    for i, path in enumerate(
        files,
        1
    ):

        print()
        print(
            f"[{i}/{len(files)}] "
            f"{path.name}"
        )

        result = process_file(
            path
        )

        results.append(
            result
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    reliable = sum(
        r.get("status")
        ==
        "RELIABLE"
        for r in results
    )

    questionable = sum(
        r.get("status")
        ==
        "QUESTIONABLE"
        for r in results
    )

    rejected = sum(
        r.get("status")
        ==
        "REJECTED"
        for r in results
    )

    errors = sum(
        r.get("status")
        ==
        "ERROR"
        for r in results
    )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        f"Total recordings: "
        f"{len(results)}"
    )

    print(
        f"RELIABLE: "
        f"{reliable}"
    )

    print(
        f"QUESTIONABLE: "
        f"{questionable}"
    )

    print(
        f"REJECTED: "
        f"{rejected}"
    )

    print(
        f"ERRORS: "
        f"{errors}"
    )

    if len(results) > 0:

        usable = (
            reliable
            +
            questionable
        )

        print()

        print(
            f"7-event paths found: "
            f"{usable}/{len(results)} "
            f"("
            f"{100 * usable / len(results):.1f}"
            f"%)"
        )

        print(
            f"Reliable sequences: "
            f"{reliable}/{len(results)} "
            f"("
            f"{100 * reliable / len(results):.1f}"
            f"%)"
        )

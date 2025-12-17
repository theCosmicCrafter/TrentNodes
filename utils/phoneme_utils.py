"""
Phoneme to viseme mapping utilities for lip sync.

Maps phonemes from speech recognition to mouth shape indices
compatible with standard animation mouth charts (A-H + X).
"""
from typing import Dict, List, Tuple

# Standard mouth shape definitions (Rhubarb-compatible)
# Index 0-7 are active shapes, 8 is idle/silence
MOUTH_SHAPES = {
    0: "A",  # Closed (M, B, P)
    1: "B",  # Slight opening (F, V)
    2: "C",  # Open (TH, DH)
    3: "D",  # Wide (T, D, S, Z, N, L)
    4: "E",  # Round-ish (CH, SH, JH)
    5: "F",  # UU shape (UW, OW, W)
    6: "G",  # OH shape (AA, AO)
    7: "H",  # Wide open (AE, EH, AH)
    8: "X",  # Idle/silence
}

# ARPAbet phoneme to mouth shape index mapping
# Based on standard viseme groupings
ARPABET_TO_VISEME: Dict[str, int] = {
    # Silence and pauses
    "SIL": 8,
    "SP": 8,
    "": 8,

    # Bilabial stops and nasals -> A (closed)
    "M": 0,
    "B": 0,
    "P": 0,

    # Labiodental fricatives -> B (slight)
    "F": 1,
    "V": 1,

    # Dental fricatives -> C (open)
    "TH": 2,
    "DH": 2,

    # Alveolar sounds -> D (wide)
    "T": 3,
    "D": 3,
    "S": 3,
    "Z": 3,
    "N": 3,
    "L": 3,
    "R": 3,

    # Post-alveolar and palatal -> E (round-ish)
    "SH": 4,
    "ZH": 4,
    "CH": 4,
    "JH": 4,
    "Y": 4,

    # Rounded vowels and W -> F (UU)
    "UW": 5,
    "UH": 5,
    "OW": 5,
    "W": 5,

    # Open back vowels -> G (OH)
    "AA": 6,
    "AO": 6,
    "OY": 6,

    # Open front/central vowels -> H (wide open)
    "AE": 7,
    "EH": 7,
    "AH": 7,
    "IH": 7,
    "IY": 7,
    "EY": 7,
    "AW": 7,
    "AY": 7,
    "ER": 7,

    # Velar sounds -> D (similar to alveolar)
    "K": 3,
    "G": 3,
    "NG": 3,

    # Glottal -> A (closed)
    "HH": 0,
}

# IPA phoneme to mouth shape index mapping (for Allosaurus)
IPA_TO_VISEME: Dict[str, int] = {
    # Silence
    "": 8,

    # Bilabial -> A (closed)
    "m": 0,
    "b": 0,
    "p": 0,

    # Labiodental -> B (slight)
    "f": 1,
    "v": 1,

    # Dental -> C (open)
    "th": 2,  # voiceless dental fricative
    "dh": 2,  # voiced dental fricative
    "\u03b8": 2,  # theta
    "\u00f0": 2,  # eth

    # Alveolar -> D (wide)
    "t": 3,
    "d": 3,
    "s": 3,
    "z": 3,
    "n": 3,
    "l": 3,
    "r": 3,
    "\u027e": 3,  # flap

    # Post-alveolar/Palatal -> E (round-ish)
    "\u0283": 4,  # esh (sh)
    "\u0292": 4,  # ezh (zh)
    "t\u0283": 4,  # ch
    "d\u0292": 4,  # jh
    "j": 4,

    # Rounded vowels -> F (UU)
    "u": 5,
    "\u028a": 5,  # upsilon
    "o": 5,
    "w": 5,

    # Open back vowels -> G (OH)
    "\u0251": 6,  # open back unrounded
    "\u0254": 6,  # open-mid back rounded

    # Open front/central -> H (wide open)
    "\u00e6": 7,  # ash
    "\u025b": 7,  # epsilon
    "\u028c": 7,  # caret
    "\u026a": 7,  # small cap I
    "i": 7,
    "e": 7,
    "a": 7,
    "\u0259": 7,  # schwa

    # Velar -> D
    "k": 3,
    "g": 3,
    "\u014b": 3,  # eng

    # Glottal -> A
    "h": 0,
    "\u0294": 0,  # glottal stop
}

# Simplified mapping with fewer distinctions (for lower quality input)
SIMPLIFIED_TO_VISEME: Dict[str, int] = {
    "closed": 0,
    "slight": 1,
    "open": 2,
    "wide": 3,
    "round": 4,
    "uu": 5,
    "oh": 6,
    "ah": 7,
    "silence": 8,
}


def get_viseme_from_phoneme(
    phoneme: str,
    mapping: str = "arpabet"
) -> int:
    """
    Convert a phoneme string to a viseme index.

    Args:
        phoneme: The phoneme string (e.g., "AA", "B", "SH")
        mapping: Which mapping to use ("arpabet", "ipa", "simplified")

    Returns:
        Viseme index (0-8)
    """
    phoneme_upper = phoneme.upper().strip()
    phoneme_lower = phoneme.lower().strip()

    if mapping == "arpabet":
        # Remove stress markers (0, 1, 2)
        phoneme_clean = "".join(
            c for c in phoneme_upper if not c.isdigit()
        )
        return ARPABET_TO_VISEME.get(phoneme_clean, 8)
    elif mapping == "ipa":
        return IPA_TO_VISEME.get(phoneme_lower, 8)
    elif mapping == "simplified":
        return SIMPLIFIED_TO_VISEME.get(phoneme_lower, 8)
    else:
        return 8


def phonemes_to_frame_sequence(
    phoneme_data: List[Dict],
    fps: float,
    audio_duration: float,
    mapping: str = "arpabet"
) -> List[int]:
    """
    Convert phoneme timing data to per-frame mouth shape indices.

    Uses vectorized slice assignment for better performance.

    Args:
        phoneme_data: List of dicts with "start", "end", "phoneme" keys
        fps: Frames per second of the video
        audio_duration: Total duration of audio in seconds
        mapping: Phoneme mapping type to use

    Returns:
        List of viseme indices, one per frame
    """
    total_frames = int(audio_duration * fps)

    # Initialize all frames to idle (X)
    sequence = [8] * total_frames

    for phoneme_info in phoneme_data:
        start_time = phoneme_info.get("start", 0.0)
        end_time = phoneme_info.get("end", start_time + 0.1)
        phoneme = phoneme_info.get("phoneme", "")

        viseme = get_viseme_from_phoneme(phoneme, mapping)

        # Calculate frame range for this phoneme
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames))

        # Vectorized slice assignment (much faster than loop)
        if start_frame < end_frame:
            sequence[start_frame:end_frame] = [viseme] * (end_frame - start_frame)

    return sequence


def smooth_viseme_sequence(
    sequence: List[int],
    hold_frames: int = 2
) -> List[int]:
    """
    Smooth a viseme sequence to avoid rapid flickering.

    Args:
        sequence: Original viseme index sequence
        hold_frames: Minimum frames to hold each viseme

    Returns:
        Smoothed sequence
    """
    if not sequence or hold_frames <= 1:
        return sequence

    smoothed = sequence.copy()
    i = 0
    while i < len(smoothed):
        current = smoothed[i]
        # Find how long this viseme runs
        run_length = 1
        while (i + run_length < len(smoothed) and
               smoothed[i + run_length] == current):
            run_length += 1

        # If run is too short, extend it
        if run_length < hold_frames and i + run_length < len(smoothed):
            for j in range(run_length, min(hold_frames, len(smoothed) - i)):
                smoothed[i + j] = current

        i += run_length

    return smoothed


def get_mouth_shape_name(index: int) -> str:
    """Get the letter name for a mouth shape index."""
    return MOUTH_SHAPES.get(index, "X")


def get_all_mouth_shapes() -> Dict[int, str]:
    """Get all mouth shape definitions."""
    return MOUTH_SHAPES.copy()

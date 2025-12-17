"""
Audio to phonemes node using Vosk speech recognition.

Extracts phoneme-level timing information from audio input
for use in lip sync animation.
"""
import json
import os
import tempfile
import logging
from typing import Dict, Any, List, Tuple, Optional

import torch

from ...utils.audio_utils import (
    preprocess_audio_for_recognition,
    save_audio_to_wav,
    get_audio_duration,
    extract_audio_from_dict,
)

logger = logging.getLogger(__name__)

# Model download URLs and paths
VOSK_MODEL_URLS = {
    "en-small": (
        "https://alphacephei.com/vosk/models/"
        "vosk-model-small-en-us-0.15.zip"
    ),
    "en-large": (
        "https://alphacephei.com/vosk/models/"
        "vosk-model-en-us-0.22.zip"
    ),
}


def get_model_path(model_name: str = "en-small") -> str:
    """Get or download the Vosk model path."""
    import folder_paths

    models_dir = os.path.join(
        folder_paths.models_dir,
        "vosk"
    )
    os.makedirs(models_dir, exist_ok=True)

    model_subdir = f"vosk-model-{model_name}"
    model_path = os.path.join(models_dir, model_subdir)

    if os.path.exists(model_path):
        return model_path

    # Check for alternative naming
    alt_names = [
        "vosk-model-small-en-us-0.15",
        "vosk-model-en-us-0.22",
        "vosk-model-small-en-us",
        "vosk-model-en-us",
    ]
    for alt in alt_names:
        alt_path = os.path.join(models_dir, alt)
        if os.path.exists(alt_path):
            return alt_path

    return model_path


def download_vosk_model(model_name: str = "en-small") -> str:
    """Download Vosk model if not present."""
    import folder_paths
    import zipfile
    import urllib.request

    model_path = get_model_path(model_name)
    if os.path.exists(model_path):
        return model_path

    models_dir = os.path.join(folder_paths.models_dir, "vosk")
    os.makedirs(models_dir, exist_ok=True)

    url = VOSK_MODEL_URLS.get(model_name)
    if not url:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info(f"Downloading Vosk model from {url}...")

    zip_path = os.path.join(models_dir, f"{model_name}.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(models_dir)

        os.remove(zip_path)

        # Find extracted directory
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and "vosk-model" in item:
                return item_path

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(
            f"Failed to download Vosk model. Please manually download "
            f"from {url} and extract to {models_dir}"
        )

    return model_path


def recognize_with_vosk(
    wav_path: str,
    model_path: str,
    sample_rate: int = 16000
) -> List[Dict[str, Any]]:
    """
    Run Vosk recognition and extract phoneme timings.

    Args:
        wav_path: Path to WAV file
        model_path: Path to Vosk model directory
        sample_rate: Audio sample rate

    Returns:
        List of phoneme timing dicts
    """
    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)  # Suppress Vosk logging
    except ImportError:
        raise ImportError(
            "Vosk not installed. Please run: pip install vosk"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Vosk model not found at {model_path}. "
            f"Please download from https://alphacephei.com/vosk/models"
        )

    model = Model(model_path)
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)

    phonemes = []

    with open(wav_path, "rb") as f:
        # Skip WAV header
        f.read(44)

        while True:
            data = f.read(4000)
            if not data:
                break
            rec.AcceptWaveform(data)

    # Get final result
    result = json.loads(rec.FinalResult())

    # Extract word timings and convert to phoneme approximations
    if "result" in result:
        for word_info in result["result"]:
            word = word_info.get("word", "")
            start = word_info.get("start", 0.0)
            end = word_info.get("end", start + 0.1)

            # Approximate phonemes from word
            word_phonemes = _word_to_phonemes(word, start, end)
            phonemes.extend(word_phonemes)

    return phonemes


def _word_to_phonemes(
    word: str,
    start_time: float,
    end_time: float
) -> List[Dict[str, Any]]:
    """
    Approximate phoneme sequence from a word.

    This is a simplified phoneme estimation. For more accurate
    results, use a proper phoneme recognizer or G2P system.
    """
    # Simple letter-to-phoneme mapping for common patterns
    phoneme_map = {
        "a": "AH",
        "e": "EH",
        "i": "IH",
        "o": "AO",
        "u": "UH",
        "b": "B",
        "c": "K",
        "d": "D",
        "f": "F",
        "g": "G",
        "h": "HH",
        "j": "JH",
        "k": "K",
        "l": "L",
        "m": "M",
        "n": "N",
        "p": "P",
        "q": "K",
        "r": "R",
        "s": "S",
        "t": "T",
        "v": "V",
        "w": "W",
        "x": "K",
        "y": "IY",
        "z": "Z",
    }

    # Digraph patterns
    digraphs = {
        "th": "TH",
        "sh": "SH",
        "ch": "CH",
        "ng": "NG",
        "ck": "K",
        "ph": "F",
        "wh": "W",
        "oo": "UW",
        "ee": "IY",
        "ea": "IY",
        "ai": "EY",
        "ay": "EY",
        "ow": "OW",
        "ou": "AW",
        "oi": "OY",
        "oy": "OY",
    }

    word_lower = word.lower()
    phonemes = []

    i = 0
    while i < len(word_lower):
        # Check for digraphs first
        if i < len(word_lower) - 1:
            digraph = word_lower[i:i+2]
            if digraph in digraphs:
                phonemes.append(digraphs[digraph])
                i += 2
                continue

        # Single character
        char = word_lower[i]
        if char in phoneme_map:
            phonemes.append(phoneme_map[char])
        i += 1

    if not phonemes:
        # Return silence for empty/unknown words
        return [{
            "start": start_time,
            "end": end_time,
            "phoneme": "SIL"
        }]

    # Distribute phonemes evenly across word duration
    duration = end_time - start_time
    phoneme_duration = duration / len(phonemes)

    result = []
    for idx, phoneme in enumerate(phonemes):
        result.append({
            "start": start_time + idx * phoneme_duration,
            "end": start_time + (idx + 1) * phoneme_duration,
            "phoneme": phoneme
        })

    return result


class AudioToPhonemes:
    """
    Extract phoneme timing data from audio using Vosk.

    Takes audio input and produces phoneme sequence with
    timing information for lip sync.
    """

    CATEGORY = "Trent/LipSync"
    DESCRIPTION = (
        "Extract phoneme timings from audio using Vosk speech recognition."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio input (waveform + sample_rate)"
                }),
            },
            "optional": {
                "model_size": (["small", "large"], {
                    "default": "small",
                    "tooltip": "Vosk model size (small=50MB, large=1.8GB)"
                }),
                "language": (["en"], {
                    "default": "en",
                    "tooltip": "Recognition language"
                }),
            },
        }

    RETURN_TYPES = ("PHONEME_DATA", "FLOAT")
    RETURN_NAMES = ("phoneme_data", "duration")
    FUNCTION = "extract_phonemes"

    def extract_phonemes(
        self,
        audio: Dict[str, Any],
        model_size: str = "small",
        language: str = "en"
    ) -> Tuple[List[Dict], float]:
        """
        Extract phonemes from audio.

        Args:
            audio: Audio dict with waveform and sample_rate
            model_size: Which Vosk model to use
            language: Recognition language

        Returns:
            Tuple of (phoneme_data_list, audio_duration)
        """
        # Extract audio data
        waveform, sample_rate = extract_audio_from_dict(audio)

        # Get audio duration before preprocessing
        duration = get_audio_duration(waveform, sample_rate)

        # Preprocess for recognition
        processed, target_sr = preprocess_audio_for_recognition(
            waveform, sample_rate, target_sr=16000
        )

        # Save to temp WAV file
        wav_path = save_audio_to_wav(processed, target_sr)

        try:
            # Get model path (download if needed)
            model_name = f"{language}-{model_size}"
            try:
                model_path = get_model_path(model_name)
                if not os.path.exists(model_path):
                    logger.info("Vosk model not found, downloading...")
                    model_path = download_vosk_model(model_name)
            except Exception as e:
                logger.warning(f"Auto-download failed: {e}")
                # Try to find any available model
                import folder_paths
                models_dir = os.path.join(folder_paths.models_dir, "vosk")
                if os.path.exists(models_dir):
                    for item in os.listdir(models_dir):
                        item_path = os.path.join(models_dir, item)
                        if os.path.isdir(item_path):
                            model_path = item_path
                            break
                else:
                    raise

            # Run recognition
            phonemes = recognize_with_vosk(wav_path, model_path, target_sr)

            # Add silence padding at start and end
            if phonemes:
                if phonemes[0]["start"] > 0.05:
                    phonemes.insert(0, {
                        "start": 0.0,
                        "end": phonemes[0]["start"],
                        "phoneme": "SIL"
                    })
                if phonemes[-1]["end"] < duration - 0.05:
                    phonemes.append({
                        "start": phonemes[-1]["end"],
                        "end": duration,
                        "phoneme": "SIL"
                    })
            else:
                # No speech detected
                phonemes = [{
                    "start": 0.0,
                    "end": duration,
                    "phoneme": "SIL"
                }]

            return (phonemes, duration)

        finally:
            # Clean up temp file
            if os.path.exists(wav_path):
                os.remove(wav_path)

"""
Lip sync nodes for non-human character animation.

Provides phoneme detection from audio and frame compositing
with user-provided mouth shape images.
"""
from .audio_to_phonemes import AudioToPhonemes
from .phoneme_to_mouth_shapes import PhonemeToMouthShapes, MouthShapePreview
from .mouth_shape_compositor import MouthShapeCompositor
from .creature_lipsync import CreatureLipSync, MouthShapeLoader
from .tracked_compositor import (
    RemoveMouthBackground,
    MouthShapeCompositorTracked,
)
from .point_tracker import PointTracker, PointsToMasks, PointPreview

NODE_CLASS_MAPPINGS = {
    "AudioToPhonemes": AudioToPhonemes,
    "PhonemeToMouthShapes": PhonemeToMouthShapes,
    "MouthShapePreview": MouthShapePreview,
    "MouthShapeCompositor": MouthShapeCompositor,
    "MouthShapeLoader": MouthShapeLoader,
    "CreatureLipSync": CreatureLipSync,
    "RemoveMouthBackground": RemoveMouthBackground,
    "MouthShapeCompositorTracked": MouthShapeCompositorTracked,
    "PointTracker": PointTracker,
    "PointsToMasks": PointsToMasks,
    "PointPreview": PointPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToPhonemes": "Audio To Phonemes",
    "PhonemeToMouthShapes": "Phoneme To Mouth Shapes",
    "MouthShapePreview": "Mouth Shape Preview",
    "MouthShapeCompositor": "Mouth Shape Compositor",
    "MouthShapeLoader": "Mouth Shape Loader",
    "CreatureLipSync": "Creature Lip Sync",
    "RemoveMouthBackground": "Remove Mouth Background",
    "MouthShapeCompositorTracked": "Mouth Shape Compositor (Tracked)",
    "PointTracker": "Point Tracker",
    "PointsToMasks": "Points To Masks",
    "PointPreview": "Point Preview",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "AudioToPhonemes",
    "PhonemeToMouthShapes",
    "MouthShapePreview",
    "MouthShapeCompositor",
    "MouthShapeLoader",
    "CreatureLipSync",
    "RemoveMouthBackground",
    "MouthShapeCompositorTracked",
    "PointTracker",
    "PointsToMasks",
    "PointPreview",
]

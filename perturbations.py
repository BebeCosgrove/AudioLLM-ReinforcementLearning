"""
Audio Perturbations for Contrastive Decoding

This module provides audio perturbations designed to DESTROY specific audio information
for use in Audio-Aware Decoding (AAD). Unlike data augmentation (which preserves info),
these perturbations are intentionally aggressive.

Each perturbation targets specific audio features:
- Spectral/Timbre: What something sounds like (material, source identity)
- Temporal: When things happen, rhythm, counting events
- Frequency: High vs low pitched sounds
- Dynamics: Loudness variations, emphasis
- Spatial: Room characteristics, distance
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from enum import Enum, auto
from typing import Dict, Callable, Optional, Any, List, Tuple
from dataclasses import dataclass


class Perturbation(Enum):
    """Available perturbation types for contrastive decoding."""
    
    # === BASELINE (for comparison) ===
    ORIGINAL = auto()           # No perturbation
    NO_AUDIO = auto()           # Not implemented here but run_qwen.py / run_af3.py treat it as a special case
    
    # === ADDITIVE NOISE ===
    NOISE = auto()              # Add Gaussian noise
    COLORED_NOISE = auto()      # Add pink/brown noise (more natural)
    
    # === TEMPORAL PERTURBATIONS ===
    TIMESTRETCH = auto()        # Change speed (affects pitch)
    PITCH_SHIFT = auto()        # Change pitch (preserves speed)
    SEGMENT_SHUFFLE = auto()    # Shuffle audio segments
    SEGMENT_REVERSE = auto()    # Reverse segments individually
    REVERSE = auto()            # Reverse entire audio
    DROPOUT = auto()            # Random sample dropout
    TIME_MASK = auto()          # Mask contiguous time regions
    REPEAT_SEGMENT = auto()     # Repeat a segment (destroys uniqueness)
    
    # === FREQUENCY DOMAIN ===
    LOW_PASS = auto()           # Remove high frequencies
    HIGH_PASS = auto()          # Remove low frequencies  
    BANDPASS = auto()           # Keep only a frequency band
    BANDSTOP = auto()           # Remove a frequency band (notch)
    FREQ_MASK = auto()          # Random frequency band masking
    SPECTRAL_BLUR = auto()      # Blur spectrogram (smear frequencies)
    
    # === AMPLITUDE/DYNAMICS ===
    CLIP = auto()               # Hard clipping (distortion)
    QUANTIZE = auto()           # Reduce bit depth
    COMPRESS = auto()           # Dynamic range compression
    GATE = auto()               # Noise gate (silence quiet parts)
    GATE_INVERTED = auto()      # Inverted gate (silence loud parts, keep quiet)
    GATE_SOFT = auto()          # Soft gate (attenuate quiet parts instead of zeroing)
    GATE_INVERTED_SOFT = auto() # Soft inverted gate (attenuate loud parts)
    NORMALIZE_CHUNKS = auto()   # Normalize each chunk independently
    
    # === SPECTRAL PERTURBATIONS ===
    SPEC_NOISE = auto()         # Add noise in STFT magnitude domain (true spectral noise)
    SPEC_BLUR = auto()          # Gaussian blur on spectrogram
    HARMONIC_REMOVE = auto()    # Remove harmonic content (keep percussive)
    PERCUSSIVE_REMOVE = auto()  # Remove percussive content (keep harmonic)
    SPEC_REVERSE = auto()           # Reverse all STFT time frames
    SPEC_SEGMENT_REVERSE = auto()   # Reverse frames within each segment
    SPEC_SEGMENT_SHUFFLE = auto()   # Shuffle segments of frames
    
    # === ENVIRONMENTAL ===
    REVERB = auto()             # Add artificial reverb
    ECHO = auto()               # Add echo/delay
    PHONE_FILTER = auto()       # Telephone bandpass (300-3400 Hz)
    UNDERWATER = auto()         # Low-pass + reverb (muffled)
    
    # === DESTRUCTIVE ===
    RESAMPLE_LOW = auto()       # Downsample then upsample (lose HF)
    BIT_CRUSH = auto()          # Combine quantize + downsample


@dataclass
class PerturbationInfo:
    """Metadata about a perturbation for LLM prompting."""
    name: str
    description: str
    targets: List[str]  # What audio features this destroys
    settings: Dict[str, Dict[str, Any]]
    

# =============================================================================
# PERTURBATION SPECIFICATIONS (Aggressive settings for CD)
# =============================================================================

PERTURBATION_SPECS: Dict[Perturbation, Dict[str, Dict]] = {
    
    # === ADDITIVE NOISE ===
    Perturbation.NOISE: {
        "heavy": {"sigma": 0.3},
        "very_heavy": {"sigma": 0.5},
        "extreme": {"sigma": 0.6},
        "overwhelming": {"sigma": 1.0},
    },
    
    Perturbation.COLORED_NOISE: {
        "pink_heavy": {"color": "pink", "sigma": 0.4},
        "brown_heavy": {"color": "brown", "sigma": 0.5},
    },
    
    # === TEMPORAL ===
    Perturbation.TIMESTRETCH: {
        "very_slow": {"rate": 0.4},
        "very_fast": {"rate": 2.5},
    },
    
    Perturbation.PITCH_SHIFT: {
        # Original extreme settings
        "down_octave": {"semitones": -12},
        "up_octave": {"semitones": 12},
        "extreme_down": {"semitones": -24},
        "extreme_up": {"semitones": 24},
        # New moderate settings (less OOD, keeps events recognizable)
        "down_moderate": {"semitones": -6},
        "up_moderate": {"semitones": 6},
        "down_mild": {"semitones": -4},
        "up_mild": {"semitones": 4},
    },
    
    Perturbation.SEGMENT_SHUFFLE: {
        "coarse": {"n_segments": 10},
        "fine": {"n_segments": 50},
        "extreme": {"n_segments": 200},
    },
    
    Perturbation.SEGMENT_REVERSE: {
        "coarse": {"n_segments": 10},
        "fine": {"n_segments": 50},
    },
    
    Perturbation.REVERSE: {
        "full": {},
    },
    
    Perturbation.DROPOUT: {
        "heavy": {"p": 0.4},
        "extreme": {"p": 0.7},
    },
    
    Perturbation.TIME_MASK: {
        "light": {"n_masks": 3, "max_width": 0.08},  # New: few short masks
        "heavy": {"n_masks": 5, "max_width": 0.15},
        "extreme": {"n_masks": 10, "max_width": 0.1},
    },
    
    Perturbation.REPEAT_SEGMENT: {
        "repeat_start": {"segment": "start", "repeats": 5},
        "repeat_middle": {"segment": "middle", "repeats": 5},
    },
    
    # === FREQUENCY DOMAIN ===
    Perturbation.LOW_PASS: {
        "moderate": {"cutoff_hz": 1000},
        "aggressive": {"cutoff_hz": 500},
        "extreme": {"cutoff_hz": 250},
    },
    
    Perturbation.HIGH_PASS: {
        "moderate": {"cutoff_hz": 1000},
        "aggressive": {"cutoff_hz": 2000},
        "extreme": {"cutoff_hz": 4000},
        # New: very aggressive for removing almost all body/voicing
        "very_extreme": {"cutoff_hz": 5000},
        "ultra_extreme": {"cutoff_hz": 6000},
    },
    
    Perturbation.BANDPASS: {
        "bass_only": {"low_hz": 50, "high_hz": 300},
        "bass_wide": {"low_hz": 50, "high_hz": 500},  # New: wider bass
        "low_mid": {"low_hz": 200, "high_hz": 800},
        "mid_only": {"low_hz": 500, "high_hz": 2000},
        "mid_narrow": {"low_hz": 700, "high_hz": 1400},  # New: narrower mid
        "high_mid": {"low_hz": 1500, "high_hz": 4000},
        "high_mid_narrow": {"low_hz": 2000, "high_hz": 3500},  # New: speech/impact focus
        "treble_only": {"low_hz": 3000, "high_hz": 8000},
        "treble_extreme": {"low_hz": 4000, "high_hz": 8000},  # New: higher cutoff
        "treble_ultra": {"low_hz": 5000, "high_hz": 8000},  # New: almost only hiss
    },
    
    Perturbation.BANDSTOP: {
        "remove_mids": {"low_hz": 500, "high_hz": 2000},
        "remove_low": {"low_hz": 50, "high_hz": 500},
        "remove_high": {"low_hz": 3000, "high_hz": 8000},
    },
    
    Perturbation.FREQ_MASK: {
        "heavy": {"n_masks": 8, "max_width_hz": 500},
        "extreme": {"n_masks": 15, "max_width_hz": 400},
        "very_heavy": {"n_masks": 12, "max_width_hz": 600},  # New: more masks
    },
    
    Perturbation.SPECTRAL_BLUR: {
        "light": {"sigma_time": 3, "sigma_freq": 3},  # New: light blur
        "moderate": {"sigma_time": 5, "sigma_freq": 5},
        "heavy": {"sigma_time": 15, "sigma_freq": 15},
    },
    
    # === AMPLITUDE/DYNAMICS ===
    Perturbation.CLIP: {
        "hard": {"threshold": 0.2},
        "extreme": {"threshold": 0.1},
    },
    
    Perturbation.QUANTIZE: {
        "4bit": {"bits": 4},
        "3bit": {"bits": 3},
        "2bit": {"bits": 2},
    },
    
    Perturbation.COMPRESS: {
        "heavy": {"threshold": 0.2, "ratio": 10},
        "extreme": {"threshold": 0.1, "ratio": 20},
    },
    
    # Original hard gate (silence quiet parts)
    Perturbation.GATE: {
        "aggressive": {"threshold": 0.3},
        "extreme": {"threshold": 0.5},
        "very_extreme": {"threshold": 0.65},  # New: higher threshold
        "ultra_extreme": {"threshold": 0.75},  # New: only loudest transients survive
    },
    
    # Inverted gate: silence loud parts, keep quiet (ambience/context)
    Perturbation.GATE_INVERTED: {
        "moderate": {"threshold": 0.5},   # Keep bottom 50% energy
        "aggressive": {"threshold": 0.3}, # Keep bottom 30% energy
        "extreme": {"threshold": 0.2},    # Keep bottom 20% energy
        "very_extreme": {"threshold": 0.1}, # Keep only bottom 10% energy
    },
    
    # Soft gate: attenuate quiet parts instead of zeroing (more realistic)
    Perturbation.GATE_SOFT: {
        "soft_aggressive": {"threshold": 0.3, "attenuation_db": -18},
        "soft_extreme": {"threshold": 0.3, "attenuation_db": -30},
        "soft_very_extreme": {"threshold": 0.5, "attenuation_db": -24},
        "soft_ultra": {"threshold": 0.5, "attenuation_db": -45},
    },
    
    # Inverted soft gate: attenuate loud parts (reduce salient events)
    Perturbation.GATE_INVERTED_SOFT: {
        "soft_moderate": {"threshold": 0.5, "attenuation_db": -12},
        "soft_aggressive": {"threshold": 0.3, "attenuation_db": -18},
        "soft_extreme": {"threshold": 0.2, "attenuation_db": -30},
    },
    
    Perturbation.NORMALIZE_CHUNKS: {
        "coarse": {"n_chunks": 10},
        "fine": {"n_chunks": 50},
    },
    
    # === SPECTRAL ===
    # sigma is relative to magnitude RMS: 0.1 = light, 0.3 = heavy, 0.6 = extreme, 1.0 = overwhelming
    Perturbation.SPEC_NOISE: {
        "light": {"sigma": 0.1},
        "heavy": {"sigma": 0.3},
        "extreme": {"sigma": 0.6},
        "overwhelming": {"sigma": 1.0},
    },
    
    Perturbation.SPEC_BLUR: {
        "light": {"sigma_time": 5, "sigma_freq": 5},  # New: light blur
        "heavy": {"sigma_time": 15, "sigma_freq": 15},
        "extreme": {"sigma_time": 25, "sigma_freq": 25},
    },
    
    Perturbation.SPEC_REVERSE: {
        "full": {},
    },

    Perturbation.SPEC_SEGMENT_REVERSE: {
        "coarse": {"n_segments": 10},
        "fine":   {"n_segments": 50},
    },

    Perturbation.SPEC_SEGMENT_SHUFFLE: {
        "coarse":  {"n_segments": 10},
        "fine":    {"n_segments": 50},
        "extreme": {"n_segments": 200},
    },

    Perturbation.HARMONIC_REMOVE: {
        "full": {"margin": 3.0},
    },
    
    Perturbation.PERCUSSIVE_REMOVE: {
        "full": {"margin": 3.0},
    },
    
    # === ENVIRONMENTAL ===
    Perturbation.REVERB: {
        "large_hall": {"decay": 0.8, "delay": 0.05},
        "extreme": {"decay": 0.95, "delay": 0.1},
    },
    
    Perturbation.ECHO: {
        "short": {"delay_ms": 100, "decay": 0.6, "repeats": 5},
        "long": {"delay_ms": 300, "decay": 0.7, "repeats": 8},
    },
    
    Perturbation.PHONE_FILTER: {
        "standard": {"low_hz": 300, "high_hz": 3400},
        "narrow": {"low_hz": 500, "high_hz": 2500},  # New: narrower than standard
    },
    
    Perturbation.UNDERWATER: {
        "deep": {"cutoff_hz": 400, "reverb_decay": 0.7},
    },
    
    # === DESTRUCTIVE ===
    Perturbation.RESAMPLE_LOW: {
        "8khz": {"target_sr": 8000},
        "4khz": {"target_sr": 4000},
        "2khz": {"target_sr": 2000},
    },
    
    Perturbation.BIT_CRUSH: {
        "retro": {"bits": 4, "target_sr": 8000},
        "extreme": {"bits": 2, "target_sr": 4000},
    },
}


# =============================================================================
# PERTURBATION METADATA (for LLM prompting)
# =============================================================================

PERTURBATION_INFO: Dict[Perturbation, PerturbationInfo] = {
    
    Perturbation.NOISE: PerturbationInfo(
        name="NOISE",
        description="Adds Gaussian white noise that masks the original audio",
        targets=["clarity", "quiet_sounds", "background_sounds", "speech_intelligibility"],
        settings=PERTURBATION_SPECS[Perturbation.NOISE]
    ),
    
    Perturbation.COLORED_NOISE: PerturbationInfo(
        name="COLORED_NOISE",
        description="Adds natural-sounding pink or brown noise",
        targets=["clarity", "ambient_sounds", "subtle_details"],
        settings=PERTURBATION_SPECS[Perturbation.COLORED_NOISE]
    ),
    
    Perturbation.TIMESTRETCH: PerturbationInfo(
        name="TIMESTRETCH",
        description="Drastically changes playback speed and pitch",
        targets=["rhythm", "tempo", "pitch", "speaker_identity", "music_recognition"],
        settings=PERTURBATION_SPECS[Perturbation.TIMESTRETCH]
    ),
    
    Perturbation.PITCH_SHIFT: PerturbationInfo(
        name="PITCH_SHIFT",
        description="Shifts all frequencies up or down (preserves tempo)",
        targets=["pitch", "speaker_identity", "instrument_recognition", "animal_sounds"],
        settings=PERTURBATION_SPECS[Perturbation.PITCH_SHIFT]
    ),
    
    Perturbation.SEGMENT_SHUFFLE: PerturbationInfo(
        name="SEGMENT_SHUFFLE",
        description="Randomly shuffles audio segments, destroying temporal order",
        targets=["temporal_order", "counting", "sequences", "before_after", "rhythm"],
        settings=PERTURBATION_SPECS[Perturbation.SEGMENT_SHUFFLE]
    ),
    
    Perturbation.SEGMENT_REVERSE: PerturbationInfo(
        name="SEGMENT_REVERSE",
        description="Reverses audio within segments, creating unnatural sound",
        targets=["temporal_order", "attack_sounds", "speech", "natural_sounds"],
        settings=PERTURBATION_SPECS[Perturbation.SEGMENT_REVERSE]
    ),
    
    Perturbation.REVERSE: PerturbationInfo(
        name="REVERSE",
        description="Plays entire audio backwards",
        targets=["temporal_order", "speech", "sequences", "natural_flow"],
        settings=PERTURBATION_SPECS[Perturbation.REVERSE]
    ),
    
    Perturbation.DROPOUT: PerturbationInfo(
        name="DROPOUT",
        description="Randomly drops audio samples, creating choppy sound",
        targets=["continuity", "smooth_sounds", "sustained_notes", "speech_clarity"],
        settings=PERTURBATION_SPECS[Perturbation.DROPOUT]
    ),
    
    Perturbation.TIME_MASK: PerturbationInfo(
        name="TIME_MASK",
        description="Masks contiguous time regions with silence",
        targets=["events", "counting", "specific_moments", "temporal_coverage"],
        settings=PERTURBATION_SPECS[Perturbation.TIME_MASK]
    ),
    
    Perturbation.REPEAT_SEGMENT: PerturbationInfo(
        name="REPEAT_SEGMENT",
        description="Repeats one segment multiple times, masking variety",
        targets=["variety", "counting", "different_sounds", "progression"],
        settings=PERTURBATION_SPECS[Perturbation.REPEAT_SEGMENT]
    ),
    
    Perturbation.LOW_PASS: PerturbationInfo(
        name="LOW_PASS",
        description="Removes high frequencies, making audio muffled",
        targets=["high_pitched_sounds", "birds", "whistles", "bells", "speech_clarity", 
                 "cymbals", "hissing", "sibilance", "brightness"],
        settings=PERTURBATION_SPECS[Perturbation.LOW_PASS]
    ),
    
    Perturbation.HIGH_PASS: PerturbationInfo(
        name="HIGH_PASS",
        description="Removes low frequencies, making audio thin",
        targets=["low_pitched_sounds", "bass", "thunder", "engines", "drums", 
                 "rumble", "footsteps", "explosions", "male_voice"],
        settings=PERTURBATION_SPECS[Perturbation.HIGH_PASS]
    ),
    
    Perturbation.BANDPASS: PerturbationInfo(
        name="BANDPASS",
        description="Keeps only a specific frequency range",
        targets=["frequency_specific_sounds", "full_spectrum_sounds", "timbre"],
        settings=PERTURBATION_SPECS[Perturbation.BANDPASS]
    ),
    
    Perturbation.BANDSTOP: PerturbationInfo(
        name="BANDSTOP", 
        description="Removes a specific frequency range (notch filter)",
        targets=["specific_frequency_sounds", "resonant_sounds"],
        settings=PERTURBATION_SPECS[Perturbation.BANDSTOP]
    ),
    
    Perturbation.FREQ_MASK: PerturbationInfo(
        name="FREQ_MASK",
        description="Randomly masks multiple frequency bands",
        targets=["timbre", "harmonic_content", "instrument_identity", "source_recognition"],
        settings=PERTURBATION_SPECS[Perturbation.FREQ_MASK]
    ),
    
    Perturbation.SPECTRAL_BLUR: PerturbationInfo(
        name="SPECTRAL_BLUR",
        description="Blurs the spectrogram, smearing time and frequency",
        targets=["transients", "attacks", "timing_precision", "frequency_precision"],
        settings=PERTURBATION_SPECS[Perturbation.SPECTRAL_BLUR]
    ),
    
    Perturbation.CLIP: PerturbationInfo(
        name="CLIP",
        description="Hard clips the audio, causing severe distortion",
        targets=["dynamics", "timbre", "waveform_shape", "clarity", "loudness_variation"],
        settings=PERTURBATION_SPECS[Perturbation.CLIP]
    ),
    
    Perturbation.QUANTIZE: PerturbationInfo(
        name="QUANTIZE",
        description="Reduces bit depth, adding quantization noise",
        targets=["subtle_sounds", "timbre", "dynamics", "quiet_passages"],
        settings=PERTURBATION_SPECS[Perturbation.QUANTIZE]
    ),
    
    Perturbation.COMPRESS: PerturbationInfo(
        name="COMPRESS",
        description="Extreme dynamic range compression, flattens loudness",
        targets=["dynamics", "loudness_variation", "emphasis", "volume_changes"],
        settings=PERTURBATION_SPECS[Perturbation.COMPRESS]
    ),
    
    Perturbation.GATE: PerturbationInfo(
        name="GATE",
        description="Silences quiet parts, only loud sounds remain (hard gate)",
        targets=["quiet_sounds", "background", "ambient", "subtle_details", "reverb_tails"],
        settings=PERTURBATION_SPECS[Perturbation.GATE]
    ),
    
    Perturbation.GATE_INVERTED: PerturbationInfo(
        name="GATE_INVERTED",
        description="Silences loud parts, keeps only quiet ambience/context",
        targets=["salient_events", "loud_sounds", "impacts", "speech_bursts", "transients"],
        settings=PERTURBATION_SPECS[Perturbation.GATE_INVERTED]
    ),
    
    Perturbation.GATE_SOFT: PerturbationInfo(
        name="GATE_SOFT",
        description="Attenuates quiet parts instead of zeroing (more realistic gate)",
        targets=["quiet_sounds", "background", "ambient", "subtle_details"],
        settings=PERTURBATION_SPECS[Perturbation.GATE_SOFT]
    ),
    
    Perturbation.GATE_INVERTED_SOFT: PerturbationInfo(
        name="GATE_INVERTED_SOFT",
        description="Attenuates loud parts, reducing salient events while keeping ambience",
        targets=["salient_events", "loud_sounds", "impacts", "emphasis"],
        settings=PERTURBATION_SPECS[Perturbation.GATE_INVERTED_SOFT]
    ),
    
    Perturbation.NORMALIZE_CHUNKS: PerturbationInfo(
        name="NORMALIZE_CHUNKS",
        description="Normalizes each chunk independently, destroys relative loudness",
        targets=["loudness_variation", "dynamics", "distance_cues", "emphasis"],
        settings=PERTURBATION_SPECS[Perturbation.NORMALIZE_CHUNKS]
    ),
    
    Perturbation.SPEC_NOISE: PerturbationInfo(
        name="SPEC_NOISE",
        description="Adds Gaussian noise directly to the STFT magnitude spectrogram; degrades spectral detail without waveform-domain clipping artifacts",
        targets=["timbre", "texture", "spectral_detail", "fine_grained_audio_events", "subtle_background_sounds"],
        settings=PERTURBATION_SPECS[Perturbation.SPEC_NOISE]
    ),

    Perturbation.SPEC_REVERSE: PerturbationInfo(
        name="SPEC_REVERSE",
        description="Reverses all STFT time frames; spectral analog of waveform REVERSE",
        targets=["temporal_order", "sequences", "before_after", "speech", "natural_flow"],
        settings=PERTURBATION_SPECS[Perturbation.SPEC_REVERSE]
    ),

    Perturbation.SPEC_SEGMENT_REVERSE: PerturbationInfo(
        name="SPEC_SEGMENT_REVERSE",
        description="Reverses spectral frames within each segment; spectral analog of SEGMENT_REVERSE",
        targets=["temporal_order", "local_motion", "attack_sounds", "natural_sounds"],
        settings=PERTURBATION_SPECS[Perturbation.SPEC_SEGMENT_REVERSE]
    ),

    Perturbation.SPEC_SEGMENT_SHUFFLE: PerturbationInfo(
        name="SPEC_SEGMENT_SHUFFLE",
        description="Shuffles segments of STFT time frames; spectral analog of SEGMENT_SHUFFLE",
        targets=["temporal_order", "counting", "sequences", "before_after", "rhythm"],
        settings=PERTURBATION_SPECS[Perturbation.SPEC_SEGMENT_SHUFFLE]
    ),

    Perturbation.HARMONIC_REMOVE: PerturbationInfo(
        name="HARMONIC_REMOVE",
        description="Removes harmonic content, keeps only percussive/noise",
        targets=["tonal_sounds", "music", "singing", "instruments", "pitched_sounds"],
        settings=PERTURBATION_SPECS[Perturbation.HARMONIC_REMOVE]
    ),
    
    Perturbation.PERCUSSIVE_REMOVE: PerturbationInfo(
        name="PERCUSSIVE_REMOVE",
        description="Removes percussive content, keeps only harmonic/tonal",
        targets=["impacts", "drums", "clicks", "transients", "footsteps", "knocking"],
        settings=PERTURBATION_SPECS[Perturbation.PERCUSSIVE_REMOVE]
    ),
    
    Perturbation.REVERB: PerturbationInfo(
        name="REVERB",
        description="Adds heavy artificial reverb, blurs temporal details",
        targets=["clarity", "transients", "speech_intelligibility", "timing", "dryness"],
        settings=PERTURBATION_SPECS[Perturbation.REVERB]
    ),
    
    Perturbation.ECHO: PerturbationInfo(
        name="ECHO",
        description="Adds echo/delay, creates confusing repetitions",
        targets=["counting", "single_events", "clarity", "timing"],
        settings=PERTURBATION_SPECS[Perturbation.ECHO]
    ),
    
    Perturbation.PHONE_FILTER: PerturbationInfo(
        name="PHONE_FILTER",
        description="Applies telephone-quality bandpass filter (300-3400 Hz)",
        targets=["high_frequencies", "low_frequencies", "full_bandwidth", "music_quality"],
        settings=PERTURBATION_SPECS[Perturbation.PHONE_FILTER]
    ),
    
    Perturbation.UNDERWATER: PerturbationInfo(
        name="UNDERWATER",
        description="Simulates underwater sound (heavy low-pass + reverb)",
        targets=["high_frequencies", "clarity", "speech", "detail"],
        settings=PERTURBATION_SPECS[Perturbation.UNDERWATER]
    ),
    
    Perturbation.RESAMPLE_LOW: PerturbationInfo(
        name="RESAMPLE_LOW",
        description="Downsamples to low sample rate, losing high frequencies",
        targets=["high_frequencies", "detail", "sibilance", "brightness"],
        settings=PERTURBATION_SPECS[Perturbation.RESAMPLE_LOW]
    ),
    
    Perturbation.BIT_CRUSH: PerturbationInfo(
        name="BIT_CRUSH",
        description="Combines low sample rate and bit depth for maximum destruction",
        targets=["quality", "detail", "timbre", "dynamics", "clarity"],
        settings=PERTURBATION_SPECS[Perturbation.BIT_CRUSH]
    ),
}


# =============================================================================
# PERTURBATION IMPLEMENTATIONS
# =============================================================================

def apply_noise(audio: np.ndarray, sigma: float = 0.3) -> np.ndarray:
    """Add Gaussian white noise."""
    noise = np.random.randn(*audio.shape).astype(np.float32) * sigma
    return audio + noise


def apply_colored_noise(audio: np.ndarray, color: str = "pink", sigma: float = 0.4) -> np.ndarray:
    """Add colored noise (pink or brown)."""
    n = len(audio)
    white = np.random.randn(n).astype(np.float32)
    
    if color == "pink":
        # Pink noise: 1/f spectrum
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1  # Avoid division by zero
        fft = fft / np.sqrt(freqs)
        noise = np.fft.irfft(fft, n).astype(np.float32)
    elif color == "brown":
        # Brown noise: 1/f^2 spectrum (cumulative sum approximation)
        noise = np.cumsum(white).astype(np.float32)
        noise = noise - np.mean(noise)
    else:
        noise = white
    
    noise = noise / (np.std(noise) + 1e-8) * sigma
    return audio + noise


def apply_timestretch(audio: np.ndarray, rate: float = 0.5, sr: int = 16000) -> np.ndarray:
    """Time stretch audio (affects pitch)."""
    return librosa.effects.time_stretch(audio, rate=rate)


def apply_pitch_shift(audio: np.ndarray, semitones: int = 12, sr: int = 16000) -> np.ndarray:
    """Shift pitch by semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def apply_segment_shuffle(audio: np.ndarray, n_segments: int = 50) -> np.ndarray:
    """Shuffle audio segments randomly."""
    segment_len = len(audio) // n_segments
    if segment_len < 1:
        return audio
    
    segments = [audio[i*segment_len:(i+1)*segment_len] for i in range(n_segments)]
    # Handle remainder
    remainder = audio[n_segments*segment_len:]
    
    np.random.shuffle(segments)
    result = np.concatenate(segments)
    if len(remainder) > 0:
        result = np.concatenate([result, remainder])
    
    return result


def apply_segment_reverse(audio: np.ndarray, n_segments: int = 10) -> np.ndarray:
    """Reverse each segment individually."""
    segment_len = len(audio) // n_segments
    if segment_len < 1:
        return audio
    
    result = audio.copy()
    for i in range(n_segments):
        start = i * segment_len
        end = (i + 1) * segment_len
        result[start:end] = result[start:end][::-1]
    
    return result


def apply_reverse(audio: np.ndarray) -> np.ndarray:
    """Reverse entire audio."""
    return audio[::-1].copy()


def apply_dropout(audio: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly drop samples."""
    mask = np.random.random(len(audio)) > p
    result = audio.copy()
    result[~mask] = 0
    return result


def apply_time_mask(audio: np.ndarray, n_masks: int = 5, max_width: float = 0.15) -> np.ndarray:
    """Mask contiguous time regions."""
    result = audio.copy()
    audio_len = len(audio)
    
    for _ in range(n_masks):
        width = int(np.random.uniform(0.05, max_width) * audio_len)
        start = np.random.randint(0, max(1, audio_len - width))
        result[start:start+width] = 0
    
    return result


def apply_repeat_segment(audio: np.ndarray, segment: str = "start", repeats: int = 5) -> np.ndarray:
    """Repeat a segment multiple times."""
    segment_len = len(audio) // 5  # Take 20% of audio
    
    if segment == "start":
        seg = audio[:segment_len]
    elif segment == "middle":
        mid = len(audio) // 2
        seg = audio[mid - segment_len//2:mid + segment_len//2]
    else:  # end
        seg = audio[-segment_len:]
    
    return np.tile(seg, repeats)[:len(audio)]


def apply_low_pass(audio: np.ndarray, cutoff_hz: float = 500, sr: int = 16000) -> np.ndarray:
    """Apply low-pass filter."""
    nyq = sr / 2
    normalized_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(5, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio).astype(np.float32)


def apply_high_pass(audio: np.ndarray, cutoff_hz: float = 2000, sr: int = 16000) -> np.ndarray:
    """Apply high-pass filter."""
    nyq = sr / 2
    normalized_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(5, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio).astype(np.float32)


def apply_bandpass(audio: np.ndarray, low_hz: float = 500, high_hz: float = 2000, sr: int = 16000) -> np.ndarray:
    """Apply bandpass filter."""
    nyq = sr / 2
    low = min(low_hz / nyq, 0.99)
    high = min(high_hz / nyq, 0.99)
    if low >= high:
        low = high - 0.01
    b, a = signal.butter(5, [low, high], btype='band')
    return signal.filtfilt(b, a, audio).astype(np.float32)


def apply_bandstop(audio: np.ndarray, low_hz: float = 500, high_hz: float = 2000, sr: int = 16000) -> np.ndarray:
    """Apply bandstop (notch) filter."""
    nyq = sr / 2
    low = min(low_hz / nyq, 0.99)
    high = min(high_hz / nyq, 0.99)
    if low >= high:
        low = high - 0.01
    b, a = signal.butter(5, [low, high], btype='bandstop')
    return signal.filtfilt(b, a, audio).astype(np.float32)


def apply_freq_mask(audio: np.ndarray, n_masks: int = 8, max_width_hz: float = 500, sr: int = 16000) -> np.ndarray:
    """Mask random frequency bands in STFT domain."""
    stft = librosa.stft(audio)
    n_bins = stft.shape[0]
    hz_per_bin = (sr / 2) / n_bins
    
    for _ in range(n_masks):
        width_bins = int(np.random.uniform(100, max_width_hz) / hz_per_bin)
        start = np.random.randint(0, max(1, n_bins - width_bins))
        stft[start:start+width_bins, :] = 0
    
    return librosa.istft(stft, length=len(audio)).astype(np.float32)


def apply_spectral_blur(audio: np.ndarray, sigma_time: float = 10, sigma_freq: float = 10, sr: int = 16000) -> np.ndarray:
    """Apply Gaussian blur to spectrogram."""
    stft = librosa.stft(audio)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Blur magnitude
    mag_blurred = gaussian_filter1d(mag, sigma=sigma_freq, axis=0)
    mag_blurred = gaussian_filter1d(mag_blurred, sigma=sigma_time, axis=1)
    
    stft_blurred = mag_blurred * np.exp(1j * phase)
    return librosa.istft(stft_blurred, length=len(audio)).astype(np.float32)


def apply_clip(audio: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Hard clip audio at threshold."""
    return np.clip(audio, -threshold, threshold)


def apply_quantize(audio: np.ndarray, bits: int = 3) -> np.ndarray:
    """Quantize to fewer bits."""
    levels = 2 ** bits
    audio_norm = (audio - audio.min()) / (audio.max() - audio.min() + 1e-8)
    quantized = np.round(audio_norm * (levels - 1)) / (levels - 1)
    return (quantized * (audio.max() - audio.min()) + audio.min()).astype(np.float32)


def apply_compress(audio: np.ndarray, threshold: float = 0.2, ratio: float = 10) -> np.ndarray:
    """Apply dynamic range compression."""
    result = audio.copy()
    above_thresh = np.abs(audio) > threshold
    
    excess = np.abs(audio[above_thresh]) - threshold
    compressed_excess = excess / ratio
    result[above_thresh] = np.sign(audio[above_thresh]) * (threshold + compressed_excess)
    
    return result


def apply_gate(audio: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Apply noise gate - silence quiet parts (hard gate)."""
    result = audio.copy()
    result[np.abs(audio) < threshold] = 0
    return result


def apply_gate_inverted(audio: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Inverted gate: silence loud parts, keep quiet (ambience/context).
    
    This removes salient events (bangs, speech bursts, impacts) and preserves
    only the quiet background/ambience. Useful for testing if the model relies
    on obvious events vs. context.
    """
    result = audio.copy()
    result[np.abs(audio) >= threshold] = 0
    return result


def apply_gate_soft(audio: np.ndarray, threshold: float = 0.3, attenuation_db: float = -24) -> np.ndarray:
    """
    Soft gate: attenuate quiet parts instead of zeroing.
    
    More realistic than hard gate - doesn't create sharp artifacts.
    The neg branch remains coherent without "digital silence" dropouts.
    
    Args:
        threshold: Amplitude threshold below which to attenuate
        attenuation_db: How much to reduce quiet parts (negative dB value)
    """
    result = audio.copy()
    
    # Convert dB to linear gain
    gain = 10 ** (attenuation_db / 20)
    
    # Attenuate samples below threshold
    quiet_mask = np.abs(audio) < threshold
    result[quiet_mask] = result[quiet_mask] * gain
    
    return result


def apply_gate_inverted_soft(audio: np.ndarray, threshold: float = 0.3, attenuation_db: float = -18) -> np.ndarray:
    """
    Soft inverted gate: attenuate loud parts instead of zeroing.
    
    Reduces salient events while preserving ambience. Tests whether the model's
    answer depends on loud discriminative events.
    
    Args:
        threshold: Amplitude threshold above which to attenuate
        attenuation_db: How much to reduce loud parts (negative dB value)
    """
    result = audio.copy()
    
    # Convert dB to linear gain
    gain = 10 ** (attenuation_db / 20)
    
    # Attenuate samples above threshold
    loud_mask = np.abs(audio) >= threshold
    result[loud_mask] = result[loud_mask] * gain
    
    return result


def apply_normalize_chunks(audio: np.ndarray, n_chunks: int = 20) -> np.ndarray:
    """Normalize each chunk independently."""
    chunk_len = len(audio) // n_chunks
    if chunk_len < 1:
        return audio
    
    result = audio.copy()
    for i in range(n_chunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len
        chunk = result[start:end]
        max_val = np.max(np.abs(chunk)) + 1e-8
        result[start:end] = chunk / max_val
    
    return result


def apply_harmonic_remove(audio: np.ndarray, margin: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Remove harmonic content, keep percussive."""
    harmonic, percussive = librosa.effects.hpss(audio, margin=margin)
    return percussive


def apply_percussive_remove(audio: np.ndarray, margin: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Remove percussive content, keep harmonic."""
    harmonic, percussive = librosa.effects.hpss(audio, margin=margin)
    return harmonic


def apply_reverb(audio: np.ndarray, decay: float = 0.8, delay: float = 0.05, sr: int = 16000) -> np.ndarray:
    """Add artificial reverb using simple comb filter."""
    delay_samples = int(delay * sr)
    result = audio.copy()
    
    for i in range(5):  # Multiple delay lines
        d = delay_samples * (i + 1)
        gain = decay ** (i + 1)
        if d < len(audio):
            result[d:] += audio[:-d] * gain
    
    # Normalize
    result = result / (np.max(np.abs(result)) + 1e-8) * np.max(np.abs(audio))
    return result.astype(np.float32)


def apply_echo(audio: np.ndarray, delay_ms: float = 200, decay: float = 0.6, repeats: int = 5, sr: int = 16000) -> np.ndarray:
    """Add echo effect."""
    delay_samples = int(delay_ms / 1000 * sr)
    result = np.zeros(len(audio) + delay_samples * repeats, dtype=np.float32)
    result[:len(audio)] = audio
    
    for i in range(repeats):
        offset = delay_samples * (i + 1)
        gain = decay ** (i + 1)
        if offset < len(result):
            end_idx = min(offset + len(audio), len(result))
            result[offset:end_idx] += audio[:end_idx-offset] * gain
    
    return result[:len(audio)]


def apply_phone_filter(audio: np.ndarray, low_hz: float = 300, high_hz: float = 3400, sr: int = 16000) -> np.ndarray:
    """Apply telephone-quality bandpass filter."""
    return apply_bandpass(audio, low_hz, high_hz, sr)


def apply_underwater(audio: np.ndarray, cutoff_hz: float = 400, reverb_decay: float = 0.7, sr: int = 16000) -> np.ndarray:
    """Simulate underwater sound."""
    filtered = apply_low_pass(audio, cutoff_hz, sr)
    return apply_reverb(filtered, decay=reverb_decay, delay=0.03, sr=sr)


def apply_resample_low(audio: np.ndarray, target_sr: int = 4000, sr: int = 16000) -> np.ndarray:
    """Downsample and upsample to lose high frequencies."""
    downsampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    upsampled = librosa.resample(downsampled, orig_sr=target_sr, target_sr=sr)
    # Match length
    if len(upsampled) > len(audio):
        upsampled = upsampled[:len(audio)]
    elif len(upsampled) < len(audio):
        upsampled = np.pad(upsampled, (0, len(audio) - len(upsampled)))
    return upsampled


def apply_bit_crush(audio: np.ndarray, bits: int = 3, target_sr: int = 4000, sr: int = 16000) -> np.ndarray:
    """Combine low sample rate and bit depth."""
    resampled = apply_resample_low(audio, target_sr, sr)
    return apply_quantize(resampled, bits)


def apply_spec_noise(audio: np.ndarray, sigma: float = 0.3, sr: int = 16000) -> np.ndarray:
    """
    Add Gaussian noise in the STFT magnitude domain.

    Unlike waveform-domain noise (NOISE), this perturbs the spectral representation
    directly: the magnitude at each (frequency, time) bin is corrupted by noise
    proportional to the overall magnitude RMS. Phase is left intact, so temporal
    structure is preserved while spectral texture and timbre are degraded.

    sigma: noise level relative to magnitude RMS.
           0.1 = light, 0.3 = heavy, 0.6 = extreme, 1.0 = overwhelming.
    """
    n_fft = 1024
    hop_length = 256
    stft = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))
    rms = float(np.sqrt(np.mean(magnitude ** 2))) + 1e-8
    noise = np.random.randn(*magnitude.shape).astype(np.float32) * sigma * rms
    magnitude_noisy = np.maximum(magnitude + noise, 0.0)
    stft_noisy = magnitude_noisy * phase
    return librosa.istft(stft_noisy, hop_length=hop_length, length=len(audio)).astype(np.float32)


def apply_spec_reverse(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Reverse all STFT time frames. Spectral analog of apply_reverse."""
    stft = librosa.stft(audio.astype(np.float32), n_fft=1024, hop_length=256)
    stft_rev = stft[:, ::-1]
    return librosa.istft(stft_rev, hop_length=256, length=len(audio)).astype(np.float32)


def apply_spec_segment_reverse(audio: np.ndarray, n_segments: int = 10, sr: int = 16000) -> np.ndarray:
    """Reverse STFT frames within each segment. Spectral analog of apply_segment_reverse."""
    stft = librosa.stft(audio.astype(np.float32), n_fft=1024, hop_length=256)
    n_frames = stft.shape[1]
    seg_len = n_frames // n_segments
    if seg_len < 1:
        return audio

    result = stft.copy()
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len
        result[:, start:end] = stft[:, start:end][:, ::-1]

    return librosa.istft(result, hop_length=256, length=len(audio)).astype(np.float32)


def apply_spec_segment_shuffle(audio: np.ndarray, n_segments: int = 50, sr: int = 16000) -> np.ndarray:
    """Shuffle segments of STFT time frames. Spectral analog of apply_segment_shuffle."""
    stft = librosa.stft(audio.astype(np.float32), n_fft=1024, hop_length=256)
    n_frames = stft.shape[1]
    seg_len = n_frames // n_segments
    if seg_len < 1:
        return audio

    segments = [stft[:, i*seg_len:(i+1)*seg_len] for i in range(n_segments)]
    remainder = stft[:, n_segments*seg_len:]

    np.random.shuffle(segments)
    result = np.concatenate(segments, axis=1)
    if remainder.size:
        result = np.concatenate([result, remainder], axis=1)

    return librosa.istft(result, hop_length=256, length=len(audio)).astype(np.float32)


# =============================================================================
# MAIN INTERFACE
# =============================================================================

# Mapping from Perturbation enum to function
PERTURBATION_FUNCTIONS: Dict[Perturbation, Callable] = {
    Perturbation.ORIGINAL: lambda x, **kw: x.copy(),
    Perturbation.NO_AUDIO: lambda x, **kw: np.zeros_like(x),
    Perturbation.NOISE: apply_noise,
    Perturbation.COLORED_NOISE: apply_colored_noise,
    Perturbation.TIMESTRETCH: apply_timestretch,
    Perturbation.PITCH_SHIFT: apply_pitch_shift,
    Perturbation.SEGMENT_SHUFFLE: apply_segment_shuffle,
    Perturbation.SEGMENT_REVERSE: apply_segment_reverse,
    Perturbation.REVERSE: apply_reverse,
    Perturbation.DROPOUT: apply_dropout,
    Perturbation.TIME_MASK: apply_time_mask,
    Perturbation.REPEAT_SEGMENT: apply_repeat_segment,
    Perturbation.LOW_PASS: apply_low_pass,
    Perturbation.HIGH_PASS: apply_high_pass,
    Perturbation.BANDPASS: apply_bandpass,
    Perturbation.BANDSTOP: apply_bandstop,
    Perturbation.FREQ_MASK: apply_freq_mask,
    Perturbation.SPECTRAL_BLUR: apply_spectral_blur,
    Perturbation.CLIP: apply_clip,
    Perturbation.QUANTIZE: apply_quantize,
    Perturbation.COMPRESS: apply_compress,
    Perturbation.GATE: apply_gate,
    Perturbation.GATE_INVERTED: apply_gate_inverted,
    Perturbation.GATE_SOFT: apply_gate_soft,
    Perturbation.GATE_INVERTED_SOFT: apply_gate_inverted_soft,
    Perturbation.NORMALIZE_CHUNKS: apply_normalize_chunks,
    Perturbation.SPEC_NOISE: apply_spec_noise,
    Perturbation.SPEC_BLUR: apply_spectral_blur,
    Perturbation.SPEC_REVERSE: apply_spec_reverse,
    Perturbation.SPEC_SEGMENT_REVERSE: apply_spec_segment_reverse,
    Perturbation.SPEC_SEGMENT_SHUFFLE: apply_spec_segment_shuffle,
    Perturbation.HARMONIC_REMOVE: apply_harmonic_remove,
    Perturbation.PERCUSSIVE_REMOVE: apply_percussive_remove,
    Perturbation.REVERB: apply_reverb,
    Perturbation.ECHO: apply_echo,
    Perturbation.PHONE_FILTER: apply_phone_filter,
    Perturbation.UNDERWATER: apply_underwater,
    Perturbation.RESAMPLE_LOW: apply_resample_low,
    Perturbation.BIT_CRUSH: apply_bit_crush,
}


def get_perturbation(pert_type: Perturbation, setting: Optional[str] = None, sr: int = 16000) -> Callable:
    """
    Get a perturbation function with the specified setting.
    
    Args:
        pert_type: Perturbation enum value
        setting: Setting name (e.g., "heavy", "extreme")
        sr: Sample rate for frequency-based perturbations
    
    Returns:
        Callable that takes audio array and returns perturbed audio
    """
    base_fn = PERTURBATION_FUNCTIONS.get(pert_type)
    if base_fn is None:
        raise ValueError(f"Unknown perturbation: {pert_type}")
    
    # Get settings
    if pert_type in PERTURBATION_SPECS and setting:
        if setting not in PERTURBATION_SPECS[pert_type]:
            available = list(PERTURBATION_SPECS[pert_type].keys())
            raise ValueError(f"Unknown setting '{setting}' for {pert_type}. Available: {available}")
        params = PERTURBATION_SPECS[pert_type][setting].copy()
    else:
        params = {}
    
    # Add sample rate for frequency-based perturbations
    if pert_type in [Perturbation.LOW_PASS, Perturbation.HIGH_PASS, Perturbation.BANDPASS,
                     Perturbation.BANDSTOP, Perturbation.FREQ_MASK, Perturbation.SPECTRAL_BLUR,
                     Perturbation.REVERB, Perturbation.ECHO, Perturbation.PHONE_FILTER,
                     Perturbation.UNDERWATER, Perturbation.RESAMPLE_LOW, Perturbation.BIT_CRUSH,
                     Perturbation.TIMESTRETCH, Perturbation.PITCH_SHIFT, Perturbation.HARMONIC_REMOVE,
                     Perturbation.PERCUSSIVE_REMOVE, Perturbation.SPEC_REVERSE,
                     Perturbation.SPEC_SEGMENT_REVERSE, Perturbation.SPEC_SEGMENT_SHUFFLE]:
        params['sr'] = sr
    
    return lambda audio: base_fn(audio, **params)


def list_perturbations() -> List[str]:
    """List all available perturbation names."""
    return [p.name for p in Perturbation if p != Perturbation.ORIGINAL]


def list_settings(pert_type: Perturbation) -> List[str]:
    """List available settings for a perturbation."""
    if pert_type in PERTURBATION_SPECS:
        return list(PERTURBATION_SPECS[pert_type].keys())
    return []


def get_perturbation_info(pert_type: Perturbation) -> Optional[PerturbationInfo]:
    """Get metadata about a perturbation."""
    return PERTURBATION_INFO.get(pert_type)


# =============================================================================
# GENERATE PROMPT TEXT (for LLM)
# =============================================================================

def generate_perturbation_descriptions(exclude: List[str] = None) -> str:
    """
    Generate formatted perturbation descriptions for LLM prompting.
    
    Args:
        exclude: List of perturbation names to exclude (e.g., ["ORIGINAL"])
    """
    exclude = exclude or ["ORIGINAL"]
    
    lines = ["Available Perturbations:\n"]
    
    for pert in Perturbation:
        if pert.name in exclude:
            continue
        
        info = PERTURBATION_INFO.get(pert)
        if info is None:
            continue
        
        settings_str = ", ".join(info.settings.keys()) if info.settings else "none"
        targets_str = ", ".join(info.targets[:5])  # First 5 targets
        
        lines.append(f"- {info.name}")
        lines.append(f"  Description: {info.description}")
        lines.append(f"  Destroys: {targets_str}")
        lines.append(f"  Settings: {settings_str}")
        lines.append("")
    
    return "\n".join(lines)


def generate_compact_perturbation_list(exclude: List[str] = None) -> str:
    """Generate a compact one-line list for prompting."""
    exclude = exclude or ["ORIGINAL"]
    
    items = []
    for pert in Perturbation:
        if pert.name in exclude:
            continue
        if pert not in PERTURBATION_INFO:
            continue
        
        info = PERTURBATION_INFO[pert]
        if info.settings:
            settings = "/".join(info.settings.keys())
            items.append(f"{info.name} ({settings})")
        else:
            items.append(info.name)
    
    return ", ".join(items)




# =============================================================================
# TESTING (only runs when executed directly)
# =============================================================================

if __name__ == "__main__":
    import json
    import os
    import librosa
    import soundfile as sf


    INPUT_JSON = "datasets/ah_existence/train.json"
    OUTPUT_JSON = "datasets/ah_existence/train_with_reverse.json"

    OUTPUT_AUDIO_DIR = "datasets/ah_existence/perturbed_audio"
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    with open(INPUT_JSON) as f:
        data = json.load(f)

    new_data = []

    pert_type = Perturbation.NO_AUDIO
    setting = "full"

    pert_fn = get_perturbation(pert_type, setting, sr=16000)

    

    for item in data:

        audio_path = item["path"]

        try:
            audio, _ = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Failed: {audio_path}")
            continue

        pert_audio = pert_fn(audio)

        stem = os.path.splitext(os.path.basename(audio_path))[0]

        pert_path = os.path.join(
            OUTPUT_AUDIO_DIR,
            f"{stem}_{pert_type.name.lower()}.wav"
        )

        sf.write(pert_path, pert_audio, 16000)

        new_item = item.copy()
        new_item["perturbed_path"] = pert_path
        new_item["perturbation"] = pert_type.name
        new_item["perturbation_setting"] = setting

        new_data.append(new_item)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Saved {len(new_data)} examples")

    # # # --- CONFIG ---
    # # clotho_audio_dir = "datasets/clotho_aqa/audio_files"
    # output_base = "debug/perturbation_samples"
    # audio, sr = sf.read("/data/not_backed_up/cosgrv/af3_project/data/_UvwGWvKmcg_1.wav")

    # if len(audio.shape) > 1:
    #     audio = audio.mean(axis=1)

    # audio = audio.astype("float32")

    # audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    # sr = 16000

    # # Create a subfolder for this sample
    # sample_folder = os.path.join(output_base, "test")
    # os.makedirs(sample_folder, exist_ok=True)

    # # Save the original
    
    # pert_audio = apply_reverse(audio)
    # out_path = os.path.join(sample_folder, "test_pert1.wav")
    # sf.write(out_path, pert_audio, sr)
    # print(f"Saved")
    
"""
Audio Perturbations for Contrastive Decoding

Each perturbation targets specific audio features to be used as a negative branch
in Audio-Aware Decoding (AAD). Unlike data augmentation, these are intentionally
aggressive and destructive.
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from typing import Callable, Dict, Iterator, Optional, Tuple


# =============================================================================
# PERTURBATION CLASSES
#
# Setting keys use descriptive severity labels (e.g. "heavy", "extreme").
# Each setting line shows: paper display string, then → proposed literal key
# that would replace the current key in a future rename pass.
# Full rename table: observations/proposed_setting_renames.md
# =============================================================================

class NoisePerturbation:
    name = "NOISE"
    settings = {
        "heavy":        {"sigma": 0.3},  # σ=0.3       →  "sigma_0.3"
        "very_heavy":   {"sigma": 0.5},  # σ=0.5       →  "sigma_0.5"
        "extreme":      {"sigma": 0.6},  # σ=0.6       →  "sigma_0.6"
        "overwhelming": {"sigma": 1.0},  # σ=1.0       →  "sigma_1.0"
    }

    @staticmethod
    def apply(audio: np.ndarray, sigma: float = 0.3, sr: int = 16000) -> np.ndarray:
        noise = np.random.randn(*audio.shape).astype(np.float32) * sigma
        return audio + noise


class ColoredNoisePerturbation:
    name = "COLORED_NOISE"
    settings = {
        "pink_heavy":  {"color": "pink",  "sigma": 0.4},  # pink σ=0.4   →  "pink_0.4"
        "brown_heavy": {"color": "brown", "sigma": 0.5},  # brown σ=0.5  →  "brown_0.5"
    }

    @staticmethod
    def apply(audio: np.ndarray, color: str = "pink", sigma: float = 0.4, sr: int = 16000) -> np.ndarray:
        n = len(audio)
        white = np.random.randn(n).astype(np.float32)
        if color == "pink":
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n)
            freqs[0] = 1
            fft = fft / np.sqrt(freqs)
            noise = np.fft.irfft(fft, n).astype(np.float32)
        elif color == "brown":
            noise = np.cumsum(white).astype(np.float32)
            noise = noise - np.mean(noise)
        else:
            noise = white
        noise = noise / (np.std(noise) + 1e-8) * sigma
        return audio + noise


class TimestretchPerturbation:
    name = "TIMESTRETCH"
    settings = {
        "very_slow": {"rate": 0.4},  # 0.4×  →  "rate_0.4x"
        "very_fast": {"rate": 2.5},  # 2.5×  →  "rate_2.5x"
    }

    @staticmethod
    def apply(audio: np.ndarray, rate: float = 0.5, sr: int = 16000) -> np.ndarray:
        return librosa.effects.time_stretch(audio, rate=rate)


class SegmentShufflePerturbation:
    name = "SEGMENT_SHUFFLE"
    settings = {
        "coarse":  {"n_segments": 10},   # 10 seg   →  "10seg"
        "fine":    {"n_segments": 50},   # 50 seg   →  "50seg"
        "extreme": {"n_segments": 200},  # 200 seg  →  "200seg"
    }

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 50, sr: int = 16000) -> np.ndarray:
        segment_len = len(audio) // n_segments
        if segment_len < 1:
            return audio
        segments = [audio[i*segment_len:(i+1)*segment_len] for i in range(n_segments)]
        remainder = audio[n_segments*segment_len:]
        np.random.shuffle(segments)
        result = np.concatenate(segments)
        if len(remainder) > 0:
            result = np.concatenate([result, remainder])
        return result


class SegmentReversePerturbation:
    name = "SEGMENT_REVERSE"
    settings = {
        "coarse": {"n_segments": 10},  # 10 seg  →  "10seg"
        "fine":   {"n_segments": 50},  # 50 seg  →  "50seg"
    }

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 10, sr: int = 16000) -> np.ndarray:
        segment_len = len(audio) // n_segments
        if segment_len < 1:
            return audio
        result = audio.copy()
        for i in range(n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len
            result[start:end] = result[start:end][::-1]
        return result


class ReversePerturbation:
    name = "REVERSE"
    settings = {
        "full": {},  # no parameters — key unchanged
    }

    @staticmethod
    def apply(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        return audio[::-1].copy()


class DropoutPerturbation:
    name = "DROPOUT"
    settings = {
        "heavy":   {"p": 0.4},  # p=0.4  →  "p_0.4"
        "extreme": {"p": 0.7},  # p=0.7  →  "p_0.7"
    }

    @staticmethod
    def apply(audio: np.ndarray, p: float = 0.5, sr: int = 16000) -> np.ndarray:
        mask = np.random.random(len(audio)) > p
        result = audio.copy()
        result[~mask] = 0
        return result


class TimeMaskPerturbation:
    name = "TIME_MASK"
    settings = {
        "light":   {"n_masks": 3,  "max_width": 0.08},  # 3 masks 8%   →  "3mask_8pct"
        "heavy":   {"n_masks": 5,  "max_width": 0.15},  # 5 masks 15%  →  "5mask_15pct"
        "extreme": {"n_masks": 10, "max_width": 0.10},  # 10 masks 10% →  "10mask_10pct"
    }

    @staticmethod
    def apply(audio: np.ndarray, n_masks: int = 5, max_width: float = 0.15, sr: int = 16000) -> np.ndarray:
        result = audio.copy()
        audio_len = len(audio)
        for _ in range(n_masks):
            width = int(np.random.uniform(0.05, max_width) * audio_len)
            start = np.random.randint(0, max(1, audio_len - width))
            result[start:start+width] = 0
        return result


class RepeatSegmentPerturbation:
    name = "REPEAT_SEGMENT"
    settings = {
        "repeat_start":  {"segment": "start",  "repeats": 5},  # no rename needed
        "repeat_middle": {"segment": "middle", "repeats": 5},  # no rename needed
    }

    @staticmethod
    def apply(audio: np.ndarray, segment: str = "start", repeats: int = 5, sr: int = 16000) -> np.ndarray:
        segment_len = len(audio) // 5
        if segment == "start":
            seg = audio[:segment_len]
        elif segment == "middle":
            mid = len(audio) // 2
            seg = audio[mid - segment_len//2:mid + segment_len//2]
        else:
            seg = audio[-segment_len:]
        return np.tile(seg, repeats)[:len(audio)]


class LowPassPerturbation:
    name = "LOW_PASS"
    settings = {
        "extreme":    {"cutoff_hz": 250},   # 250 Hz  →  "250hz"
        "aggressive": {"cutoff_hz": 500},   # 500 Hz  →  "500hz"
        "moderate":   {"cutoff_hz": 1000},  # 1 kHz   →  "1khz"
    }

    @staticmethod
    def apply(audio: np.ndarray, cutoff_hz: float = 500, sr: int = 16000) -> np.ndarray:
        nyq = sr / 2
        normalized_cutoff = min(cutoff_hz / nyq, 0.99)
        b, a = signal.butter(5, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio).astype(np.float32)


class HighPassPerturbation:
    name = "HIGH_PASS"
    settings = {
        "moderate":      {"cutoff_hz": 1000},  # 1 kHz  →  "1khz"
        "aggressive":    {"cutoff_hz": 2000},  # 2 kHz  →  "2khz"
        "extreme":       {"cutoff_hz": 4000},  # 4 kHz  →  "4khz"
        "very_extreme":  {"cutoff_hz": 5000},  # 5 kHz  →  "5khz"
        "ultra_extreme": {"cutoff_hz": 6000},  # 6 kHz  →  "6khz"
    }

    @staticmethod
    def apply(audio: np.ndarray, cutoff_hz: float = 2000, sr: int = 16000) -> np.ndarray:
        nyq = sr / 2
        normalized_cutoff = min(cutoff_hz / nyq, 0.99)
        b, a = signal.butter(5, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio).astype(np.float32)


class BandpassPerturbation:
    name = "BANDPASS"
    settings = {
        "bass_only":       {"low_hz": 50,   "high_hz": 300},   # 50–300 Hz        →  "50_300hz"
        "bass_wide":       {"low_hz": 50,   "high_hz": 500},   # 50–500 Hz        →  "50_500hz"
        "low_mid":         {"low_hz": 200,  "high_hz": 800},   # 200–800 Hz       →  "200_800hz"
        "mid_narrow":      {"low_hz": 700,  "high_hz": 1400},  # 700 Hz–1.4 kHz   →  "700hz_1.4khz"
        "mid_only":        {"low_hz": 500,  "high_hz": 2000},  # 500 Hz–2 kHz     →  "500hz_2khz"
        "high_mid":        {"low_hz": 1500, "high_hz": 4000},  # 1.5–4 kHz        →  "1.5khz_4khz"
        "high_mid_narrow": {"low_hz": 2000, "high_hz": 3500},  # 2–3.5 kHz        →  "2khz_3.5khz"
        "treble_only":     {"low_hz": 3000, "high_hz": 8000},  # 3–8 kHz          →  "3khz_8khz"
        "treble_extreme":  {"low_hz": 4000, "high_hz": 8000},  # 4–8 kHz          →  "4khz_8khz"
        "treble_ultra":    {"low_hz": 5000, "high_hz": 8000},  # 5–8 kHz          →  "5khz_8khz"
    }

    @staticmethod
    def apply(audio: np.ndarray, low_hz: float = 500, high_hz: float = 2000, sr: int = 16000) -> np.ndarray:
        nyq = sr / 2
        low = min(low_hz / nyq, 0.99)
        high = min(high_hz / nyq, 0.99)
        if low >= high:
            low = high - 0.01
        b, a = signal.butter(5, [low, high], btype='band')
        return signal.filtfilt(b, a, audio).astype(np.float32)


class BandstopPerturbation:
    name = "BANDSTOP"
    settings = {
        "remove_low":  {"low_hz": 50,   "high_hz": 500},   # 50–500 Hz     →  "50_500hz"
        "remove_mids": {"low_hz": 500,  "high_hz": 2000},  # 500 Hz–2 kHz  →  "500hz_2khz"
        "remove_high": {"low_hz": 3000, "high_hz": 8000},  # 3–8 kHz       →  "3khz_8khz"
    }

    @staticmethod
    def apply(audio: np.ndarray, low_hz: float = 500, high_hz: float = 2000, sr: int = 16000) -> np.ndarray:
        nyq = sr / 2
        low = min(low_hz / nyq, 0.99)
        high = min(high_hz / nyq, 0.99)
        if low >= high:
            low = high - 0.01
        b, a = signal.butter(5, [low, high], btype='bandstop')
        return signal.filtfilt(b, a, audio).astype(np.float32)


class FreqMaskPerturbation:
    name = "FREQ_MASK"
    settings = {
        "heavy":      {"n_masks": 8,  "max_width_hz": 500},  # 8 masks ≤500 Hz   →  "8mask_500hz"
        "very_heavy": {"n_masks": 12, "max_width_hz": 600},  # 12 masks ≤600 Hz  →  "12mask_600hz"
        "extreme":    {"n_masks": 15, "max_width_hz": 400},  # 15 masks ≤400 Hz  →  "15mask_400hz"
    }

    @staticmethod
    def apply(audio: np.ndarray, n_masks: int = 8, max_width_hz: float = 500, sr: int = 16000) -> np.ndarray:
        stft = librosa.stft(audio)
        n_bins = stft.shape[0]
        hz_per_bin = (sr / 2) / n_bins
        for _ in range(n_masks):
            width_bins = int(np.random.uniform(100, max_width_hz) / hz_per_bin)
            start = np.random.randint(0, max(1, n_bins - width_bins))
            stft[start:start+width_bins, :] = 0
        return librosa.istft(stft, length=len(audio)).astype(np.float32)


class PitchShiftPerturbation:
    name = "PITCH_SHIFT"
    settings = {
        "extreme_down":  {"semitones": -24},  # down two octaves  →  "down_2oct"
        "down_octave":   {"semitones": -12},  # down octave       →  "down_1oct"
        "down_moderate": {"semitones": -6},   # −6 st             →  "minus_6st"
        "down_mild":     {"semitones": -4},   # −4 st             →  "minus_4st"
        "up_mild":       {"semitones": 4},    # +4 st             →  "plus_4st"
        "up_moderate":   {"semitones": 6},    # +6 st             →  "plus_6st"
        "up_octave":     {"semitones": 12},   # up octave         →  "up_1oct"
        "extreme_up":    {"semitones": 24},   # up two octaves    →  "up_2oct"
    }

    @staticmethod
    def apply(audio: np.ndarray, semitones: int = 12, sr: int = 16000) -> np.ndarray:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


class SpecNoisePerturbation:
    name = "SPEC_NOISE"
    settings = {
        "light":        {"sigma": 0.1},  # σ=0.1  →  "sigma_0.1"
        "heavy":        {"sigma": 0.3},  # σ=0.3  →  "sigma_0.3"
        "extreme":      {"sigma": 0.6},  # σ=0.6  →  "sigma_0.6"
        "overwhelming": {"sigma": 1.0},  # σ=1.0  →  "sigma_1.0"
    }

    @staticmethod
    def apply(audio: np.ndarray, sigma: float = 0.3, sr: int = 16000) -> np.ndarray:
        n_fft, hop_length = 1024, 256
        stft = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        rms = float(np.sqrt(np.mean(magnitude ** 2))) + 1e-8
        noise = np.random.randn(*magnitude.shape).astype(np.float32) * sigma * rms
        magnitude_noisy = np.maximum(magnitude + noise, 0.0)
        return librosa.istft(magnitude_noisy * phase, hop_length=hop_length, length=len(audio)).astype(np.float32)


class SpectralBlurPerturbation:
    name = "SPECTRAL_BLUR"
    settings = {
        "light":   {"sigma_time": 5,  "sigma_freq": 5},   # σ=5   →  "sigma_5"
        "heavy":   {"sigma_time": 15, "sigma_freq": 15},  # σ=15  →  "sigma_15"
        "extreme": {"sigma_time": 25, "sigma_freq": 25},  # σ=25  →  "sigma_25"
    }

    @staticmethod
    def apply(audio: np.ndarray, sigma_time: float = 10, sigma_freq: float = 10, sr: int = 16000) -> np.ndarray:
        stft = librosa.stft(audio)
        mag = np.abs(stft)
        phase = np.angle(stft)
        mag_blurred = gaussian_filter1d(mag, sigma=sigma_freq, axis=0)
        mag_blurred = gaussian_filter1d(mag_blurred, sigma=sigma_time, axis=1)
        return librosa.istft(mag_blurred * np.exp(1j * phase), length=len(audio)).astype(np.float32)


class SpecReversePerturbation:
    name = "SPEC_REVERSE"
    settings = {
        "full": {},  # no parameters — key unchanged
    }

    @staticmethod
    def apply(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        stft = librosa.stft(audio.astype(np.float32), n_fft=1024, hop_length=256)
        return librosa.istft(stft[:, ::-1], hop_length=256, length=len(audio)).astype(np.float32)


class SpecSegmentReversePerturbation:
    name = "SPEC_SEGMENT_REVERSE"
    settings = {
        "coarse": {"n_segments": 10},  # 10 seg  →  "10seg"
        "fine":   {"n_segments": 50},  # 50 seg  →  "50seg"
    }

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 10, sr: int = 16000) -> np.ndarray:
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


class SpecSegmentShufflePerturbation:
    name = "SPEC_SEGMENT_SHUFFLE"
    settings = {
        "coarse":  {"n_segments": 10},   # 10 seg   →  "10seg"
        "fine":    {"n_segments": 50},   # 50 seg   →  "50seg"
        "extreme": {"n_segments": 200},  # 200 seg  →  "200seg"
    }

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 50, sr: int = 16000) -> np.ndarray:
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


class HarmonicRemovePerturbation:
    name = "HARMONIC_REMOVE"
    settings = {
        "full": {"margin": 3.0},  # no rename needed
    }

    @staticmethod
    def apply(audio: np.ndarray, margin: float = 3.0, sr: int = 16000) -> np.ndarray:
        _, percussive = librosa.effects.hpss(audio, margin=margin)
        return percussive


class PercussiveRemovePerturbation:
    name = "PERCUSSIVE_REMOVE"
    settings = {
        "full": {"margin": 3.0},  # no rename needed
    }

    @staticmethod
    def apply(audio: np.ndarray, margin: float = 3.0, sr: int = 16000) -> np.ndarray:
        harmonic, _ = librosa.effects.hpss(audio, margin=margin)
        return harmonic


class ClipPerturbation:
    name = "CLIP"
    settings = {
        "extreme": {"threshold": 0.1},  # thr=0.1  →  "thr_0.1"
        "hard":    {"threshold": 0.2},  # thr=0.2  →  "thr_0.2"
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.2, sr: int = 16000) -> np.ndarray:
        return np.clip(audio, -threshold, threshold)


class QuantizePerturbation:
    name = "QUANTIZE"
    settings = {
        "4bit": {"bits": 4},  # already literal
        "3bit": {"bits": 3},  # already literal
        "2bit": {"bits": 2},  # already literal
    }

    @staticmethod
    def apply(audio: np.ndarray, bits: int = 3, sr: int = 16000) -> np.ndarray:
        levels = 2 ** bits
        audio_norm = (audio - audio.min()) / (audio.max() - audio.min() + 1e-8)
        quantized = np.round(audio_norm * (levels - 1)) / (levels - 1)
        return (quantized * (audio.max() - audio.min()) + audio.min()).astype(np.float32)


class CompressPerturbation:
    name = "COMPRESS"
    settings = {
        "heavy":   {"threshold": 0.2, "ratio": 10},  # thr=0.2 10:1  →  "thr0.2_r10"
        "extreme": {"threshold": 0.1, "ratio": 20},  # thr=0.1 20:1  →  "thr0.1_r20"
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.2, ratio: float = 10, sr: int = 16000) -> np.ndarray:
        result = audio.copy()
        above_thresh = np.abs(audio) > threshold
        excess = np.abs(audio[above_thresh]) - threshold
        result[above_thresh] = np.sign(audio[above_thresh]) * (threshold + excess / ratio)
        return result


class GatePerturbation:
    name = "GATE"
    settings = {
        "aggressive":    {"threshold": 0.30},  # thr=0.30  →  "thr_0.30"
        "extreme":       {"threshold": 0.50},  # thr=0.50  →  "thr_0.50"
        "very_extreme":  {"threshold": 0.65},  # thr=0.65  →  "thr_0.65"
        "ultra_extreme": {"threshold": 0.75},  # thr=0.75  →  "thr_0.75"
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, sr: int = 16000) -> np.ndarray:
        result = audio.copy()
        result[np.abs(audio) < threshold] = 0
        return result


class GateInvertedPerturbation:
    name = "GATE_INVERTED"
    settings = {
        "moderate":     {"threshold": 0.50},  # thr=0.50  →  "thr_0.50"
        "aggressive":   {"threshold": 0.30},  # thr=0.30  →  "thr_0.30"
        "extreme":      {"threshold": 0.20},  # thr=0.20  →  "thr_0.20"
        "very_extreme": {"threshold": 0.10},  # thr=0.10  →  "thr_0.10"
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, sr: int = 16000) -> np.ndarray:
        result = audio.copy()
        result[np.abs(audio) >= threshold] = 0
        return result


class GateSoftPerturbation:
    name = "GATE_SOFT"
    settings = {
        "soft_aggressive":   {"threshold": 0.3, "attenuation_db": -18},  # thr=0.3 −18 dB  →  "thr0.3_neg18db"
        "soft_extreme":      {"threshold": 0.3, "attenuation_db": -30},  # thr=0.3 −30 dB  →  "thr0.3_neg30db"
        "soft_very_extreme": {"threshold": 0.5, "attenuation_db": -24},  # thr=0.5 −24 dB  →  "thr0.5_neg24db"
        "soft_ultra":        {"threshold": 0.5, "attenuation_db": -45},  # thr=0.5 −45 dB  →  "thr0.5_neg45db"
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, attenuation_db: float = -24, sr: int = 16000) -> np.ndarray:
        result = audio.copy()
        gain = 10 ** (attenuation_db / 20)
        result[np.abs(audio) < threshold] *= gain
        return result


class GateInvertedSoftPerturbation:
    name = "GATE_INVERTED_SOFT"
    settings = {
        "soft_moderate":   {"threshold": 0.5, "attenuation_db": -12},  # thr=0.5 −12 dB  →  "thr0.5_neg12db"
        "soft_aggressive": {"threshold": 0.3, "attenuation_db": -18},  # thr=0.3 −18 dB  →  "thr0.3_neg18db"
        "soft_extreme":    {"threshold": 0.2, "attenuation_db": -30},  # thr=0.2 −30 dB  →  "thr0.2_neg30db"
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, attenuation_db: float = -18, sr: int = 16000) -> np.ndarray:
        result = audio.copy()
        gain = 10 ** (attenuation_db / 20)
        result[np.abs(audio) >= threshold] *= gain
        return result


class NormalizeChunksPerturbation:
    name = "NORMALIZE_CHUNKS"
    settings = {
        "coarse": {"n_chunks": 10},  # 10 chunks  →  "10chunks"
        "fine":   {"n_chunks": 50},  # 50 chunks  →  "50chunks"
    }

    @staticmethod
    def apply(audio: np.ndarray, n_chunks: int = 20, sr: int = 16000) -> np.ndarray:
        chunk_len = len(audio) // n_chunks
        if chunk_len < 1:
            return audio
        result = audio.copy()
        for i in range(n_chunks):
            start = i * chunk_len
            end = (i + 1) * chunk_len
            chunk = result[start:end]
            result[start:end] = chunk / (np.max(np.abs(chunk)) + 1e-8)
        return result


class ResampleLowPerturbation:
    name = "RESAMPLE_LOW"
    settings = {
        "8khz": {"target_sr": 8000},  # already literal
        "4khz": {"target_sr": 4000},  # already literal
        "2khz": {"target_sr": 2000},  # already literal
    }

    @staticmethod
    def apply(audio: np.ndarray, target_sr: int = 4000, sr: int = 16000) -> np.ndarray:
        downsampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        upsampled = librosa.resample(downsampled, orig_sr=target_sr, target_sr=sr)
        if len(upsampled) > len(audio):
            upsampled = upsampled[:len(audio)]
        elif len(upsampled) < len(audio):
            upsampled = np.pad(upsampled, (0, len(audio) - len(upsampled)))
        return upsampled


class BitCrushPerturbation:
    name = "BIT_CRUSH"
    settings = {
        "retro":   {"bits": 4, "target_sr": 8000},  # 4-bit 8 kHz  →  "4bit_8khz"
        "extreme": {"bits": 2, "target_sr": 4000},  # 2-bit 4 kHz  →  "2bit_4khz"
    }

    @staticmethod
    def apply(audio: np.ndarray, bits: int = 3, target_sr: int = 4000, sr: int = 16000) -> np.ndarray:
        resampled = ResampleLowPerturbation.apply(audio, target_sr=target_sr, sr=sr)
        return QuantizePerturbation.apply(resampled, bits=bits, sr=sr)


class ReverbPerturbation:
    name = "REVERB"
    settings = {
        "large_hall": {"decay": 0.80, "delay": 0.05},  # decay=0.80 50 ms   →  "d0.80_50ms"
        "extreme":    {"decay": 0.95, "delay": 0.10},  # decay=0.95 100 ms  →  "d0.95_100ms"
    }

    @staticmethod
    def apply(audio: np.ndarray, decay: float = 0.8, delay: float = 0.05, sr: int = 16000) -> np.ndarray:
        delay_samples = int(delay * sr)
        result = audio.copy()
        for i in range(5):
            d = delay_samples * (i + 1)
            gain = decay ** (i + 1)
            if d < len(audio):
                result[d:] += audio[:-d] * gain
        result = result / (np.max(np.abs(result)) + 1e-8) * np.max(np.abs(audio))
        return result.astype(np.float32)


class EchoPerturbation:
    name = "ECHO"
    settings = {
        "short": {"delay_ms": 100, "decay": 0.6, "repeats": 5},  # 100 ms 0.6 ×5  →  "100ms_0.6_5x"
        "long":  {"delay_ms": 300, "decay": 0.7, "repeats": 8},  # 300 ms 0.7 ×8  →  "300ms_0.7_8x"
    }

    @staticmethod
    def apply(audio: np.ndarray, delay_ms: float = 200, decay: float = 0.6, repeats: int = 5, sr: int = 16000) -> np.ndarray:
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


class PhoneFilterPerturbation:
    name = "PHONE_FILTER"
    settings = {
        "standard": {"low_hz": 300, "high_hz": 3400},  # 300 Hz–3.4 kHz  →  "300hz_3.4khz"
        "narrow":   {"low_hz": 500, "high_hz": 2500},  # 500 Hz–2.5 kHz  →  "500hz_2.5khz"
    }

    @staticmethod
    def apply(audio: np.ndarray, low_hz: float = 300, high_hz: float = 3400, sr: int = 16000) -> np.ndarray:
        return BandpassPerturbation.apply(audio, low_hz=low_hz, high_hz=high_hz, sr=sr)


class UnderwaterPerturbation:
    name = "UNDERWATER"
    settings = {
        "deep": {"cutoff_hz": 400, "reverb_decay": 0.7},  # 400 Hz decay=0.7  →  "400hz_d0.7"
    }

    @staticmethod
    def apply(audio: np.ndarray, cutoff_hz: float = 400, reverb_decay: float = 0.7, sr: int = 16000) -> np.ndarray:
        filtered = LowPassPerturbation.apply(audio, cutoff_hz=cutoff_hz, sr=sr)
        return ReverbPerturbation.apply(filtered, decay=reverb_decay, delay=0.03, sr=sr)


# =============================================================================
# REGISTRY & PUBLIC API
# =============================================================================

ALL_CLASSES = [
    NoisePerturbation,
    ColoredNoisePerturbation,
    TimestretchPerturbation,
    SegmentShufflePerturbation,
    SegmentReversePerturbation,
    ReversePerturbation,
    DropoutPerturbation,
    TimeMaskPerturbation,
    RepeatSegmentPerturbation,
    LowPassPerturbation,
    HighPassPerturbation,
    BandpassPerturbation,
    BandstopPerturbation,
    FreqMaskPerturbation,
    PitchShiftPerturbation,
    SpecNoisePerturbation,
    SpectralBlurPerturbation,
    SpecReversePerturbation,
    SpecSegmentReversePerturbation,
    SpecSegmentShufflePerturbation,
    HarmonicRemovePerturbation,
    PercussiveRemovePerturbation,
    ClipPerturbation,
    QuantizePerturbation,
    CompressPerturbation,
    GatePerturbation,
    GateInvertedPerturbation,
    GateSoftPerturbation,
    GateInvertedSoftPerturbation,
    NormalizeChunksPerturbation,
    ResampleLowPerturbation,
    BitCrushPerturbation,
    ReverbPerturbation,
    EchoPerturbation,
    PhoneFilterPerturbation,
    UnderwaterPerturbation,
]

# Backward-compatible alias used by the runners
PERTURBATION_SPECS: Dict[type, Dict] = {
    cls: cls.settings for cls in ALL_CLASSES
}

BASELINE_PERTURBATIONS: Tuple[str, ...] = ("ORIGINAL", "NO_AUDIO")
PERTURBATION_BY_NAME: Dict[str, type] = {
    cls.name: cls for cls in ALL_CLASSES
}


def iter_perturbation_configs(include_baselines: bool = True) -> Iterator[Tuple[str, Optional[str]]]:
    if include_baselines:
        for name in BASELINE_PERTURBATIONS:
            yield name, None

    for cls in ALL_CLASSES:
        for setting in cls.settings:
            yield cls.name, setting


def get_perturbation_class(name: str) -> type:
    normalized = name.upper()
    try:
        return PERTURBATION_BY_NAME[normalized]
    except KeyError:
        valid = ", ".join(BASELINE_PERTURBATIONS + tuple(PERTURBATION_BY_NAME))
        raise KeyError(f"Unknown perturbation type {name!r}; expected one of: {valid}") from None


def get_perturbation(cls: type, setting: Optional[str] = None, sr: int = 16000) -> Callable:
    params = cls.settings[setting].copy() if setting is not None else {}
    return lambda audio: cls.apply(audio, sr=sr, **params)


# =============================================================================
# TESTING (only runs when executed directly)
# =============================================================================

if __name__ == "__main__":
    import os
    import random
    import soundfile as sf

    clotho_audio_dir = "datasets/clotho_aqa/audio_files"
    output_base = "debug/perturbation_samples"
    sr = 16000

    all_files = [f for f in os.listdir(clotho_audio_dir) if f.lower().endswith(".wav")]
    sample_file = random.choice(all_files)
    sample_name = os.path.splitext(sample_file)[0]

    audio, _ = librosa.load(os.path.join(clotho_audio_dir, sample_file), sr=sr)

    sample_folder = os.path.join(output_base, sample_name)
    os.makedirs(sample_folder, exist_ok=True)
    sf.write(os.path.join(sample_folder, f"{sample_name}_ORIGINAL.wav"), audio, sr)

    for cls in ALL_CLASSES:
        for setting in cls.settings:
            fn = get_perturbation(cls, setting, sr=sr)
            pert_audio = fn(audio)
            label = f"{cls.name}_{setting}"
            out_path = os.path.join(sample_folder, f"{sample_name}_{label}.wav")
            sf.write(out_path, pert_audio, sr)
            print(f"Saved {label} -> {out_path}")

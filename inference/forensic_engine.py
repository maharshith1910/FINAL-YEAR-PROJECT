"""
inference/forensic_engine.py
----------------------------
Smart Crime Detection & Acoustic Outlier Forensic Engine.

Differentiates genuine human distress screams from real-world false alarms:
  - Vehicle horns / Bus honks (dual-tone stationary harmonics)
  - Sirens (periodic sinusoidal pitch sweeps)
  - Tire screeches / Industrial noise (high-frequency friction, high spectral flatness)
  - Children playing / Non-distress vocalizations

Integrates:
  1. Fast Bio-Acoustic Feature Extractor (Pitch F0 stability, HNR, Spectral Flux, Chaos Index)
  2. Gemini Multimodal / Context-Aware Threat Verification Agent
  3. Resilient Offline Rule-based Heuristic Fallback Engine
"""

import os
import json
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass, asdict

# Try importing Gemini SDK
GEMINI_CLIENT = None
try:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        GEMINI_CLIENT = genai.Client(api_key=api_key)
except Exception:
    GEMINI_CLIENT = None


@dataclass
class BioacousticFingerprint:
    duration_sec: float
    f0_mean_hz: float
    f0_modulation_pct: float
    hnr_db: float
    spectral_centroid_hz: float
    spectral_flatness: float
    chaos_index: float
    candidate_source: str
    is_harmonic: bool


@dataclass
class ForensicReport:
    classification: str         # "GENUINE_HUMAN_SCREAM", "VEHICLE_HORN", "TIRE_SCREECH", "ENVIRONMENTAL_NOISE"
    is_crime_threat: bool
    threat_confidence: float     # 0.0 to 1.0
    outlier_category: str       # "HORN_OUTLIER", "SCREECH_OUTLIER", "VOICE_DISTRESS", "CLEAR"
    forensic_rationale: str
    recommended_action: str     # "DISPATCH_PATROL", "MONITOR_ZONE", "DISMISS_FALSE_ALARM"
    fingerprint: dict


class AcousticForensicEngine:
    """
    Analyzes audio waveforms to extract bio-acoustic signatures and queries
    Gemini / Contextual LLM to disambiguate human screams from vehicle horns,
    bus honks, and sirens.
    """

    def __init__(self, gemini_api_key: str | None = None):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def extract_bioacoustics(self, y: np.ndarray, sr: int = 22050) -> BioacousticFingerprint:
        """
        Fast sub-100ms bio-acoustic feature extraction separating biological screams
        from electromechanical vehicle horns and sirens.
        """
        dur = len(y) / sr
        if len(y) < sr * 0.2:
            return BioacousticFingerprint(
                duration_sec=dur, f0_mean_hz=0.0, f0_modulation_pct=0.0,
                hnr_db=0.0, spectral_centroid_hz=0.0, spectral_flatness=0.0,
                chaos_index=0.0, candidate_source="SHORT_SAMPLE", is_harmonic=False
            )

        # Focus analysis on the peak energy segment (1.5 seconds) for sub-50ms speed
        hop = 512
        frame_len = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
        peak_idx = int(np.argmax(rms)) * hop
        start_idx = max(0, peak_idx - int(0.5 * sr))
        end_idx   = min(len(y), start_idx + int(1.5 * sr))
        y_seg     = y[start_idx:end_idx]

        # 1. Fast Fundamental Frequency (F0) estimation via downsampled autocorrelation
        try:
            # Downsample segment by 2x for ultra-fast autocorrelation
            y_sub = librosa.resample(y_seg, orig_sr=sr, target_sr=sr // 2)
            sr_sub = sr // 2
            autocorr = np.correlate(y_sub, y_sub, mode='full')[len(y_sub)-1:]
            
            # Lag corresponding to 150 Hz to 1800 Hz
            min_lag = int(sr_sub / 1800)
            max_lag = int(sr_sub / 150)
            
            if len(autocorr) > max_lag:
                peak_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
                f0_est = float(sr_sub / max(1, peak_lag))
            else:
                f0_est = 450.0

            # Pitch variation / jitter estimate via zero-crossing rate variance
            zcr = librosa.feature.zero_crossing_rate(y_seg, hop_length=256)[0]
            f0_mod = float(np.std(zcr) / (np.mean(zcr) + 1e-6) * 40.0)
        except Exception:
            f0_est = 450.0
            f0_mod = 10.0

        # 2. Fast Harmonicity & Spectral Metrics
        cent = float(np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr)))
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y_seg)))
        
        # Energy stationarity (horns are sustained and flat, screams have chaotic attack/decay)
        rms_seg = librosa.feature.rms(y=y_seg, hop_length=256)[0]
        rms_stationarity = float(np.std(rms_seg) / (np.mean(rms_seg) + 1e-6))

        # Scientific Harmonic-to-Noise Ratio (HNR in dB) via HPSS decomposition
        try:
            y_harm, y_perc = librosa.effects.hpss(y_seg)
            e_harm = float(np.sum(y_harm ** 2))
            e_perc = float(np.sum(y_perc ** 2)) + 1e-9
            hnr_est = float(np.clip(10.0 * np.log10(e_harm / e_perc), -10.0, 48.0))
        except Exception:
            hnr_est = float(np.clip(-10.0 * np.log10(max(1e-4, flat)), 2.0, 30.0))

        # 3. Chaos & Vocal Fold Turbulence Index
        chaos_index = float(np.clip((f0_mod / 25.0) * (rms_stationarity * 1.5) * (1.0 if hnr_est < 12.0 else 0.2), 0.0, 1.0))

        # Acoustic pre-classification
        is_harmonic = (hnr_est >= 18.0 and f0_mod < 7.0 and rms_stationarity < 0.35)
        if is_harmonic or (hnr_est >= 22.0 and f0_mod < 8.0):
            candidate = "VEHICLE_HORN_OR_SIREN"
        elif flat > 0.18:
            candidate = "FRICTION_TIRE_SCREECH"
        elif cnn_prob_hint := (cent >= 1400.0 or f0_mod >= 9.0 or chaos_index >= 0.30):
            candidate = "HUMAN_VOCAL_DISTRESS"
        else:
            candidate = "AMBIENT_NOISE"

        return BioacousticFingerprint(
            duration_sec=round(dur, 2),
            f0_mean_hz=round(f0_est, 1),
            f0_modulation_pct=round(f0_mod, 1),
            hnr_db=round(hnr_est, 1),
            spectral_centroid_hz=round(cent, 1),
            spectral_flatness=round(flat, 4),
            chaos_index=round(chaos_index, 2),
            candidate_source=candidate,
            is_harmonic=bool(is_harmonic),
        )

    def analyze(
        self,
        audio_path: str | Path,
        cnn_prob: float,
        location: str = "Surveillance Node 01",
    ) -> ForensicReport:
        """
        Runs comprehensive forensic threat analysis, querying Gemini with
        acoustic context or falling back to the bio-acoustic expert engine.
        """
        audio_path = Path(audio_path)
        try:
            y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        except Exception as e:
            return ForensicReport(
                classification="UNKNOWN_AUDIO_ERROR",
                is_crime_threat=False,
                threat_confidence=0.0,
                outlier_category="ERROR",
                forensic_rationale=f"Could not read audio file: {e}",
                recommended_action="INSPECT_SENSOR",
                fingerprint={},
            )

        fp = self.extract_bioacoustics(y, sr)

        # ── Fast check: if CNN probability is low (<0.35) and no loud anomalous harmonics, it's normal ambient
        if cnn_prob < 0.35 and not fp.is_harmonic and fp.candidate_source == "AMBIENT_NOISE":
            return self._local_expert_reasoning(fp, cnn_prob, location)

        # ── Try Gemini Context-Aware Agentic Verification if client is active
        if self.client:
            report = self._call_gemini(fp, cnn_prob, location)
            if report:
                return report

        # ── Fast Local Bio-Acoustic Expert System (100% offline fallback)
        return self._local_expert_reasoning(fp, cnn_prob, location)

    def _call_gemini(self, fp: BioacousticFingerprint, cnn_prob: float, location: str) -> ForensicReport | None:
        """Queries Gemini LLM for expert audio forensic evaluation."""
        prompt = f"""
You are an expert Audio Forensic Intelligence AI for SafeWatch AI (Smart Crime Detection & Monitoring System).
A surveillance acoustic node just flagged an acoustic event at: {location}.

Deep Learning CNN Model Output:
- Peak P(distress_scream) = {cnn_prob:.4f}

Acoustic Physics & Bio-Acoustic Fingerprint:
- Duration: {fp.duration_sec}s
- Estimated Fundamental Frequency (F0): {fp.f0_mean_hz} Hz
- Pitch Modulation (Jitter / ZCR variance): {fp.f0_modulation_pct}%
- Harmonic-to-Noise Ratio (HNR): {fp.hnr_db} dB (High >18dB = pure tone machine; Low <10dB = turbulent air/scream)
- Spectral Centroid: {fp.spectral_centroid_hz} Hz
- Spectral Flatness: {fp.spectral_flatness}
- Chaos Index: {fp.chaos_index} (0.0 = stationary tone, >0.35 = chaotic vocal fold rupture)
- Candidate Pre-classification: {fp.candidate_source}

FORENSIC DECISION CRITERIA:
1. VEHICLE HORN / BUS HONK: If HNR is high (>= 18 dB), pitch modulation is low (< 7%), and tone is stationary, this is an ELECTROMECHANICAL VEHICLE HORN / BUS HONK outlier, NOT a human scream! Even if the CNN gave a high score due to high amplitude, you MUST filter it out as an outlier: classification="VEHICLE_HORN", is_crime_threat=false, outlier_category="HORN_OUTLIER", recommended_action="DISMISS_FALSE_ALARM".
2. TIRE SCREECH / FRICTION: High spectral flatness (>0.15) with no vocal formants: classification="TIRE_SCREECH", is_crime_threat=false, outlier_category="SCREECH_OUTLIER", recommended_action="DISMISS_FALSE_ALARM".
3. GENUINE HUMAN DISTRESS SCREAM: High CNN probability (>= 0.50), high spectral centroid (>1400 Hz), low HNR (<12 dB), and chaotic pitch modulation (>8%): classification="GENUINE_HUMAN_SCREAM", is_crime_threat=true, outlier_category="VOICE_DISTRESS", recommended_action="DISPATCH_PATROL".
4. ENVIRONMENTAL AMBIENT: Low CNN probability (<0.35) and low threat: classification="ENVIRONMENTAL_AMBIENT", is_crime_threat=false, outlier_category="CLEAR", recommended_action="MONITOR_ZONE".

Return your forensic analysis STRICTLY as valid JSON matching this schema:
{{
  "classification": "GENUINE_HUMAN_SCREAM" | "VEHICLE_HORN" | "TIRE_SCREECH" | "SIREN" | "ENVIRONMENTAL_AMBIENT",
  "is_crime_threat": boolean,
  "threat_confidence": float between 0.0 and 1.0,
  "outlier_category": "HORN_OUTLIER" | "SCREECH_OUTLIER" | "VOICE_DISTRESS" | "CLEAR",
  "forensic_rationale": "Clear 2-sentence forensic rationale explaining why this is or isn't a genuine human scream based on the acoustic physics.",
  "recommended_action": "DISPATCH_PATROL" | "MONITOR_ZONE" | "DISMISS_FALSE_ALARM"
}}
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
            elif text.startswith("```"):
                text = text.replace("```", "", 1).rsplit("```", 1)[0].strip()

            data = json.loads(text)
            return ForensicReport(
                classification=data.get("classification", "GENUINE_HUMAN_SCREAM"),
                is_crime_threat=bool(data.get("is_crime_threat", False)),
                threat_confidence=float(data.get("threat_confidence", cnn_prob)),
                outlier_category=data.get("outlier_category", "VOICE_DISTRESS"),
                forensic_rationale=data.get("forensic_rationale", ""),
                recommended_action=data.get("recommended_action", "MONITOR_ZONE"),
                fingerprint=asdict(fp),
            )
        except Exception:
            return None

    def _local_expert_reasoning(
        self, fp: BioacousticFingerprint, cnn_prob: float, location: str
    ) -> ForensicReport:
        """
        Deterministic bio-acoustic expert rules replicating acoustic physics.
        Guarantees zero crashes when offline or when no Gemini API key is configured.
        """
        # Case 1: Vehicle Horn / Bus Honk Detection
        if (fp.hnr_db >= 18.0 and fp.f0_modulation_pct <= 7.0) or fp.candidate_source == "VEHICLE_HORN_OR_SIREN":
            return ForensicReport(
                classification="VEHICLE_HORN",
                is_crime_threat=False,
                threat_confidence=round(max(0.04, 1.0 - cnn_prob), 4),
                outlier_category="HORN_OUTLIER",
                forensic_rationale=(
                    f"Acoustic profile demonstrates electromechanical harmonic stationarity: "
                    f"F0 modulation is only {fp.f0_modulation_pct}% with high HNR ({fp.hnr_db} dB). "
                    f"Stationary frequency structure rules out human vocal fold distress."
                ),
                recommended_action="DISMISS_FALSE_ALARM",
                fingerprint=asdict(fp),
            )

        # Case 2: Tire Screech / High Friction Noise
        if fp.spectral_flatness > 0.18 or fp.candidate_source == "FRICTION_TIRE_SCREECH":
            return ForensicReport(
                classification="TIRE_SCREECH",
                is_crime_threat=False,
                threat_confidence=0.12,
                outlier_category="SCREECH_OUTLIER",
                forensic_rationale=(
                    f"Acoustic energy displays high spectral flatness ({fp.spectral_flatness:.3f}) "
                    f"and wideband friction dispersion without biological vocal tract formant poles."
                ),
                recommended_action="DISMISS_FALSE_ALARM",
                fingerprint=asdict(fp),
            )

        # Case 3: Genuine Human Distress Scream
        if cnn_prob >= 0.50 or fp.chaos_index > 0.35 or fp.candidate_source == "HUMAN_VOCAL_DISTRESS":
            conf = round(float(np.clip(0.6 * cnn_prob + 0.4 * (fp.chaos_index + 0.35), 0.58, 0.99)), 4)
            return ForensicReport(
                classification="GENUINE_HUMAN_SCREAM",
                is_crime_threat=True,
                threat_confidence=conf,
                outlier_category="VOICE_DISTRESS",
                forensic_rationale=(
                    f"Bio-acoustic signature exhibits turbulent non-linear vocal fold chaos "
                    f"(Chaos Index: {fp.chaos_index}, F0 modulation: {fp.f0_modulation_pct}%). "
                    f"Turbulence and low harmonic stability confirm biological vocal distress."
                ),
                recommended_action="DISPATCH_PATROL",
                fingerprint=asdict(fp),
            )

        # Case 4: Normal Ambient
        return ForensicReport(
            classification="ENVIRONMENTAL_AMBIENT",
            is_crime_threat=False,
            threat_confidence=round(cnn_prob, 4),
            outlier_category="CLEAR",
            forensic_rationale="Acoustic levels remain within baseline normal ambient energy envelopes.",
            recommended_action="MONITOR_ZONE",
            fingerprint=asdict(fp),
        )

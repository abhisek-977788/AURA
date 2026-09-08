"""
Score Fusion Engine for AURA.
Multimodal Bayesian/weighted score fusion with missing modality resilience and calibrated probabilities.
"""

from __future__ import annotations

import time
import uuid
from typing import Any
import numpy as np

from packages.common.logging import get_logger
from packages.schemas.analysis_result import (
    AnalysisResult,
    BlinkAnalysis,
    Decision,
    ModalityResult,
    ModalityType,
    QualityReport,
    SyncResult,
)
from packages.schemas.media_event import MediaEvent

logger = get_logger("fusion.engine")


class FusionEngine:
    def __init__(
        self,
        audio_weight: float = 0.40,
        video_weight: float = 0.40,
        sync_weight: float = 0.20,
        high_risk_threshold: float = 0.80,
        medium_risk_threshold: float = 0.50,
        min_confidence_for_decision: float = 0.40,
    ) -> None:
        self.audio_weight = audio_weight
        self.video_weight = video_weight
        self.sync_weight = sync_weight
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.min_confidence_for_decision = min_confidence_for_decision

    async def fuse(
        self,
        event_id: str,
        case_id: str | None = None,
        job_id: str | None = None,
        audio_result: ModalityResult | None = None,
        video_result: ModalityResult | None = None,
        sync_result: SyncResult | None = None,
        blink_analysis: BlinkAnalysis | None = None,
        quality_report: QualityReport | None = None,
    ) -> AnalysisResult:
        available_modalities: list[ModalityType] = []
        missing_modalities: list[ModalityType] = []
        modality_results_map: dict[str, ModalityResult] = {}
        all_limitations: list[str] = []
        top_signals: list[str] = []

        active_weights = []
        scores = []
        confidences = []

        # 1. Process Audio Modality
        if audio_result and audio_result.available and audio_result.ensemble_score is not None:
            available_modalities.append(ModalityType.AUDIO)
            modality_results_map["audio"] = audio_result
            scores.append(audio_result.ensemble_score)
            confidences.append(audio_result.ensemble_confidence or 0.7)
            active_weights.append(self.audio_weight)
            all_limitations.extend(audio_result.limitations)
            top_signals.extend(audio_result.top_features)
        else:
            missing_modalities.append(ModalityType.AUDIO)
            all_limitations.append("Audio modality not present in source media. Evaluated without audio verification.")

        # 2. Process Video Modality
        if video_result and video_result.available and video_result.ensemble_score is not None:
            available_modalities.append(ModalityType.VIDEO)
            modality_results_map["video"] = video_result
            scores.append(video_result.ensemble_score)
            confidences.append(video_result.ensemble_confidence or 0.7)
            active_weights.append(self.video_weight)
            all_limitations.extend(video_result.limitations)
            top_signals.extend(video_result.top_features)
        else:
            missing_modalities.append(ModalityType.VIDEO)
            all_limitations.append("Video modality not present in source media.")

        # 3. Process Audio-Visual Sync
        if sync_result and sync_result.available and sync_result.sync_score is not None:
            available_modalities.append(ModalityType.SYNCHRONIZATION)
            scores.append(sync_result.sync_score)
            confidences.append(sync_result.sync_confidence or 0.6)
            active_weights.append(self.sync_weight)
        elif ModalityType.AUDIO in available_modalities and ModalityType.VIDEO in available_modalities:
            missing_modalities.append(ModalityType.SYNCHRONIZATION)

        # 4. Handle Case: No modalities available
        if not scores:
            return AnalysisResult.inconclusive(
                event_id=event_id,
                reason="No verifiable modalities were present or processed",
                quality_report=quality_report,
            )

        # 5. Weighted Bayesian Fusion (Missing modalities are NOT treated as evidence of fakery!)
        norm_weights = np.array(active_weights) / sum(active_weights)
        fused_prob = float(np.dot(norm_weights, np.array(scores)))
        mean_confidence = float(np.min(confidences))  # Bottlenecked by least confident modality

        # Uncertainty estimate
        model_uncertainty = float(1.0 - mean_confidence)
        if len(scores) > 1:
            # Model disagreement increases uncertainty
            disagreement = float(np.std(scores))
            model_uncertainty = min(1.0, model_uncertainty + disagreement * 0.5)

        # 6. Determine Calibrated Decision
        if mean_confidence < self.min_confidence_for_decision:
            decision = Decision.INCONCLUSIVE
            rationale = (
                f"Model uncertainty is too high (confidence {mean_confidence:.2f} < "
                f"{self.min_confidence_for_decision}). Result cannot be stated conclusively."
            )
        elif fused_prob >= self.high_risk_threshold:
            decision = Decision.HIGH_RISK
            rationale = (
                f"High likelihood of synthetic manipulation detected ({fused_prob:.1%}). "
                "Primary contributing signals: " + ", ".join(top_signals[:3])
            )
        elif fused_prob >= self.medium_risk_threshold:
            decision = Decision.MEDIUM_RISK
            rationale = (
                f"Moderate synthetic indicators detected ({fused_prob:.1%}). "
                "May indicate minor localized tampering, re-encoding, or voice conversion."
            )
        else:
            decision = Decision.LOW_RISK
            rationale = f"Natural representations observed ({fused_prob:.1%}). Consistent with genuine media."

        return AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            event_id=event_id,
            case_id=case_id,
            job_id=job_id,
            synthetic_media_probability=round(fused_prob, 4),
            confidence=round(mean_confidence, 4),
            decision=decision,
            available_modalities=available_modalities,
            missing_modalities=missing_modalities,
            modality_results=modality_results_map,
            sync_result=sync_result or SyncResult(available=False),
            blink_analysis=blink_analysis or BlinkAnalysis(available=False),
            top_contributing_signals=top_signals[:5],
            quality_report=quality_report,
            model_uncertainty=round(model_uncertainty, 4),
            limitations=list(dict.fromkeys(all_limitations)),
            decision_rationale=rationale,
            requires_human_review=fused_prob >= 0.5 or decision == Decision.HIGH_RISK,
            fusion_version="1.0.0",
            software_version="0.1.0",
        )

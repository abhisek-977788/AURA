"""
Real-Time and Batch Alert Policy Engine.
Routes detection outputs into actionable responses without making automated accusations.
"""

from __future__ import annotations

from packages.schemas.alert import AlertAction, AlertEvent, AlertLevel, AlertPolicy
from packages.schemas.analysis_result import AnalysisResult, Decision


class AlertPolicyEngine:
    def __init__(self, policy: AlertPolicy | None = None) -> None:
        self.policy = policy or AlertPolicy()

    def evaluate(self, result: AnalysisResult) -> AlertEvent:
        prob = result.synthetic_media_probability
        conf = result.confidence or 0.0

        actions: list[AlertAction] = []
        decision_trace: str

        if result.decision == Decision.INCONCLUSIVE or conf < self.policy.min_confidence_for_decision:
            level = AlertLevel.INCONCLUSIVE
            actions = [
                AlertAction.REQUEST_BETTER_CAPTURE,
                AlertAction.ESCALATE_TO_MANUAL_REVIEW,
            ]
            decision_trace = (
                f"Signal quality or model confidence ({conf:.2f}) was inadequate. "
                "Triggering manual review rather than false alarm."
            )
        elif result.decision == Decision.HIGH_RISK or (prob is not None and prob >= self.policy.high_risk_threshold):
            level = AlertLevel.HIGH_RISK
            actions = [
                AlertAction.NOTIFY_INVESTIGATORS,
                AlertAction.FREEZE_ANALYSIS_WINDOW,
                AlertAction.CAPTURE_MINIMUM_EVIDENCE,
                AlertAction.GENERATE_EVIDENCE_PACKAGE,
                AlertAction.LOG_POLICY_DECISION,
            ]
            decision_trace = (
                f"High-risk synthetic media likelihood ({prob:.1%}) exceeded threshold "
                f"({self.policy.high_risk_threshold:.1%}). Enforcing minimum evidence capture."
            )
        elif result.decision == Decision.MEDIUM_RISK or (prob is not None and prob >= self.policy.medium_risk_threshold):
            level = AlertLevel.MEDIUM_RISK
            actions = [
                AlertAction.WARN_ANALYST,
                AlertAction.INCREASE_SAMPLING,
                AlertAction.REQUEST_VERIFICATION,
            ]
            decision_trace = (
                f"Moderate synthetic anomaly ({prob:.1%}) detected. Notifying analyst "
                "to inspect without auto-freeze."
            )
        else:
            level = AlertLevel.LOW_RISK
            actions = [
                AlertAction.CONTINUE_MONITORING,
                AlertAction.STORE_AGGREGATE_TELEMETRY,
            ]
            decision_trace = "Media indicators remain within normal biological variance."

        return AlertEvent(
            analysis_id=result.analysis_id,
            event_id=result.event_id,
            case_id=result.case_id,
            level=level,
            actions=actions,
            synthetic_media_probability=prob,
            confidence=conf,
            top_signals=result.top_contributing_signals,
            limitations=result.limitations,
            policy_version=self.policy.policy_version,
            policy_decision_rationale=decision_trace,
        )

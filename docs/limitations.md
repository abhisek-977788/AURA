# AURA — Limitations and Responsible Use Documentation

## 1. Statutory & Forensic Disclaimer
AURA (Audio-Visual Unnatural Representation Analyzer) produces **calibrated statistical estimations of synthetic manipulation**, not legally conclusive determinations of guilt, identity theft, or criminal conduct.

1. **Court Admissibility**: Software output from AURA is **not automatically court-admissible**. Admissibility depends on jurisdictional rules of evidence (e.g., Section 65B of the Indian Evidence Act, US Federal Rules of Evidence 902), procedural chain-of-custody authentication, and qualified expert testimony.
2. **Investigative Support Only**: High-risk or medium-risk alerts serve as triage indicators for forensic examiners and authorized cybercrime investigators, not grounds for autonomous punitive action.
3. **Inconclusive Gating**: AURA explicitly gates degraded signals and returns `decision: "inconclusive"` with `insufficient_signal_quality` rather than forcing a speculative classification.

---

## 2. Technical Limitations

### Narrowband Telephony (8 kHz)
- Standard telephone networks (PSTN, GSM, 2G/3G, G.711 codecs) aggressively filter out all acoustic content above 3.4 kHz.
- **Critical Caution**: Missing high-frequency harmonics in telephone calls is normal codec behavior and must **never** be used as standalone evidence of synthetic vocoder generation. Telephony evaluation requires the dedicated `TelephonyCNN-8k` model.

### Blink Dynamics & Eye Aspect Ratio (EAR)
- Blink rates vary widely across human populations due to fatigue, cognitive workload, dry eyes, camera lighting, and viewing angle.
- AURA **strictly rejects** universal heuristic thresholds (such as "fewer than 10 blinks per minute"). Blink analysis is integrated solely as a secondary temporal feature within the score fusion layer.

### Adversarial Robustness & Novel Generators
- Deepfake detectors trained on specific benchmarks (e.g., FaceForensics++ or ASVspoof) may experience performance degradation on unseen generator architectures (e.g., newer diffusion or latent audio models).
- Regular cross-dataset benchmarking and human review are mandatory before operational conclusions.

---

## 3. Privacy & Ephemeral Data Retention
- Live video/audio interception (via WebRTC browser extensions or SBC media relays) operates under **ephemeral mode**: raw media is processed in sliding in-memory buffers and is discarded immediately unless a high-threat alert triggers authorized evidence preservation.
- All subject identifiers in common media events are pseudonymous (e.g., `anon-subject-1234`). Real biometric templates or identifiers are never persisted without statutory authorization.

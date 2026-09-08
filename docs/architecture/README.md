# AURA System Architecture

AURA decouples media ingestion from machine learning inference through a unified, event-driven streaming pipeline.

```
Input Source (Photo / Audio / Video / Live Call)
       │
       ▼
[ Ingestion Adapters ] (PhotoAdapter, VideoAdapter, AudioAdapter, LiveAdapters)
       │  Emits MediaEvent (common schema)
       ▼
[ Redis Stream: aura:ingestion ]
       │
       ▼
[ Normalization Service ] (Demux audio, extract frames @ 2fps, resample 8k/16k)
       │
       ▼
[ Quality Gate Service ] (SNR, clipping, face size, blur, resolution check)
       │
       ├─────────────────────────────────┐
  (Quality Insufficient)           (Quality Sufficient)
       │                                 │
       ▼                                 ├──► [ aura:audio_inference ]
[ Inconclusive Decision ]                │         │
       │                                 │         ▼
       │                                 │    [ Audio Inference Worker ]
       │                                 │    (AASIST, RawNet2, LFCC, TelephonyCNN)
       │                                 │         │
       │                                 │         ▼
       │                                 │    [ aura:audio_results ]
       │                                 │
       │                                 └──► [ aura:video_inference ]
       │                                           │
       │                                           ▼
       │                                      [ Video Inference Worker ]
       │                                      (MesoNet, FFT, Boundary, Blink EAR)
       │                                           │
       │                                           ▼
       │                                      [ aura:video_results ]
       │                                           │
       ▼                                           ▼
[ Score Fusion & Calibration Engine ] ◄────────────┘
(Bayesian / weighted ensemble, missing-modality resilience, Platt/isotonic calibration)
       │
       ▼
[ aura:analysis_complete ]
       │
       ├──► [ Alert Policy Engine ] (Low / Medium / High Risk / Inconclusive)
       │         │
       │         └──► [ Evidence Packaging ] (Ed25519 Signed Manifest & ZIP)
       │
       ├──► [ Immutable Audit Logger ] (SHA-256 Hash Chain Ledger)
       │
       ▼
[ Forensics Review Portal & Case Management API ]
```

## Core Principles
1. **Common Media Event Schema**: Ingestion is decoupled from inference models. Any new input type (e.g. SIP telephony, WebRTC browser extension, or body-worn camera) only requires a new Ingestion Adapter.
2. **Missing Modality Resilience**: An absent modality (e.g., audio-only call with no video) is **never** treated as evidence of fakery. Missing modalities are weighted out dynamically with uncertainty reporting.
3. **Calibrated Probabilities**: Scores reflect empirical likelihood rather than arbitrary confidence scores.
4. **Tamper-Evident Evidence**: Every exported evidence manifest is signed with Ed25519 and includes cryptographic hashes of all original and derived artifacts.

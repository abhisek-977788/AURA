# AURA — Audio-Visual Unnatural Representation Analyzer

A privacy-preserving, real-time and batch deepfake detection platform for authorized cybercrime investigations, fraud prevention, incident response, and media forensics.

## ⚠️ Important Limitations & Responsible Use

AURA produces **calibrated probability scores**, not legal determinations. Detection results:
- Are **not** automatically court-admissible
- Require **human analyst review** before any investigative action
- Are subject to model uncertainty, signal quality limitations, and dataset bias
- Must not be used to make irreversible identity or criminal conduct decisions

See [`docs/limitations.md`](docs/limitations.md) for full responsible-use documentation.

---

## Architecture

```
Input Source
→ Ingestion Adapter
→ Media Normalization
→ Quality Assessment
→ Modality Router
→ Feature Extraction
→ Model Inference
→ Score Fusion
→ Calibration
→ Alert Policy
→ Explanation Generation
→ Evidence Packaging
→ Investigative Forensics Portal
```

All ingestion adapters produce a **common `MediaEvent` schema**. The inference layer is fully decoupled from frontend, codec, and transport concerns.

---

## Repository Structure

```
/apps
  /api                  FastAPI gateway
  /forensics-portal     Next.js analyst portal
  /browser-extension    Chrome Extension (Phase 4)
/services
  /ingestion            Ingestion adapters (photo, audio, video, live)
  /normalization        FFmpeg-based media normalization
  /quality              Quality assessment layer
  /audio-inference      AASIST, RawNet2, LFCC, Mel classifiers
  /video-inference      MesoNet, EfficientNet-B4, FFT classifiers
  /fusion               Score fusion + calibration
  /alerts               Alert policy engine
  /evidence             Forensic evidence packaging
  /audit                Immutable audit logging
/ml
  /datasets             Dataset manifests & download scripts
  /preprocessing        Audio/video feature extraction
  /training             Model training scripts
  /evaluation           Metrics & benchmarking
  /calibration          Platt scaling / isotonic regression
  /export               ONNX export scripts
  /quantization         INT8 quantization for browser WASM
/packages
  /schemas              Pydantic/JSON schemas (shared)
  /common               Shared utilities
  /security             Crypto, signing, RBAC
  /client-sdk           Python client SDK
/infrastructure
  /docker               Dockerfiles per service
  /compose              docker-compose.yml
  /kubernetes           K8s manifests (optional)
/tests
  /unit
  /integration
  /e2e
  /security
/docs
  /architecture
  /api
  /threat-model
  /model-cards
  /runbooks
```

---

## Quick Start (Local Development)

### Prerequisites
- Docker Desktop 24+
- Docker Compose v2
- Python 3.11+
- Node.js 20+ (for forensics portal)
- FFmpeg (for local dev outside Docker)

### Start All Services

```bash
make dev
```

Or manually:

```bash
cd infrastructure/compose
docker compose up --build
```

Services will be available at:
| Service | URL |
|---------|-----|
| API Gateway | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Forensics Portal | http://localhost:3000 |
| MinIO Console | http://localhost:9001 |
| PgAdmin | http://localhost:5050 |

### Run Tests

```bash
make test-unit
make test-integration
make test-e2e
make test-security
```

---

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ In Progress | Core batch pipeline (recorded video/audio) |
| 2 | 🔜 Planned | Forensics portal, case management |
| 3 | 🔜 Planned | Static photo analysis |
| 4 | 🔜 Planned | Live video (Chrome Extension + WASM) |
| 5 | 🔜 Planned | Live audio (SIP/RTP) |
| 6 | 🔜 Planned | Production hardening, calibration, security |

---

## Dataset Licensing

AURA does **not** bundle training data. Each dataset must be acquired under its own license:

| Dataset | Use | License |
|---------|-----|---------|
| ASVspoof 2021 | Audio deepfake detection | Research use, see asvspoof.org |
| FaceForensics++ | Video manipulation detection | Non-commercial research |
| DFDC | Video deepfake detection | CC BY-NC 4.0 |
| Celeb-DF | Face-forensics benchmark | Research only |

See [`ml/datasets/MANIFEST.md`](ml/datasets/MANIFEST.md) for download instructions.

---

## License & Legal

AURA is intended **solely** for authorized use by law enforcement, licensed forensic professionals, and security researchers operating under applicable legal frameworks. Unauthorized interception of communications is illegal. All evidence exports are for investigative support only and require qualified legal review before use in judicial proceedings.

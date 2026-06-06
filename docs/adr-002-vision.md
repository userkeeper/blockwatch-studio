# ADR-002: Vision-based detection pipeline

**Status**: Accepted (Phase 0 scaffold landed; Phases 1-4 planned)
**Date**: 2026-06-07

## Context

The Phase 1/2 detection pipeline runs Windows.Media.Ocr → regex
detectors → sticky-blur. After spending a full session tuning
patterns, regex thresholds, normalisation, row-merging and
frame-diff gating against real screen content, we hit fundamental
limits:

1. **OCR fragmentation.** Windows.Media.Ocr splits long alphanumeric
   strings at unpredictable positions. A 50-char API key returns as
   two 25-char fragments, and no targeted regex anchored on the
   prefix can match either half. Even our HighEntropyToken catch-all
   (18+ char alphanumeric blobs) over-blurs short slug-like words
   and *still* misses fragments shorter than 18 chars.

2. **OCR character confusion.** `K` ↔ `k`, `I` ↔ `l`, `O` ↔ `0`, `5` ↔ `S`
   happen on every small-font screen. Case-insensitive regexes help
   but case is genuinely lost information once the OCR has decided.

3. **Localised UI noise.** Cyrillic UI elements bleed into OCR
   output on Russian / mixed-locale Windows — `KEY` becomes `КЕУ`.
   Patterns can't anchor on those tokens.

4. **The fundamental error.** We are inferring sensitive content
   from a noisy text channel, when the actual signal is *visual*:
   a Phantom popup looks like a Phantom popup, a `.env` file in
   VS Code looks like a `.env` file. A trained vision model
   recognises the *thing* on screen, not its textual transcription.

## Decision

Migrate the detection layer from "OCR + regex" to "vision +
classifier", in four phases. Keep the regex path running as a
fall-back during the migration so we never regress overall
coverage.

## Architecture

```
                        ┌──────────────────────────────────┐
                        │  Phase 1: YOLOv8 popup detector  │
   ┌─────────┐          │  (640×640 frame → boxes for      │
   │ Capture │ ───────▶ │   wallet popups, .env editors,   │
   └─────────┘          │   terminals, dashboards)         │
                        └──────────────┬───────────────────┘
                                       │ DetectedObject[]
                                       ▼
                        ┌──────────────────────────────────┐
                        │  Phase 2: text-region detector   │
                        │  (DBNet or similar — finds text  │
                        │   pixels INSIDE the popup boxes  │
                        │   for surgical blur, not whole-  │
                        │   region blur)                   │
                        └──────────────┬───────────────────┘
                                       │ Refined BBox[]
                                       ▼
                        ┌──────────────────────────────────┐
                        │  Phase 3: regex fall-back        │
                        │  (only inside Phase-1 boxes; if  │
                        │   YOLO didn't fire, regex still  │
                        │   runs over the whole frame as a │
                        │   safety net during migration)   │
                        └──────────────┬───────────────────┘
                                       │ Vec<Hit>
                                       ▼
                                  StickyHits + blur
                                  (unchanged)
```

## Class taxonomy

The YOLOv8 model is trained against these 6 classes (mirrored in
`bw_vision::ObjectClass`):

| Class                  | What it covers                                     |
| ---------------------- | -------------------------------------------------- |
| `WalletReceivePopup`   | Phantom, MetaMask, Tonkeeper, Trust, Backpack receive screens |
| `SeedPhraseReveal`     | Wallet seed-phrase backup / reveal screen          |
| `PrivateKeyReveal`     | Wallet export-private-key screen                   |
| `EnvFileEditor`        | Notepad / VS Code / Sublime open on a `.env` file  |
| `CredentialTerminal`   | Terminal showing `export ...`, `aws configure`, etc. |
| `PortfolioBalance`     | TradingView, CEX dashboards, balance widgets       |

We deliberately do NOT include a "anything sensitive" class; the
model must choose one of these six. New leak vectors get their own
class + their own labelled examples.

## Dataset

- **Source**: hand-collected screenshots of each class, ~50-100 per
  class. We will record on Windows 11 (representative of the target
  OS) and capture variations: light/dark theme, different DPI
  scales, different window sizes.
- **Annotation**: LabelImg, YOLO format. Single annotator initially
  (the project owner) to avoid disagreement on edge cases.
- **Augmentation**: stock Ultralytics augmentation (mosaic, HSV jitter,
  perspective). No domain-specific augmentation needed at first.
- **Split**: 80% train / 10% val / 10% test, stratified per class.

## Training

Trained outside this repo, in Python (Ultralytics CLI). Resulting
weights checkpointed to `crates/vision/models/popup-yolov8n-vN.onnx`.
The model file IS committed (binary, ~6 MB) so users don't need a
training pipeline to run the detector. Each model version is named
in semver: `v1.0.0`, `v1.1.0` etc.; the runtime loads the path
declared in `MODEL_PATH` const.

## Inference

- **Runtime**: `ort` 2.x (ONNX Runtime via Microsoft). CPU provider
  only; no GPU dependency. ~40 ms per 640×640 frame on a recent
  laptop iGPU — fits the 5-10 fps pipeline budget.
- **Status (2026-06-07)**: `ort` 2.0.0-rc.10 has a version-mismatch
  bug against `ort-sys` rc.12 (the rc.10 crate calls
  `SessionOptionsAppendExecutionProvider_VitisAI` which doesn't exist
  in rc.12's struct). We defer adding `ort` as a dependency until
  either rc.11+ pins a compatible `ort-sys` or the 2.0 stable
  release ships. `bw-vision` compiles cleanly without it because
  the inference path is stubbed at Phase 0.
- **Pre-processing**: BGRA frame → resize to 640×640 (letterbox) →
  normalise to [0, 1] → CHW format.
- **Post-processing**: confidence threshold 0.35, NMS at IoU 0.5,
  rescale boxes back to original frame coordinates.

## Rejected alternatives

- **CLIP zero-shot**: we considered prompting CLIP with "a wallet
  receive QR popup" / "an .env file" but CLIP zero-shot accuracy
  on this kind of structured-UI input is poor (~60%), versus a
  purpose-trained YOLOv8 (~95% on held-out test set in similar
  domains).
- **GPT-4-Vision API**: latency and cost rule it out for a 5-10 fps
  per-frame loop. Even at 1 fps, $0.01/call × 5 fps × 3600 s =
  $180/hour per streamer is unworkable.
- **OCR + better regex**: tried; this ADR exists because that path
  was exhausted.

## Migration plan

| Phase | Deliverable                                      | Status   |
| ----- | ------------------------------------------------ | -------- |
| 0     | `bw-vision` crate skeleton + ADR                 | DONE (this PR) |
| 1     | 300+ labelled screenshots, YOLOv8n trained, .onnx committed | NEXT |
| 2     | CLI `--vision` flag, runs alongside regex        |          |
| 3     | DBNet text-region refinement inside YOLO boxes   |          |
| 4     | Make `--vision` the default; regex becomes fall-back |      |
| 5     | Remove regex path entirely, ship `--vision`-only |          |

## Consequences

- **Bundle size grows** from ~4 MB (current release) to ~10-15 MB
  including the ONNX model. Acceptable for a desktop installer.
- **Detection latency grows** by ~40 ms / frame. Still well under
  the 100 ms / frame budget at 10 fps.
- **Dataset curation is a real ongoing project.** Adding support
  for a new wallet means collecting 30-50 screenshots and re-training,
  not editing a regex. We trade per-second tuning for per-week
  upgrades.
- **No false positives on plain text.** A long English word, a
  filename, a slug — none of these visually resemble a wallet popup.
  The annoying over-blur we have in panic-mode disappears.

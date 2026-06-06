# BlockWatch Shield Studio — Roadmap

## Phase 0 — Foundation (week 1)

- [x] Repo scaffold, README, ROADMAP, ADRs
- [ ] Cargo workspace with stub crates that compile to nothing useful
- [ ] CI: `cargo build`, `cargo clippy`, `cargo test` on Windows + macOS + Linux
- [ ] Apple Developer account + signing certificate (gating mac build)
- [ ] Set up `crates/core/src/bip39.rs` — copy 2048-word list from extension
- [ ] Set up `crates/core/src/detect.rs` — port BIP-39 / regex detectors
      from `packages/extension/src/streamMode/patterns.ts`

## Phase 1 — PoC on Windows (weeks 2-3)

Goal: a CLI binary that captures the screen, OCRs, blurs, and writes the
output to a `.mp4` file (no virtual camera yet). Proves the pipeline.

- [ ] `crates/capture/src/windows.rs` — DDA via `windows-rs`,
      60 fps RGBA frames
- [ ] `crates/core/src/buffer.rs` — ring buffer with metadata slots
- [ ] `crates/ocr/src/windows.rs` — `Windows.Media.Ocr` via `windows-rs`
- [ ] `crates/core/src/blur.rs` — gaussian blur via `wgpu`
      compute shader
- [ ] CLI tool `tools/poc-windows` — runs the full pipeline to MP4
- [ ] Manual test: open Tonkeeper recovery screen, run PoC, confirm
      seed is blurred in output MP4

**Success criteria**: open a screen with a visible BIP-39 phrase,
recording shows the phrase blurred for the entire visible duration,
end-to-end latency under 3.5 s.

## Phase 2 — Virtual camera on Windows (week 4)

- [ ] `crates/virtualcam/src/windows.rs` — DirectShow filter via COM, or
      shim onto OBS Virtual Camera kernel driver (lower-effort path)
- [ ] Wire PoC output to virtual camera instead of MP4
- [ ] Manual test: OBS → Source → Video Capture Device → "BlockWatch
      Shield Cam" → preview shows the blurred desktop
- [ ] Manual test: Zoom, Discord, Teams — same flow

## Phase 3 — Tauri UI shell (week 5)

- [ ] `crates/app-tauri/src-tauri` — backend that owns the pipeline
- [ ] System-tray app with one window
- [ ] UI: ON/OFF toggle, buffer-delay dropdown (3/5/10 s), detector
      checkboxes (seed phrases, private keys, QR codes, addresses opt-in)
- [ ] Status line: "Blocked this session: 2 seed phrases, 1 QR code"
- [ ] Windows installer (MSI) via `cargo-wix`
- [ ] Auto-update via Tauri updater

## Phase 4 — macOS (weeks 6-7)

- [ ] `crates/capture/src/macos.rs` — ScreenCaptureKit via `objc2`
- [ ] `crates/ocr/src/macos.rs` — Vision framework via `objc2`
- [ ] `crates/virtualcam/src/macos.rs` — CMIO Sample Extension
      (full Sample Extension scaffold, signed + notarised)
- [ ] DMG installer with auto-extension-load instructions
- [ ] Manual test: OBS on macOS → "BlockWatch Shield Cam" appears in
      sources

## Phase 5 — Linux (week 8)

- [ ] `crates/capture/src/linux.rs` — PipeWire screen capture
- [ ] `crates/ocr/src/linux.rs` — Tesseract via `leptess`
- [ ] `crates/virtualcam/src/linux.rs` — write raw frames to
      `/dev/videoN` via `v4l2loopback`
- [ ] AppImage + Flatpak releases
- [ ] Manual test: OBS on Ubuntu → "BlockWatch Shield Cam" works

## Phase 6 — Polish + ship (week 9)

- [ ] Landing page at `blockwatch.tech/studio` with download buttons
- [ ] Pricing: free 1h/day with watermark, Pro $5/mo
- [ ] Payment via TG Stars / TON / Stripe (reuse `block-watch-v2` API)
- [ ] Bundled with Block Watch Premium subscription
- [ ] Public beta in `@BWDevapp`
- [ ] Submit to AlternativeTo, ProductHunt

## Phase 7 — Mobile (Phase 2 of the product, NOT in this 9-week window)

- [ ] iOS Broadcast Extension
- [ ] Android MediaProjection + accessibility hook

## Risks & open questions

- **macOS notarisation** for the CMIO Sample Extension is the biggest
  bureaucratic risk. Mac users will have to enable the extension in
  System Settings → Privacy & Security; this is friction.
- **Anti-cheat software** in some games detects screen-capture and may
  ban. Need clear in-UI warning.
- **OBS Virtual Camera driver** is GPL'd. If we shim onto it instead
  of writing our own DShow filter, license obligations need a lawyer's
  read.
- **Performance on integrated GPUs**: blur via wgpu compute should fall
  back gracefully if no GPU is available.

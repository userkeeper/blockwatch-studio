# BlockWatch Shield Studio

Cross-platform virtual camera that blurs crypto secrets in real time before
they reach the stream output.

The user picks **"BlockWatch Shield Cam"** as their video source in OBS,
Streamlabs, Zoom, Discord, Meet, Teams, or anything that accepts a camera.
The app captures the screen, holds it in a 3-second buffer, OCRs each
frame to find sensitive content (BIP-39 seed phrases, hex/WIF private
keys, xprv/xpub, crypto addresses, QR codes), and blurs detected regions
before passing the frame to the virtual camera output. Because OCR
happens on a frame that won't be shown for 3 more seconds, the blur is
always already in place when viewers see the frame.

## Why a virtual camera and not an OBS plugin

| Approach              | Reach                                                                              | Effort     |
| --------------------- | ---------------------------------------------------------------------------------- | ---------- |
| OBS plugin (C++)      | OBS only                                                                           | Lower      |
| **Virtual camera**    | **OBS + Streamlabs + Twitch Studio + Zoom + Discord + Meet + Teams + TikTok + …** | Higher     |
| Browser via WebRTC    | Browser only — no output sink as camera                                            | Impossible |

A virtual camera covers every desktop streaming/conferencing app in one
build, which is the entire goal.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       Capture layer                            │
│  Windows: DDA (DXGI)   macOS: ScreenCaptureKit   Linux: PipeWire│
└────────────────────────────────────────────────────────────────┘
                              │ 60 fps RGBA frames
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  Ring buffer (3 s @ 60 fps = 180 frames)       │
│  Lock-free SPSC queue with shared frame metadata               │
└────────────────────────────────────────────────────────────────┘
            │  HEAD: write new frame      TAIL: emit oldest frame
            ▼                                            ▲
┌───────────────────────┐                  ┌──────────────────────┐
│   OCR worker (5 fps)  │                  │   Blur compositor    │
│ • Downscale to 720p   │                  │ • Read bbox metadata │
│ • Frame diff vs prev  │                  │ • Apply gaussian     │
│ • Tesseract / native  │                  │   blur to bboxes     │
│ • BIP-39 / regex / QR │                  │   on full-res frame  │
│ • Write bbox list to  │ ─── metadata ──▶ │ • Encode RGBA → NV12 │
│   frames N..N+30      │     (sticky bbox)│                      │
└───────────────────────┘                  └──────────────────────┘
                                                       │
                                                       ▼
┌────────────────────────────────────────────────────────────────┐
│                 Virtual camera output                          │
│  Windows: DShow filter   macOS: CMIO Sample Extension          │
│  Linux: v4l2loopback module                                    │
└────────────────────────────────────────────────────────────────┘
```

### Key insight: detect on the head, blur on the tail

Frame N enters the buffer at time T. The OCR worker (5 Hz) picks up the
most recent frame (still N), runs detection, and writes detected bboxes
back into the frame's metadata. When frame N reaches the buffer tail at
T+3 s, the compositor applies blur over the bboxes already attached to it.

By the time viewers see the frame, the blur is already there. The 3-
second buffer is the entire "delay" the streamer has to absorb — small
enough not to break interactivity, large enough that even multi-region
OCR (~150 ms at 720p) finishes with margin.

## Detectors

All implemented in `crates/core/src/detect.rs`:

- **BIP-39 seed phrases** — runs of ≥8 consecutive valid words from the
  2048-word English list (and localised lists). False-positive rate at
  ≥8 is effectively zero.
- **Hex private keys** — `\b(?:0x)?[0-9a-fA-F]{64}\b`
- **WIF private keys** — `\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b`
- **xprv / xpub / ypub / zpub / tprv / tpub** — base58 107-108 chars
- **Solana base58 secret keys** — 64-char base58 (when not in TX field)
- **Optional: crypto addresses** — toggleable, off by default (many
  streamers want to show donation addresses)
- **QR codes** — `rxing` (Rust port of zxing), all sizes, always blurred

## Performance budget

| Step               | Target time per cycle | Notes                              |
| ------------------ | --------------------- | ---------------------------------- |
| Capture frame      | < 5 ms                | GPU surface, no CPU copy           |
| Push to buffer     | < 1 ms                | Lock-free                          |
| OCR pass           | 100–200 ms @ 5 fps    | Downscale to 720p, frame-diff      |
| Blur compositor    | < 10 ms per region    | GPU shader                         |
| Encode + emit      | 5–10 ms               | NV12 from RGBA                     |

5 fps OCR + sticky-bbox-for-30-frames means the cost averages ~30 ms/s,
i.e. ~3 % of one CPU core. Capture + blur stay on the GPU.

## Stack

- **UI**: Tauri 2 (Rust + system WebView) — ~10 MB bundle, native feel
- **Capture & blur**: Rust + wgpu (cross-platform GPU compute)
- **OCR**:
  - macOS: native Vision framework (`objc2` bindings)
  - Windows: native `Windows.Media.Ocr`
  - Linux: Tesseract via `leptess` crate
- **QR**: `rxing`
- **Virtual cam**:
  - Windows: bundle a DirectShow filter or shim onto OBS Virtual Camera driver
  - macOS: CMIO Sample Extension (notarised; requires Apple Developer Program)
  - Linux: `v4l2loopback` kernel module + raw frame write
- **Updates**: Tauri's built-in updater (signed releases)

## Repository layout

```
blockwatch-studio/
├── Cargo.toml               # Workspace manifest
├── README.md                # You are here
├── ROADMAP.md               # Week-by-week build plan
├── docs/
│   ├── adr-001-virtcam.md   # Why virtual camera over OBS plugin
│   ├── adr-002-buffer.md    # Buffer delay design
│   └── platform-notes.md    # Per-OS quirks
└── crates/
    ├── core/                # Pure logic — runs anywhere
    │   ├── detect.rs        # BIP-39 / regex / QR detectors
    │   ├── bip39.rs         # 2048-word lists per language
    │   ├── buffer.rs        # Ring buffer + metadata
    │   └── blur.rs          # Gaussian blur (wgpu)
    ├── capture/             # Platform-specific screen capture
    │   ├── windows.rs       # DDA via windows-rs
    │   ├── macos.rs         # ScreenCaptureKit via objc2
    │   └── linux.rs         # PipeWire via pipewire-rs
    ├── ocr/                 # OCR abstraction
    │   ├── windows.rs       # Windows.Media.Ocr
    │   ├── macos.rs         # Vision framework
    │   └── linux.rs         # Tesseract
    ├── virtualcam/          # Virtual camera driver bindings
    │   ├── windows.rs       # DShow filter / OBS shim
    │   ├── macos.rs         # CMIO Sample Extension
    │   └── linux.rs         # v4l2loopback
    └── app-tauri/           # UI shell
        ├── src-tauri/       # Rust backend
        └── ui/              # HTML/CSS/JS for the trivial UI
```

## Build & run (developer setup)

```bash
# Prerequisites
rustup install stable
cargo install tauri-cli@^2

# Per OS:
# Windows: Windows 10 SDK 19041+, Visual Studio Build Tools
# macOS:   Xcode 15+, Apple Developer account (for cam extension)
# Linux:   pkg-config, libpipewire-0.3-dev, v4l2loopback-dkms

cd blockwatch-studio
cargo tauri dev
```

## License

Proprietary. Part of the BlockWatch product family.

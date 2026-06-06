# ADR-001: Virtual camera over OBS plugin

**Status**: Accepted  
**Date**: 2026-06-06

## Context

We need to blur on-screen crypto secrets (BIP-39 phrases, private keys,
QR codes) before they leave the streamer's machine. The Chrome-extension
"Stream Mode" only blurs the DOM, which doesn't help when the streamer
shares their full desktop through OBS, Streamlabs, Zoom, Discord, or any
other capture-based broadcaster. We need a real native solution that
intercepts the captured frame, not just the rendered HTML.

Three architectures were considered:

1. **OBS plugin (native C++)** — registers as an `obs_source_filter` in
   OBS's pipeline. Tight integration, lowest latency.
2. **Virtual camera (this ADR)** — own a system-wide virtual camera
   device. Streamer picks it as their video source in OBS, Zoom, etc.
3. **Browser-based via `getDisplayMedia` + WebRTC** — runs in a browser
   tab. No way to output back as a system camera.

## Decision

We build a **virtual camera**.

## Why

**Coverage.** One installer covers every desktop streaming / video-conf
app that accepts a camera input. OBS, Streamlabs, Twitch Studio, Zoom,
Discord, Meet, Teams, Skype, TikTok Live Studio, OBS Studio, vMix — all
expose the same "Video Capture Device" picker. An OBS plugin works in
exactly one app. Streamers who use Streamlabs (~30% of Twitch creators)
would not be covered by the plugin path.

**Single product story.** "Pick BlockWatch Shield Cam as your source" is
trivially explainable to a non-technical streamer. "Install this plugin
in OBS, but only if you use OBS, and re-install if OBS updates" is not.

**Cross-platform on one codebase.** Each OS exposes a virtual-camera API
(DShow on Windows, CMIO on macOS, v4l2loopback on Linux). The OCR + blur
pipeline above the driver is OS-independent; only the thin capture and
output adapters change per OS. An OBS plugin would still need three OS
builds, so plugin-vs-virtcam is not a cross-platform win for the plugin.

**Latency is acceptable.** Our 3-second buffer is what gives OCR time
to detect; a plugin would have the same buffer for the same reason. The
plugin path saves the in-OBS handoff (~16 ms), which is irrelevant next
to the 3-second delay.

## Rejected: OBS plugin

- Covers only OBS. Streamlabs, Twitch Studio, Zoom users get nothing.
- Native C++ in the OBS plugin SDK. Different distribution per OBS version.
- License: OBS plugins must be GPL-compatible. Our risk-analyzer code
  shipped from the Chrome-extension repo is BSL — re-licensing is a
  lawyer call.
- No auto-update story without our own update server.

## Rejected: browser/WebRTC

- `getDisplayMedia` captures the screen, but there is no standardised way
  to emit a stream back into the OS as a camera. The closest is OBS
  Browser Source plus a WebRTC bridge, which loops us back to needing
  OBS-only and adding a bunch of moving pieces.

## Consequences

- We must ship and notarise a virtual-camera driver per OS. macOS Sample
  Extension requires Apple Developer Program enrollment and notarisation —
  bureaucratic but solved problem.
- On Windows, we can either bundle our own DShow filter (full control,
  more code) or shim onto OBS Virtual Camera's already-installed driver
  (less code but adds OBS as an install dependency). We will bundle our
  own to avoid the dependency; the DShow filter is ~300 lines of COM.
- Linux v4l2loopback is a kernel module; users must install it via their
  package manager. We will detect missing module and surface a one-line
  install command (`sudo apt install v4l2loopback-dkms`).
- We never replace the wallet's signing UI — we only blur output. The
  streamer's real wallet behaviour is unchanged. This matches user
  expectation and means we cannot accidentally lock the streamer out of
  their funds.

## Open follow-ups

- Mobile (iOS Broadcast Extension, Android MediaProjection) is a
  separate product cycle, see ROADMAP Phase 7.
- Some games detect virtual cameras and may flag the stream. We will
  surface an in-app warning if we detect a flagged game (allowlist later).

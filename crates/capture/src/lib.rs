//! Platform-specific screen capture. Windows = DDA via windows-rs, macOS = ScreenCaptureKit via objc2, Linux = PipeWire via pipewire-rs.
//!
//! This crate is a stub today. Platform modules will be added in
//! Phase 1 (Windows), Phase 4 (macOS), Phase 5 (Linux) per ROADMAP.md.

#![allow(dead_code)]

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

//! Per-OS text recognition.
//!
//! Each OS exposes a high-quality OCR engine that's already on the
//! machine, so we never bundle Tesseract or another heavyweight model:
//!   - Windows: `Windows.Media.Ocr.OcrEngine` (uses the user's installed
//!     language packs; ~30 ms / 1080p frame).
//!   - macOS: Vision framework `VNRecognizeTextRequest`.
//!   - Linux: Tesseract via `leptess` (Linux ships nothing system-level,
//!     so the user installs `tesseract-ocr` from their package manager).
//!
//! All backends emit [`OcrResult`] = list of recognised lines, each
//! with the union bbox of its words. We deliberately collapse to lines
//! rather than expose per-word boxes because the secret detector matches
//! seed phrases (a *sequence* of words) — running detection per-word
//! would force re-stitching upstream.

#![allow(dead_code)]

use bw_capture::Frame;

#[derive(Debug, Clone)]
pub struct OcrLine {
    /// Recognised text, normalised by stripping repeated whitespace.
    pub text: String,
    /// Pixel bbox in the captured frame's coordinate system. Y-axis
    /// grows downward.
    pub bbox: OcrBBox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OcrBBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Default, Clone)]
pub struct OcrResult {
    pub lines: Vec<OcrLine>,
}

pub trait OcrBackend {
    fn recognise(&mut self, frame: &Frame) -> Result<OcrResult, OcrError>;
}

#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error("OCR engine init failed: {0}")]
    Init(String),
    #[error("OCR recognise failed: {0}")]
    Recognise(String),
}

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

/// Construct the default OCR backend for the host OS.
pub fn default_backend() -> Result<Box<dyn OcrBackend>, OcrError> {
    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(windows::WinRtOcr::new()?))
    }
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(macos::VisionOcr::new()?))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(Box::new(linux::TesseractOcr::new()?))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Err(OcrError::Init("no OCR backend for this OS".into()))
    }
}

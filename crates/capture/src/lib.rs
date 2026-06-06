//! Cross-platform screen capture.
//!
//! Exposes a single [`Frame`] type and a [`Capturer`] trait. The trait has
//! per-OS implementations in [`windows`], [`macos`], [`linux`]. All three
//! currently back onto the `xcap` crate, which handles the OS-specific
//! capture APIs (DDA on Windows, ScreenCaptureKit on macOS, PipeWire on
//! Linux). We still split into per-OS modules so that future revisions
//! (e.g. swapping to native `windows-rs` DDA for lower latency on
//! Windows) only touch one file.
//!
//! Frames are returned as raw BGRA8 (4 bytes per pixel, top-down). This
//! matches:
//!   - DDA's `DXGI_FORMAT_B8G8R8A8_UNORM`
//!   - `Windows.Graphics.Imaging.SoftwareBitmap`'s preferred format for
//!     `Windows.Media.Ocr` (avoids a conversion step on the OCR thread)
//!   - macOS `CVPixelBuffer` `kCVPixelFormatType_32BGRA`

#![allow(dead_code)]

use std::time::Instant;

/// A single captured frame. Owns its pixel buffer.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Raw BGRA pixels. Length is always `width * height * 4`.
    pub bgra: Vec<u8>,
    /// Time at which the frame was captured (monotonic, for buffer
    /// ordering and latency measurement).
    pub captured_at: Instant,
}

impl Frame {
    #[must_use]
    pub fn stride(&self) -> usize {
        (self.width as usize) * 4
    }
}

/// Abstraction over per-OS capture backends.
pub trait Capturer {
    /// Grab one frame from the primary display. Blocks until a frame
    /// is available (typically <16 ms at 60 Hz).
    fn grab(&mut self) -> Result<Frame, CaptureError>;
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("no display found")]
    NoDisplay,
    #[error("capture backend failed: {0}")]
    Backend(String),
}

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

pub mod window_info;

/// Construct the default capturer for the host OS.
///
/// On Windows this is [`windows::DxgiCapturer`]; on macOS,
/// [`macos::ScreenCaptureKitCapturer`]; on Linux,
/// [`linux::PipeWireCapturer`].
pub fn default_capturer() -> Result<Box<dyn Capturer>, CaptureError> {
    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(windows::DxgiCapturer::new()?))
    }
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(macos::ScreenCaptureKitCapturer::new()?))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(Box::new(linux::PipeWireCapturer::new()?))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Err(CaptureError::NoDisplay)
    }
}

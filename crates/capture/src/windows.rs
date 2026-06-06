//! Windows screen capture.
//!
//! Backed by `xcap::Monitor`, which uses the Desktop Duplication API
//! internally. We pick the primary monitor on construction and grab
//! frames on demand. The intent in Phase 1 is correctness, not 60 fps —
//! the OCR loop only needs 3–5 fps anyway.

use std::time::Instant;

use xcap::Monitor;

use crate::{CaptureError, Capturer, Frame};

pub struct DxgiCapturer {
    monitor: Monitor,
}

impl DxgiCapturer {
    pub fn new() -> Result<Self, CaptureError> {
        let monitors = Monitor::all().map_err(|e| CaptureError::Backend(e.to_string()))?;
        let monitor = monitors
            .into_iter()
            .find(|m| m.is_primary())
            .ok_or(CaptureError::NoDisplay)?;
        Ok(Self { monitor })
    }
}

impl Capturer for DxgiCapturer {
    fn grab(&mut self) -> Result<Frame, CaptureError> {
        let img = self
            .monitor
            .capture_image()
            .map_err(|e| CaptureError::Backend(e.to_string()))?;

        let width = img.width();
        let height = img.height();

        // xcap returns RGBA; convert to BGRA in place (swap R and B).
        // We pay the copy here so downstream stages — OCR, blur, encode —
        // can rely on BGRA without per-stage branches.
        let mut bgra = img.into_raw();
        debug_assert_eq!(bgra.len(), (width as usize) * (height as usize) * 4);
        for chunk in bgra.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }

        Ok(Frame {
            width,
            height,
            bgra,
            captured_at: Instant::now(),
        })
    }
}

//! macOS OCR — stub. Phase 4 wires this through Vision.

#![allow(dead_code)]

use bw_capture::Frame;

use crate::{OcrBackend, OcrError, OcrResult};

pub struct VisionOcr;

impl VisionOcr {
    pub fn new() -> Result<Self, OcrError> {
        Err(OcrError::Init("Vision OCR not yet implemented (Phase 4)".into()))
    }
}

impl OcrBackend for VisionOcr {
    fn recognise(&mut self, _frame: &Frame) -> Result<OcrResult, OcrError> {
        Err(OcrError::Recognise("Vision OCR not yet implemented".into()))
    }
}

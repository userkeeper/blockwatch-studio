//! Linux OCR — stub. Phase 5 wires this through Tesseract via `leptess`.

#![allow(dead_code)]

use bw_capture::Frame;

use crate::{OcrBackend, OcrError, OcrResult};

pub struct TesseractOcr;

impl TesseractOcr {
    pub fn new() -> Result<Self, OcrError> {
        Err(OcrError::Init("Tesseract OCR not yet implemented (Phase 5)".into()))
    }
}

impl OcrBackend for TesseractOcr {
    fn recognise(&mut self, _frame: &Frame) -> Result<OcrResult, OcrError> {
        Err(OcrError::Recognise("Tesseract OCR not yet implemented".into()))
    }
}

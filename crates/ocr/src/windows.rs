//! Windows OCR via `Windows.Media.Ocr.OcrEngine`.
//!
//! Flow:
//!   1. BGRA bytes from `Frame` → `SoftwareBitmap` (BGRA8, premultiplied).
//!   2. `OcrEngine::TryCreateFromUserProfileLanguages` once at startup,
//!      falling back to forced en-US (always present on Windows).
//!   3. `RecognizeAsync(bitmap)` per frame.
//!   4. Walk `OcrResult.Lines` → for each line, accumulate the union of
//!      its `OcrWord.BoundingRect`s into a single `OcrBBox`.
//!
//! Costs on a recent Intel iGPU at 1080p:
//!   - SoftwareBitmap allocation + memcpy: ~3 ms
//!   - RecognizeAsync (blocking via `.get()`): ~25–50 ms depending on
//!     text density
//!
//! Comfortable for a 5 fps detection loop.

use windows::core::HSTRING;
use windows::Foundation::IAsyncOperation;
use windows::Globalization::Language;
use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Storage::Streams::DataWriter;

use bw_capture::Frame;

use crate::{OcrBBox, OcrBackend, OcrError, OcrLine, OcrResult};

pub struct WinRtOcr {
    engine: OcrEngine,
}

impl WinRtOcr {
    pub fn new() -> Result<Self, OcrError> {
        // Try the user-profile engine first (uses installed language
        // packs). On systems without an OCR language pack this can return
        // an error; fall back to forcing en-US, which Windows includes by
        // default.
        let engine = match OcrEngine::TryCreateFromUserProfileLanguages() {
            Ok(e) => e,
            Err(_) => {
                let en = Language::CreateLanguage(&HSTRING::from("en-US"))
                    .map_err(|e| OcrError::Init(format!("CreateLanguage en-US: {e}")))?;
                OcrEngine::TryCreateFromLanguage(&en).map_err(|e| {
                    OcrError::Init(format!("OcrEngine TryCreateFromLanguage en-US: {e}"))
                })?
            }
        };
        Ok(Self { engine })
    }

    /// Build a WinRT `SoftwareBitmap` from raw BGRA bytes. The bitmap
    /// owns its pixel buffer (the writer flushes into it).
    fn bgra_to_bitmap(frame: &Frame) -> Result<SoftwareBitmap, OcrError> {
        let len = frame.bgra.len() as u32;

        // We use a DataWriter → IBuffer → SoftwareBitmap::CreateCopyFromBuffer
        // path because it's the most concise way to feed raw bytes into
        // a SoftwareBitmap without unsafe interop. Cost: one extra copy
        // we'd avoid with `LockBuffer` — acceptable in Phase 1.
        let writer = DataWriter::new().map_err(|e| OcrError::Recognise(e.to_string()))?;
        writer
            .WriteBytes(&frame.bgra)
            .map_err(|e| OcrError::Recognise(e.to_string()))?;
        let ibuf = writer
            .DetachBuffer()
            .map_err(|e| OcrError::Recognise(e.to_string()))?;
        debug_assert_eq!(ibuf.Length().unwrap_or(0), len);

        SoftwareBitmap::CreateCopyFromBuffer(
            &ibuf,
            BitmapPixelFormat::Bgra8,
            frame.width as i32,
            frame.height as i32,
        )
        .map_err(|e| OcrError::Recognise(format!("CreateCopyFromBuffer: {e}")))
    }
}

impl OcrBackend for WinRtOcr {
    fn recognise(&mut self, frame: &Frame) -> Result<OcrResult, OcrError> {
        let bitmap = Self::bgra_to_bitmap(frame)?;

        // `RecognizeAsync` returns an IAsyncOperation; we block on it
        // because we're already on a dedicated OCR thread (the frame
        // grabber owns the rendering / main thread).
        let op: IAsyncOperation<windows::Media::Ocr::OcrResult> = self
            .engine
            .RecognizeAsync(&bitmap)
            .map_err(|e| OcrError::Recognise(format!("RecognizeAsync init: {e}")))?;
        let result = op
            .get()
            .map_err(|e| OcrError::Recognise(format!("RecognizeAsync result: {e}")))?;

        let mut lines = Vec::new();
        let line_collection = result
            .Lines()
            .map_err(|e| OcrError::Recognise(format!("Lines(): {e}")))?;
        let line_count = line_collection
            .Size()
            .map_err(|e| OcrError::Recognise(e.to_string()))?;

        for i in 0..line_count {
            let line = line_collection
                .GetAt(i)
                .map_err(|e| OcrError::Recognise(format!("Lines.GetAt({i}): {e}")))?;
            let text_h = line
                .Text()
                .map_err(|e| OcrError::Recognise(format!("Line.Text: {e}")))?;
            let text: String = text_h.to_string();

            // Union of word bboxes → line bbox.
            let mut x0 = u32::MAX;
            let mut y0 = u32::MAX;
            let mut x1 = 0u32;
            let mut y1 = 0u32;
            let words = line
                .Words()
                .map_err(|e| OcrError::Recognise(format!("Line.Words: {e}")))?;
            let word_count = words
                .Size()
                .map_err(|e| OcrError::Recognise(e.to_string()))?;
            for j in 0..word_count {
                let w = words
                    .GetAt(j)
                    .map_err(|e| OcrError::Recognise(format!("Words.GetAt({j}): {e}")))?;
                let r = w
                    .BoundingRect()
                    .map_err(|e| OcrError::Recognise(format!("Word.BoundingRect: {e}")))?;
                let wx0 = r.X.max(0.0) as u32;
                let wy0 = r.Y.max(0.0) as u32;
                let wx1 = (r.X + r.Width).max(0.0) as u32;
                let wy1 = (r.Y + r.Height).max(0.0) as u32;
                x0 = x0.min(wx0);
                y0 = y0.min(wy0);
                x1 = x1.max(wx1);
                y1 = y1.max(wy1);
            }
            if x0 == u32::MAX || word_count == 0 {
                // Empty / punctuation-only line; skip.
                continue;
            }

            lines.push(OcrLine {
                text,
                bbox: OcrBBox {
                    x: x0,
                    y: y0,
                    w: x1.saturating_sub(x0),
                    h: y1.saturating_sub(y0),
                },
            });
        }

        Ok(OcrResult { lines })
    }
}

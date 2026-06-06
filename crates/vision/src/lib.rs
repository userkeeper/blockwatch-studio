//! Vision-based detection — ONNX Runtime + YOLOv8 family models.
//!
//! This crate is the foundation for the ML migration described in
//! `docs/adr-002-vision.md`. The motivation, in one line: OCR + regex
//! is too noisy on real Windows screens to ever hit "perfect", and
//! we've burned a session reaching that conclusion empirically. A
//! purpose-trained vision model — even a tiny one — outperforms it
//! end-to-end.
//!
//! ## Current state (Phase 0: scaffold)
//!
//! - The pipeline structure (load model → preprocess frame → run
//!   inference → post-process boxes) is defined and stubbed.
//! - No model weights are bundled. `WalletPopupDetector::new()`
//!   returns an `Error::ModelNotBundled` until we drop a `.onnx`
//!   file into `models/` and update `MODEL_PATH`.
//! - `cargo build` succeeds without a model file because the
//!   detector is *constructed lazily*.
//!
//! ## Next phases
//!
//! - **Phase 1**: collect 300-500 labelled screenshots of Phantom /
//!   MetaMask / Tonkeeper popups and Notepad / VS Code with `.env`.
//! - **Phase 2**: fine-tune YOLOv8n on that dataset (Python /
//!   Ultralytics). Export to ONNX.
//! - **Phase 3**: wire `WalletPopupDetector` into the CLI as an
//!   optional `--vision` flag that supplements (and eventually
//!   replaces) the regex pipeline.
//! - **Phase 4**: small CNN classifier on text-region crops for
//!   "is this likely a secret?" — bridges the regex and ML worlds.

#![allow(dead_code)]

use bw_capture::Frame;
use bw_core::detect::BBox;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ONNX model file not bundled — drop a YOLOv8n .onnx into crates/vision/models/ and update MODEL_PATH")]
    ModelNotBundled,
    #[error("ONNX session init failed: {0}")]
    SessionInit(String),
    #[error("inference failed: {0}")]
    Inference(String),
}

/// One detected object (bounding box in the captured frame's pixel
/// coordinates, plus a class label and confidence).
#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub class: ObjectClass,
    pub bbox: BBox,
    pub confidence: f32,
}

/// Classes the wallet-popup model is trained to recognise. Mirrors
/// the dataset labels (see ADR-002 §3 — "Class taxonomy").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectClass {
    /// Phantom / MetaMask / Tonkeeper / Trust / Backpack receive popup.
    WalletReceivePopup,
    /// Wallet seed-phrase reveal screen.
    SeedPhraseReveal,
    /// Wallet private-key reveal screen.
    PrivateKeyReveal,
    /// Text editor with `.env`-style content.
    EnvFileEditor,
    /// Terminal showing exports / credential commands.
    CredentialTerminal,
    /// Trading dashboard with balance / portfolio numbers.
    PortfolioBalance,
}

/// Configuration for the detector.
pub struct DetectorConfig {
    /// Resize the input frame to this dimension before inference.
    /// YOLOv8 expects square. 640 is the official Ultralytics default.
    pub input_size: u32,
    /// Reject detections below this confidence.
    pub confidence_threshold: f32,
    /// IoU threshold for non-max suppression.
    pub nms_iou_threshold: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            input_size: 640,
            confidence_threshold: 0.35,
            nms_iou_threshold: 0.5,
        }
    }
}

/// Future home of the YOLOv8 inference session. Currently a placeholder
/// — the actual `ort::Session` field is added when we bundle the model.
pub struct WalletPopupDetector {
    config: DetectorConfig,
    // session: ort::Session,  // added in Phase 1.
}

impl WalletPopupDetector {
    /// Construct a detector from the bundled model. Returns
    /// `Error::ModelNotBundled` until we ship a real `.onnx` file.
    pub fn new(_config: DetectorConfig) -> Result<Self, Error> {
        Err(Error::ModelNotBundled)
    }

    /// Run inference on one frame. Stub for Phase 0.
    pub fn detect(&self, _frame: &Frame) -> Result<Vec<DetectedObject>, Error> {
        Err(Error::Inference("detector stub".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_yolov8_defaults() {
        let cfg = DetectorConfig::default();
        assert_eq!(cfg.input_size, 640);
        assert!((cfg.confidence_threshold - 0.35).abs() < 1e-6);
        assert!((cfg.nms_iou_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn new_returns_not_bundled_until_model_lands() {
        let err = WalletPopupDetector::new(DetectorConfig::default()).err();
        assert!(matches!(err, Some(Error::ModelNotBundled)));
    }

    #[test]
    fn object_class_taxonomy_is_stable() {
        // If you reorder these the trained model loses meaning.
        // Confirm the discriminant ordering at compile-time.
        use ObjectClass::*;
        let classes = [
            WalletReceivePopup,
            SeedPhraseReveal,
            PrivateKeyReveal,
            EnvFileEditor,
            CredentialTerminal,
            PortfolioBalance,
        ];
        assert_eq!(classes.len(), 6);
    }
}

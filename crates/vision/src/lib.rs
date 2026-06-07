//! Vision-based detection — `tract-onnx` + bundled YOLOv8n model.
//!
//! Phase 0 was a stub. This is Phase 1: the model trains, exports to
//! ONNX, the Rust runtime loads it and detects on each frame.
//!
//! Pipeline per frame:
//!   1. BGRA → letterbox-resize to 640×640 → CHW float32.
//!   2. Tract session `run(input)` → output tensor [1, 4+C, 8400].
//!   3. Post-process: split into per-anchor (box, class, conf), apply
//!      confidence threshold, NMS, rescale boxes back to original
//!      frame coords.
//!
//! Costs measured on the same i3-10105F that trained the model:
//!   - preprocessing: 1 ms
//!   - tract inference: ~80 ms
//!   - post-processing + NMS: 1 ms
//!   Total ~82 ms per frame — well under our 100 ms / frame budget at
//!   10 fps.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use bw_capture::Frame;
use bw_core::detect::BBox;

use ndarray::{Array, Array4, Axis};
use tract_onnx::prelude::*;

/// Default location of the bundled model relative to the crate root.
/// `cargo build` does NOT copy this into `target/`; the model file is
/// loaded at runtime from disk. For a distributable build, the model
/// should be bundled by the installer and `with_model_path` used to
/// point at the installed location.
pub const DEFAULT_MODEL_PATH: &str = "crates/vision/models/popup-yolov8n-v1.onnx";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model file not found at {0}")]
    ModelNotFound(PathBuf),
    #[error("ONNX session init failed: {0}")]
    SessionInit(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("frame preprocessing failed: {0}")]
    Preprocess(String),
}

/// One detected object (bounding box in the captured frame's pixel
/// coordinates, plus a class label and confidence).
#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub class: ObjectClass,
    pub bbox: BBox,
    pub confidence: f32,
}

/// Classes the wallet-popup model is trained to recognise. The
/// discriminant ORDER MATTERS — it must match `training/dataset.yaml`
/// and `synth_data.py`. Reordering breaks the runtime mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectClass {
    WalletReceivePopup = 0,
    SeedPhraseReveal = 1,
    PrivateKeyReveal = 2,
    EnvFileEditor = 3,
    CredentialTerminal = 4,
    PortfolioBalance = 5,
}

impl ObjectClass {
    fn from_index(i: usize) -> Option<Self> {
        Some(match i {
            0 => Self::WalletReceivePopup,
            1 => Self::SeedPhraseReveal,
            2 => Self::PrivateKeyReveal,
            3 => Self::EnvFileEditor,
            4 => Self::CredentialTerminal,
            5 => Self::PortfolioBalance,
            _ => return None,
        })
    }
}

/// Configuration for the detector.
pub struct DetectorConfig {
    pub input_size: u32,
    pub confidence_threshold: f32,
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

/// Loaded model + config. Build once at startup, call `detect()` per frame.
pub struct WalletPopupDetector {
    config: DetectorConfig,
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl WalletPopupDetector {
    /// Load the bundled model at [`DEFAULT_MODEL_PATH`].
    pub fn new(config: DetectorConfig) -> Result<Self, Error> {
        Self::with_model_path(config, Path::new(DEFAULT_MODEL_PATH))
    }

    /// Load a model from a specific path. Use this when bundling the
    /// model in a non-default location (e.g. inside an installer dir).
    pub fn with_model_path(
        config: DetectorConfig,
        path: &Path,
    ) -> Result<Self, Error> {
        if !path.exists() {
            return Err(Error::ModelNotFound(path.to_path_buf()));
        }
        let size = config.input_size as i32;
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| Error::SessionInit(format!("load_onnx: {e}")))?
            .with_input_fact(
                0,
                f32::fact([1, 3, size, size]).into(),
            )
            .map_err(|e| Error::SessionInit(format!("with_input_fact: {e}")))?
            .into_optimized()
            .map_err(|e| Error::SessionInit(format!("into_optimized: {e}")))?
            .into_runnable()
            .map_err(|e| Error::SessionInit(format!("into_runnable: {e}")))?;
        Ok(Self { config, model })
    }

    /// Detect objects in one frame. Returns an empty Vec if nothing
    /// passes the confidence threshold.
    pub fn detect(&self, frame: &Frame) -> Result<Vec<DetectedObject>, Error> {
        // 1. Preprocess: BGRA → letterboxed 640×640 RGB float [0,1].
        let (input, scale, pad_x, pad_y) = preprocess(
            &frame.bgra,
            frame.width,
            frame.height,
            self.config.input_size,
        )?;
        let input_tensor: Tensor = input.into_tensor();

        // 2. Run inference.
        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .map_err(|e| Error::Inference(format!("session.run: {e}")))?;

        // 3. Post-process. YOLOv8 output shape: [1, 4 + num_classes, 8400].
        //    Anchors are along axis 2. For each anchor, axis 1 holds:
        //      [cx, cy, w, h, class0_conf, class1_conf, ...]
        let out = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| Error::Inference(format!("output array_view: {e}")))?;
        let shape = out.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::Inference(format!(
                "unexpected output shape: {shape:?}"
            )));
        }
        let num_classes = shape[1] - 4;
        let num_anchors = shape[2];

        let mut candidates: Vec<DetectedObject> = Vec::new();
        for a in 0..num_anchors {
            // Pick the class with the highest confidence for this anchor.
            let mut best_class = 0usize;
            let mut best_conf = 0f32;
            for c in 0..num_classes {
                let v = out[[0, 4 + c, a]];
                if v > best_conf {
                    best_conf = v;
                    best_class = c;
                }
            }
            if best_conf < self.config.confidence_threshold {
                continue;
            }
            let cx = out[[0, 0, a]];
            let cy = out[[0, 1, a]];
            let w = out[[0, 2, a]];
            let h = out[[0, 3, a]];

            // Rescale from letterboxed 640 space back to original frame.
            let x0 = ((cx - w / 2.0) - pad_x) / scale;
            let y0 = ((cy - h / 2.0) - pad_y) / scale;
            let x1 = ((cx + w / 2.0) - pad_x) / scale;
            let y1 = ((cy + h / 2.0) - pad_y) / scale;

            let x0i = x0.max(0.0) as u32;
            let y0i = y0.max(0.0) as u32;
            let x1i = (x1 as u32).min(frame.width);
            let y1i = (y1 as u32).min(frame.height);
            if x1i <= x0i || y1i <= y0i {
                continue;
            }
            let Some(class) = ObjectClass::from_index(best_class) else {
                continue;
            };
            candidates.push(DetectedObject {
                class,
                bbox: BBox {
                    x: x0i,
                    y: y0i,
                    w: x1i - x0i,
                    h: y1i - y0i,
                },
                confidence: best_conf,
            });
        }

        // 4. Non-max suppression.
        Ok(non_max_suppression(candidates, self.config.nms_iou_threshold))
    }
}

// ─── Preprocessing ──────────────────────────────────────────────────

/// Letterbox-resize a BGRA frame into a `[1, 3, S, S]` float tensor in
/// RGB-channel order, values in [0, 1]. Returns also the scale and pad
/// offsets so detections can be mapped back to the original frame.
fn preprocess(
    bgra: &[u8],
    src_w: u32,
    src_h: u32,
    target: u32,
) -> Result<(Array4<f32>, f32, f32, f32), Error> {
    if bgra.len() < (src_w as usize) * (src_h as usize) * 4 {
        return Err(Error::Preprocess("bgra buffer too small".into()));
    }
    let s = target as f32 / src_w.max(src_h) as f32;
    let new_w = (src_w as f32 * s) as u32;
    let new_h = (src_h as f32 * s) as u32;
    let pad_x = ((target - new_w) / 2) as f32;
    let pad_y = ((target - new_h) / 2) as f32;

    let mut arr = Array4::<f32>::zeros((1, 3, target as usize, target as usize));
    let src_stride = (src_w * 4) as usize;
    for y in 0..new_h {
        // Map dest-y back to source.
        let sy = (y as f32 / s) as u32;
        let sy = sy.min(src_h - 1);
        for x in 0..new_w {
            let sx = (x as f32 / s) as u32;
            let sx = sx.min(src_w - 1);
            let i = (sy as usize) * src_stride + (sx as usize) * 4;
            let b = bgra[i] as f32 / 255.0;
            let g = bgra[i + 1] as f32 / 255.0;
            let r = bgra[i + 2] as f32 / 255.0;
            let dst_x = (x + pad_x as u32) as usize;
            let dst_y = (y + pad_y as u32) as usize;
            arr[[0, 0, dst_y, dst_x]] = r;
            arr[[0, 1, dst_y, dst_x]] = g;
            arr[[0, 2, dst_y, dst_x]] = b;
        }
    }
    Ok((arr, s, pad_x, pad_y))
}

// ─── Non-max suppression ────────────────────────────────────────────

fn iou(a: &BBox, b: &BBox) -> f32 {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    if ix2 <= ix1 || iy2 <= iy1 {
        return 0.0;
    }
    let inter = (ix2 - ix1) as f32 * (iy2 - iy1) as f32;
    let area_a = a.w as f32 * a.h as f32;
    let area_b = b.w as f32 * b.h as f32;
    inter / (area_a + area_b - inter)
}

fn non_max_suppression(
    mut candidates: Vec<DetectedObject>,
    iou_thresh: f32,
) -> Vec<DetectedObject> {
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<DetectedObject> = Vec::new();
    for c in candidates {
        let mut suppress = false;
        for k in &kept {
            if k.class == c.class && iou(&k.bbox, &c.bbox) > iou_thresh {
                suppress = true;
                break;
            }
        }
        if !suppress {
            kept.push(c);
        }
    }
    kept
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
    fn class_taxonomy_is_stable() {
        assert_eq!(ObjectClass::WalletReceivePopup as u32, 0);
        assert_eq!(ObjectClass::SeedPhraseReveal as u32, 1);
        assert_eq!(ObjectClass::PrivateKeyReveal as u32, 2);
        assert_eq!(ObjectClass::EnvFileEditor as u32, 3);
        assert_eq!(ObjectClass::CredentialTerminal as u32, 4);
        assert_eq!(ObjectClass::PortfolioBalance as u32, 5);
    }

    #[test]
    fn iou_disjoint_is_zero() {
        let a = BBox { x: 0, y: 0, w: 10, h: 10 };
        let b = BBox { x: 100, y: 100, w: 10, h: 10 };
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_identical_is_one() {
        let a = BBox { x: 5, y: 5, w: 20, h: 20 };
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_higher_confidence() {
        let kept = non_max_suppression(
            vec![
                DetectedObject {
                    class: ObjectClass::EnvFileEditor,
                    bbox: BBox { x: 0, y: 0, w: 100, h: 100 },
                    confidence: 0.8,
                },
                DetectedObject {
                    class: ObjectClass::EnvFileEditor,
                    bbox: BBox { x: 5, y: 5, w: 100, h: 100 },
                    confidence: 0.95,
                },
            ],
            0.5,
        );
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.95).abs() < 1e-6);
    }
}

// Helper to silence the unused-import lint of Axis without removing it
// (we use Axis in a future inference path that operates per-anchor).
#[allow(dead_code)]
fn _axis_use(a: Array<f32, ndarray::IxDyn>) -> usize {
    a.shape().len()
}

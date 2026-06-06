//! BlockWatch Studio — core detection + buffer logic.
//!
//! This crate is **platform-agnostic**. It contains:
//!   - `detect`: secret-scanning logic (BIP-39, hex keys, WIF, xpub, base58,
//!     QR-decode wiring), with bounding-box output for the compositor.
//!   - `bip39`: word lists in every language the BIP-39 spec defines.
//!   - `buffer`: ring-buffer of frames with attached detection metadata.
//!   - `blur`: gaussian-blur compositor abstraction (real GPU work lives
//!     in `bw-capture` / `bw-virtualcam` which feed this crate frames).
//!
//! Per-OS crates (`bw-capture`, `bw-ocr`, `bw-virtualcam`) depend on
//! this crate but never the other way around.

pub mod bip39;
pub mod buffer;
pub mod detect;
pub mod frame_diff;

// Future modules — stubbed in PRs to come:
// pub mod blur;

/// Crate-level result type. Per-module errors flow up as `Error`.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("detection internal error: {0}")]
    Detect(String),
}

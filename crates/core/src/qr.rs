//! QR-code detection + filtering.
//!
//! On a stream, the highest-bandwidth secret leak is not text — it's
//! the QR code a streamer briefly shows so a viewer can scan their
//! wallet address. That QR is the entire signing key from the
//! attacker's perspective: scan it, watch the address, drain it the
//! moment funds arrive. We can't OCR a QR (it's not text), so we
//! decode it directly and treat any QR whose payload looks like a
//! crypto address, BIP-21 URI, or seed phrase as sensitive.
//!
//! Backed by `rqrr` (pure Rust, no native deps, ~30 ms / 1080p frame
//! on a recent laptop). The detector is robust to perspective and
//! moderate blur, which matters because streamers don't hold their
//! phones steady.

use once_cell::sync::Lazy;
use regex::Regex;
use rqrr::PreparedImage;

use crate::detect::BBox;

/// Result of detecting one QR. The bbox is axis-aligned in screen
/// pixels; `payload` is the decoded text and `kind` is our
/// classification of what kind of crypto-secret (if any) it carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrHit {
    pub bbox: BBox,
    pub payload: String,
    pub kind: QrKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrKind {
    /// Looks like a bare crypto address — EVM, TON, Solana, Tron, BTC.
    Address,
    /// `bitcoin:`, `ethereum:`, `ton://transfer/…` etc.
    PaymentUri,
    /// Looks like a 12/24-word BIP-39 phrase (some "paper wallets" do this).
    SeedPhrase,
    /// Anything else — URL, contact info, Wi-Fi, etc. We do NOT blur
    /// these by default but a stricter mode could.
    Unknown,
}

/// Scan a BGRA buffer for QR codes. Returns one [`QrHit`] per
/// successfully decoded code, including ones whose payload we don't
/// classify as a secret — the caller decides whether to act on
/// `QrKind::Unknown`.
///
/// Cost: ~30 ms / 1080p on a recent laptop. Wrap with the frame-diff
/// gate at the call site so we don't run on every frame.
#[must_use]
pub fn scan_qrs_bgra(bgra: &[u8], width: u32, height: u32) -> Vec<QrHit> {
    if bgra.len() < (width * height * 4) as usize {
        return Vec::new();
    }

    // rqrr operates on 8-bit grayscale. We feed it a closure that
    // converts BGRA → luma on-the-fly using the ITU-R BT.601 formula
    // (Y' = 0.299R + 0.587G + 0.114B) — same coefficients used by
    // OpenCV / Pillow, which matters because QR scanners are
    // calibrated against this exact luma curve.
    //
    // Using `prepare_from_greyscale` instead of building an `image`
    // crate ImageBuffer first means bw-core avoids depending on
    // `image` directly (only `rqrr` brings it in transitively).
    let stride = (width * 4) as usize;
    let mut prepared = PreparedImage::prepare_from_greyscale(
        width as usize,
        height as usize,
        |x, y| {
            let i = y * stride + x * 4;
            let b = u32::from(bgra[i]);
            let g = u32::from(bgra[i + 1]);
            let r = u32::from(bgra[i + 2]);
            ((299 * r + 587 * g + 114 * b) / 1000) as u8
        },
    );
    let grids = prepared.detect_grids();

    let mut hits = Vec::new();
    for grid in grids {
        let bbox = bbox_from_corners(&grid.bounds, width, height);
        match grid.decode() {
            Ok((_meta, payload)) => {
                let kind = classify(&payload);
                hits.push(QrHit { bbox, payload, kind });
            }
            Err(_) => {
                // Locator pattern detected but content unreadable
                // (motion blur, low res, partial occlusion). Still
                // treat as sensitive — better to blur an unknown QR
                // than to leak a wallet address by gambling on decode.
                hits.push(QrHit {
                    bbox,
                    payload: String::new(),
                    kind: QrKind::Unknown,
                });
            }
        }
    }
    hits
}

/// Axis-aligned bounding box covering all four QR corner points.
/// Clamped to image bounds.
fn bbox_from_corners(corners: &[rqrr::Point; 4], width: u32, height: u32) -> BBox {
    let xs = corners.iter().map(|p| p.x).collect::<Vec<i32>>();
    let ys = corners.iter().map(|p| p.y).collect::<Vec<i32>>();
    let x0 = xs.iter().copied().min().unwrap_or(0).max(0) as u32;
    let y0 = ys.iter().copied().min().unwrap_or(0).max(0) as u32;
    let x1 = xs.iter().copied().max().unwrap_or(0).max(0) as u32;
    let y1 = ys.iter().copied().max().unwrap_or(0).max(0) as u32;
    BBox {
        x: x0.min(width),
        y: y0.min(height),
        w: x1.saturating_sub(x0).min(width.saturating_sub(x0)),
        h: y1.saturating_sub(y0).min(height.saturating_sub(y0)),
    }
}

// ─── Payload classification ─────────────────────────────────────────

static EVM_ADDR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^0x[a-fA-F0-9]{40}$").expect("regex"));
/// Sui and Aptos both use 32-byte addresses serialised as `0x` + 64
/// hex chars. Distinguishing the two from the payload alone is not
/// possible (and not needed — they're both sensitive). Move/Walrus
/// object IDs use the same encoding; if a streamer's QR holds one of
/// those we'd rather over-blur than leak.
static SUI_APTOS_ADDR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^0x[a-fA-F0-9]{64}$").expect("regex"));
static TON_ADDR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?:EQ|UQ|Ef|Uf|kQ|kf|0Q|0f)[A-Za-z0-9_\-]{46}$").expect("regex"));
static SOL_ADDR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$").expect("regex"));
static TRON_ADDR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^T[1-9A-HJ-NP-Za-km-z]{33}$").expect("regex"));
static BTC_ADDR: Lazy<Regex> = Lazy::new(|| {
    // Covers legacy (P2PKH/P2SH) and bech32 (P2WPKH/P2TR). Tightish
    // bounds rather than full BIP-32 checksum validation.
    Regex::new(r"^(?:bc1[a-z0-9]{38,62}|[13][1-9A-HJ-NP-Za-km-z]{25,34})$").expect("regex")
});
static URI_PAYMENT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:bitcoin|ethereum|tron|ton|solana|web3|ethereum_uri):").expect("regex")
});
/// Any whitespace-separated string of 12, 15, 18, 21, or 24 lowercase
/// English-only words. We don't actually validate every word against
/// the BIP-39 set here — that's `crate::bip39` — but the shape alone
/// is rare enough to flag.
static SEED_PHRASE_SHAPE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:[a-z]+\s+){11,23}[a-z]+$").expect("regex")
});

fn classify(s: &str) -> QrKind {
    let trimmed = s.trim();
    if EVM_ADDR.is_match(trimmed)
        || SUI_APTOS_ADDR.is_match(trimmed)
        || TON_ADDR.is_match(trimmed)
        || TRON_ADDR.is_match(trimmed)
        || BTC_ADDR.is_match(trimmed)
        || SOL_ADDR.is_match(trimmed)
    {
        return QrKind::Address;
    }
    if URI_PAYMENT.is_match(trimmed) {
        return QrKind::PaymentUri;
    }
    if SEED_PHRASE_SHAPE.is_match(trimmed) {
        // Cheap shape match — caller can chase up with `bip39::is_word_english`
        // on each token if it wants 100% certainty.
        let words = trimmed.split_whitespace().count();
        if matches!(words, 12 | 15 | 18 | 21 | 24) {
            return QrKind::SeedPhrase;
        }
    }
    QrKind::Unknown
}

/// True if the kind warrants a blur. Used by the CLI / record loop
/// to decide which QRs to treat as sticky hits.
#[must_use]
pub fn is_sensitive(kind: QrKind) -> bool {
    matches!(
        kind,
        QrKind::Address | QrKind::PaymentUri | QrKind::SeedPhrase
    )
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_evm_address() {
        let k = classify("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2");
        assert_eq!(k, QrKind::Address);
        assert!(is_sensitive(k));
    }

    #[test]
    fn classify_sui_address() {
        // Real Sui address from a fresh wallet (32 bytes / 64 hex).
        let k = classify(
            "0x817148b27231e75fa24be2a734029ca6bbbc825117e4797695aac46e904ad526",
        );
        assert_eq!(k, QrKind::Address);
        assert!(is_sensitive(k));
    }

    #[test]
    fn classify_aptos_address() {
        // Same shape as Sui — 0x + 64 hex.
        let k = classify(
            "0xa1b2c3d4e5f6789012345678901234567890abcdef0123456789abcdef012345",
        );
        assert_eq!(k, QrKind::Address);
    }

    #[test]
    fn classify_ton_address() {
        let k = classify("EQCD39VS5jcptHL8vMjEXrzGaRcCVYto7HUn4bpAOg8xqB2N");
        assert_eq!(k, QrKind::Address);
    }

    #[test]
    fn classify_tron_address() {
        let k = classify("TXYZopH7uYBzWvJ4drHN8H4P9oeKqZqQ8a");
        assert_eq!(k, QrKind::Address);
    }

    #[test]
    fn classify_btc_bech32() {
        let k = classify("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq");
        assert_eq!(k, QrKind::Address);
    }

    #[test]
    fn classify_payment_uri() {
        let k = classify("ethereum:0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2?value=1e18");
        assert_eq!(k, QrKind::PaymentUri);
        assert!(is_sensitive(k));
    }

    #[test]
    fn classify_seed_phrase_shape() {
        let k = classify(
            "abandon ability able about above absent absorb abstract absurd abuse access accident",
        );
        assert_eq!(k, QrKind::SeedPhrase);
    }

    #[test]
    fn classify_unknown_url_is_not_sensitive() {
        let k = classify("https://example.com");
        assert_eq!(k, QrKind::Unknown);
        assert!(!is_sensitive(k));
    }

    #[test]
    fn classify_unknown_wifi_payload() {
        let k = classify("WIFI:T:WPA;S:MyNetwork;P:hunter2;;");
        assert_eq!(k, QrKind::Unknown);
        assert!(!is_sensitive(k));
    }

    #[test]
    fn empty_buffer_returns_no_hits() {
        let hits = scan_qrs_bgra(&[], 0, 0);
        assert!(hits.is_empty());
    }

    #[test]
    fn buffer_too_small_returns_no_hits() {
        // Claim 100×100 but provide far fewer bytes.
        let bytes = vec![0u8; 10];
        let hits = scan_qrs_bgra(&bytes, 100, 100);
        assert!(hits.is_empty());
    }
}

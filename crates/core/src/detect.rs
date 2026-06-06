//! Secret-scanning detectors.
//!
//! Each detector takes a slice of recognised text (from the OCR layer) plus
//! the bounding box that text occupies on screen, and returns zero or more
//! [`Hit`]s. The compositor blurs each `Hit::bbox` on the frame that goes
//! out to the virtual camera.
//!
//! All detectors are **synchronous, allocation-light, and pure** so they can
//! run inside a tight 5 fps loop on the OCR thread without contention.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::bip39;

/// Pixel-space bounding box on the captured frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// What kind of secret we matched. Tracked separately so the UI can
/// show per-type counters ("Blocked: 2 seed phrases, 1 private key").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretKind {
    /// BIP-39 mnemonic — 12, 15, 18, 21 or 24 words.
    SeedPhrase,
    /// Bare 64-char hex (or `0x` + 64 hex).
    HexPrivateKey,
    /// Wallet Import Format — base58, ~51 chars, starts with `5`, `K`, or `L`.
    Wif,
    /// Extended public/private keys — `xprv`, `xpub`, `ypub`, `zpub`, etc.
    ExtendedKey,
    /// Solana base58 secret key — 64 char base58.
    SolanaKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub kind: SecretKind,
    pub bbox: BBox,
}

/// Public entry: scan a single OCR text+bbox pair and return every hit.
///
/// The OCR layer typically calls this once per recognised text region.
/// The cost is dominated by the BIP-39 word lookup; everything else is
/// regex.
pub fn scan(text: &str, bbox: BBox) -> Vec<Hit> {
    let mut out = Vec::new();

    if let Some(_run) = find_bip39_run(text) {
        // Seed phrase wins — if it's there, the whole bbox is treated as
        // sensitive even if a substring also matches another detector.
        out.push(Hit {
            kind: SecretKind::SeedPhrase,
            bbox,
        });
        return out;
    }

    if HEX_PRIV_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::HexPrivateKey,
            bbox,
        });
    }
    if WIF_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::Wif,
            bbox,
        });
    }
    if XKEY_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::ExtendedKey,
            bbox,
        });
    }
    if SOL_KEY_RE.is_match(text) {
        // Sol secret keys are 64-char base58. They overlap visually with
        // long random strings, so we de-prioritise this detector by only
        // matching when it sits alone in the bbox text (no spaces).
        if !text.contains(' ') {
            out.push(Hit {
                kind: SecretKind::SolanaKey,
                bbox,
            });
        }
    }

    out
}

/// Find a run of ≥`MIN_RUN` consecutive BIP-39 words in the text.
/// Returns the start..end token indices of the longest run, or `None`.
///
/// BIP-39 phrases are 12, 15, 18, 21 or 24 words long. We use a run of
/// 8 as the trigger because:
///   - The chance of 8 unrelated random English words ALL appearing in
///     the 2048-word list and ALL being adjacent is vanishingly small
///     (~1 in 5^8 ≈ 1 in 400k *for unrelated text*).
///   - OCR sometimes drops a word from a 12-word seed; 8 leaves us
///     resilient to two drops in a row.
const MIN_RUN: usize = 8;

fn find_bip39_run(text: &str) -> Option<std::ops::Range<usize>> {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() < MIN_RUN {
        return None;
    }

    let mut best: Option<std::ops::Range<usize>> = None;
    let mut run_start: Option<usize> = None;
    for (i, w) in words.iter().enumerate() {
        if bip39::is_word_english(&w.to_lowercase()) {
            run_start.get_or_insert(i);
        } else if let Some(s) = run_start.take() {
            let len = i - s;
            if len >= MIN_RUN {
                let r = s..i;
                if best.as_ref().map_or(true, |b| r.len() > b.len()) {
                    best = Some(r);
                }
            }
        }
    }
    if let Some(s) = run_start {
        let len = words.len() - s;
        if len >= MIN_RUN {
            let r = s..words.len();
            if best.as_ref().map_or(true, |b| r.len() > b.len()) {
                best = Some(r);
            }
        }
    }
    best
}

// ─── Regex detectors ───────────────────────────────────────────────────
//
// Compiled once at first use. The patterns are duplicated from the Chrome
// extension's streamMode/patterns.ts so behaviour stays identical across
// surfaces.

static HEX_PRIV_RE: Lazy<Regex> = Lazy::new(|| {
    // 64 hex chars, optionally prefixed with `0x`. Word-boundary so we
    // don't match in the middle of a longer hex blob (which would just be
    // base data, not a key).
    Regex::new(r"(?i)\b(?:0x)?[0-9a-f]{64}\b").expect("static regex compiles")
});

static WIF_RE: Lazy<Regex> = Lazy::new(|| {
    // Bitcoin WIF: starts with 5 / K / L, base58 alphabet, 50–51 trailing
    // chars. Total 51–52 chars.
    Regex::new(r"\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b").expect("static regex compiles")
});

static XKEY_RE: Lazy<Regex> = Lazy::new(|| {
    // BIP-32 extended keys: xprv, xpub, ypub, zpub, tprv, tpub
    // 4-char prefix + 107–108 base58 chars (111–112 total).
    Regex::new(r"\b(?:xprv|xpub|ypub|zpub|tprv|tpub)[1-9A-HJ-NP-Za-km-z]{107,108}\b")
        .expect("static regex compiles")
});

static SOL_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    // Solana secret keys serialised as base58 are 88 chars when the
    // 64-byte ed25519 key pair is encoded. Some tools show 64 chars
    // (private half only). We match both.
    Regex::new(r"\b[1-9A-HJ-NP-Za-km-z]{64,88}\b").expect("static regex compiles")
});

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BBOX: BBox = BBox {
        x: 10,
        y: 20,
        w: 800,
        h: 50,
    };

    #[test]
    fn detects_full_bip39_phrase() {
        let phrase = "abandon ability able about above absent absorb abstract absurd abuse access accident";
        let hits = scan(phrase, TEST_BBOX);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kind, SecretKind::SeedPhrase);
    }

    #[test]
    fn detects_short_bip39_run_inside_paragraph() {
        // 9-word run hidden in a longer sentence — still triggers.
        let text = "Recovery: abandon ability able about above absent absorb abstract absurd. Done.";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::SeedPhrase));
    }

    #[test]
    fn ignores_short_runs_of_bip39_words() {
        // Only 3 BIP-39 words appearing by chance.
        let text = "the abandon over time is also good";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.is_empty());
    }

    #[test]
    fn detects_hex_private_key() {
        let key = "Key: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let hits = scan(key, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::HexPrivateKey));
    }

    #[test]
    fn detects_0x_prefixed_hex_key() {
        let key = "0xe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let hits = scan(key, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::HexPrivateKey));
    }

    #[test]
    fn detects_wif_key() {
        // Real-world example test vector.
        let key = "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ";
        let hits = scan(key, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::Wif));
    }

    #[test]
    fn detects_xprv() {
        let key = "xprv9s21ZrQH143K3QTDL4LXw2F7HEK3wJUD2nW2nRk4stbPy6cq3jPPqjiChkVvvNKmPGJxWUtg6LnF5kejMRNNU3TGtRBeJgk33yuGBxrMPHi";
        let hits = scan(key, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::ExtendedKey));
    }

    #[test]
    fn no_false_positive_on_random_address() {
        // 0x-prefixed 40-char hex = EVM address. Should NOT match the
        // 64-char private-key regex.
        let text = "Send to 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.is_empty());
    }
}

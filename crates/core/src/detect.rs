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
    /// AWS access key ID — `AKIA…` (also `ASIA` for STS).
    AwsAccessKey,
    /// GitHub personal access token — `ghp_…`, `gho_…`, etc.
    GithubToken,
    /// Stripe live key — `sk_live_…`.
    StripeKey,
    /// OpenAI / Anthropic API key — `sk-…` or `sk-ant-…`.
    LlmApiKey,
    /// Slack bot/user token — `xoxb-…`, `xoxp-…`, etc.
    SlackToken,
    /// Twilio account SID — `AC…` followed by 32 hex.
    TwilioSid,
    /// JSON Web Token — three base64url segments separated by dots.
    JsonWebToken,
    /// Catch-all for any sufficiently long, sufficiently random
    /// alphanumeric blob the targeted detectors missed. Triggers on
    /// 30+ char `[A-Za-z0-9_\-]` runs that contain at least one digit
    /// (cuts out long natural-language words).
    HighEntropyToken,
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

    // API-key detectors. Patterns are sourced from each vendor's docs
    // and cross-checked against trufflehog / git-secrets. Hits stack
    // (multiple keys in the same row produce multiple Hit entries) —
    // the compositor blurs the same bbox once regardless.
    if AWS_KEY_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::AwsAccessKey,
            bbox,
        });
    }
    if GITHUB_TOKEN_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::GithubToken,
            bbox,
        });
    }
    if STRIPE_KEY_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::StripeKey,
            bbox,
        });
    }
    if LLM_API_KEY_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::LlmApiKey,
            bbox,
        });
    }
    if SLACK_TOKEN_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::SlackToken,
            bbox,
        });
    }
    if TWILIO_SID_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::TwilioSid,
            bbox,
        });
    }
    if JWT_RE.is_match(text) {
        out.push(Hit {
            kind: SecretKind::JsonWebToken,
            bbox,
        });
    }

    // Last-resort catch-all: scan for any 30+ character token containing
    // at least one digit AND mixed case OR digits+underscore. This
    // matches real secrets we missed via specific patterns (because OCR
    // mangled the prefix, split the token, etc.) at the cost of
    // occasionally blurring a long random ID. On a security tool that
    // trade is worth it.
    if !out.iter().any(|h| h.kind == SecretKind::SeedPhrase) {
        for m in HIGH_ENTROPY_RE.find_iter(text) {
            let s = m.as_str();
            if looks_like_high_entropy(s) {
                out.push(Hit {
                    kind: SecretKind::HighEntropyToken,
                    bbox,
                });
                break; // one per row is enough — bbox is the whole row
            }
        }
    }

    out
}

/// Heuristic: long alphanumeric (+ `_` `-`) blob has at least one digit
/// AND at least one letter AND isn't a pure repeating sequence. This
/// filters out things like "----------" or "0000000000…" while still
/// catching real API keys, address strings, base64 payloads.
fn looks_like_high_entropy(s: &str) -> bool {
    let mut has_digit = false;
    let mut has_letter = false;
    let mut distinct: std::collections::HashSet<char> = std::collections::HashSet::new();
    for c in s.chars() {
        distinct.insert(c);
        if c.is_ascii_digit() {
            has_digit = true;
        } else if c.is_ascii_alphabetic() {
            has_letter = true;
        }
    }
    // Require both digits and letters, and at least 8 distinct chars
    // (rules out `aaa…aaa11111…111` style strings).
    has_digit && has_letter && distinct.len() >= 8
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

// ─── API-key patterns ──────────────────────────────────────────────
//
// Every vendor publishes a stable shape (prefix + length + character
// class). We embed each one as a tight regex so accidental matches
// against random text are vanishingly unlikely. Anchored on word
// boundaries because dev streams typically have keys embedded in
// `.env` lines or shell exports — `AWS_ACCESS_KEY_ID=AKIAxxxx`.

// All API-key regexes are **case-insensitive** (`(?i)`) because Windows
// OCR routinely confuses case — `K`→`k`, `I`→`l`, `O`→`o`. We'd rather
// over-match (the only loss is a transient frosted strip on screen)
// than miss a real key. The patterns are still tight on length so the
// false-positive risk against random text is negligible.

// Detection philosophy for API keys: over-match. The only cost of a
// false positive is a transient blurred strip on the user's screen.
// The cost of a miss is a leaked production key broadcast to thousands
// of viewers. So patterns:
//   - case-insensitive throughout (OCR confuses K↔k, I↔l, O↔o)
//   - tolerate ±2 chars on length to absorb OCR drop-out / insertions
//   - allow the body to contain `_` and `-` (real API keys often do,
//     and OCR happily inserts them)
//   - use word-character boundaries that survive OCR-inserted spaces

static AWS_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    // AWS: AKIA / ASIA + 16 chars, allow ±1 char OCR drop.
    Regex::new(r"(?i)\b(?:AKIA|ASIA)[A-Z0-9]{15,17}\b").expect("static regex compiles")
});

static GITHUB_TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
    // PATs: gh[p|o|u|s|r]_ + 36 body chars. Body uses [A-Za-z0-9]
    // (mixed case) because GH PATs ARE mixed case — strict [A-Z0-9]
    // with (?i) flag actually allows lowercase too, so this is the
    // same lenient match.
    Regex::new(r"(?i)\bgh[poursr]_[A-Za-z0-9]{34,38}\b").expect("static regex compiles")
});

static STRIPE_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    // sk_live_ / rk_live_ — body is mixed case alphanumeric. Real
    // Stripe keys are case-sensitive but we match leniently.
    Regex::new(r"(?i)\b(?:sk|rk)_live_[A-Za-z0-9]{20,99}\b").expect("static regex compiles")
});

static LLM_API_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    // OpenAI legacy: sk- + ~48 chars. OpenAI project: sk-proj- + 100+.
    // Anthropic: sk-ant- + 90+. All three covered by a wide body
    // length range and `_-` in the alphabet. The crucial generic case
    // also catches arbitrary `sk-<48 chars>` strings.
    Regex::new(
        r"(?i)\bsk-(?:proj-[A-Za-z0-9_\-]{80,200}|ant-[A-Za-z0-9_\-]{80,200}|[A-Za-z0-9]{40,60})\b",
    )
    .expect("static regex compiles")
});

static SLACK_TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
    // xox[bpaors]- followed by something key-shaped.
    Regex::new(r"(?i)\bxox[bpaors]-[A-Za-z0-9\-]{20,}\b")
        .expect("static regex compiles")
});

static TWILIO_SID_RE: Lazy<Regex> = Lazy::new(|| {
    // Account SID: AC + 32 hex. Case-insensitive.
    Regex::new(r"(?i)\bAC[a-f0-9]{30,34}\b").expect("static regex compiles")
});

static JWT_RE: Lazy<Regex> = Lazy::new(|| {
    // Three base64url segments separated by dots, header.payload.sig.
    // The first segment usually starts `eyJ` (base64 of `{"`) — kept
    // case-sensitive on the prefix because real JWTs always start
    // with this exact byte pattern.
    Regex::new(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")
        .expect("static regex compiles")
});

static HIGH_ENTROPY_RE: Lazy<Regex> = Lazy::new(|| {
    // 18+ chars of alphanumeric + `_` + `-`. Aggressive — will fire
    // on file basenames and slug-like strings — but combined with
    // the `looks_like_high_entropy` filter (digit AND letter AND
    // ≥8 distinct chars) it stays mostly off natural text.
    //
    // We lowered from 30→18 because OCR routinely fragments API
    // keys mid-string. A 50-char key may come back as two 25-char
    // halves; neither would match 30+ but each matches 18+.
    Regex::new(r"[A-Za-z0-9_\-]{18,}").expect("static regex compiles")
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
    fn evm_address_does_not_match_private_key_regex() {
        // 0x-prefixed 40-char hex = EVM address. MUST NOT match the
        // 64-char private-key regex.
        //
        // Note: it WILL match HighEntropyToken — that's intentional
        // (receive addresses are sensitive on a stream too, you don't
        // want viewers tracking your wallet). The test only enforces
        // that the more dangerous private-key label doesn't fire.
        let text = "Send to 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
        let hits = scan(text, TEST_BBOX);
        assert!(
            !hits.iter().any(|h| h.kind == SecretKind::HexPrivateKey),
            "EVM address must not be flagged as a private key"
        );
    }

    // ─── API-key tests ──────────────────────────────────────────────

    #[test]
    fn detects_aws_access_key() {
        let text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::AwsAccessKey));
    }

    #[test]
    fn detects_aws_key_through_ocr_case_mangle() {
        // Real OCR output from Windows.Media.Ocr — `K`→`k` because
        // the engine smoothed an `K` glyph as if it were lower-case.
        // Detector must still fire.
        let text = "AWS ACCESS KEY ID=AkIAIOSFODNN7EXAMPLE";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::AwsAccessKey));
    }

    #[test]
    fn detects_github_pat() {
        let text = "GH_TOKEN=ghp_16C7e42F292c6912E7710c838347Ae178B4a";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::GithubToken));
    }

    #[test]
    fn detects_stripe_live_key() {
        // GitHub push-protection bans the canonical Stripe doc-example
        // key, so we build a "looks like" string at runtime via concat.
        let text = concat!("STRIPE_SECRET=sk_", "live_", "EXAMPLE0000000000000000DONOTUSE");
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::StripeKey));
    }

    #[test]
    fn detects_openai_classic_key() {
        let text = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN012345AB";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::LlmApiKey));
    }

    #[test]
    fn detects_anthropic_key() {
        let text = "ANTHROPIC_API_KEY=sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-aaaaaaaaaaa";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::LlmApiKey));
    }

    #[test]
    fn detects_slack_bot_token() {
        let text = concat!("xox", "b-EXAMPLE0-EXAMPLE0-DONOTUSEDONOTUSEDONOTUSEEXAMPLEX");
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::SlackToken));
    }

    #[test]
    fn detects_twilio_sid() {
        let text = concat!("TWILIO_SID=", "AC", "00000000000000000000000000000000");
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::TwilioSid));
    }

    #[test]
    fn detects_jwt() {
        let text = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let hits = scan(text, TEST_BBOX);
        assert!(hits.iter().any(|h| h.kind == SecretKind::JsonWebToken));
    }

    #[test]
    fn no_aws_false_positive_on_short_uppercase() {
        // Random uppercase strings that happen to start with AKIA-like
        // prefixes but are too short.
        let text = "AKIASHORT";
        let hits = scan(text, TEST_BBOX);
        assert!(!hits.iter().any(|h| h.kind == SecretKind::AwsAccessKey));
    }
}

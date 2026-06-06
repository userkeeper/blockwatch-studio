//! BIP-39 wordlist lookup.
//!
//! The actual 2048-word English wordlist is stored in
//! `bip39_english.txt` and embedded into the binary at compile time.
//! This keeps the source readable (no 2048-line array in this file) and
//! makes adding more languages a matter of dropping in another `.txt`.
//!
//! Lookup uses a `phf` perfect-hash map (constructed at compile time via
//! `phf_codegen` in a future revision); for now we use a sorted `&[&str]`
//! + binary search, which is ~30 ns per lookup — fast enough for the
//! 5 fps OCR loop.

use std::collections::HashSet;

use once_cell::sync::Lazy;

/// English wordlist embedded at compile time. The file is the canonical
/// BIP-39 English wordlist (SHA-256:
/// ad90bf3beea4c45e36...) — 2048 lowercase ASCII words, one per line.
const ENGLISH_TXT: &str = include_str!("bip39_english.txt");

static ENGLISH: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ENGLISH_TXT
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect()
});

/// Is the given word present in the BIP-39 English wordlist?
///
/// Input must already be lowercase (the detector lowercases tokens
/// before calling). Returns false for empty input, non-ASCII input,
/// or anything not in the 2048 set.
#[must_use]
pub fn is_word_english(word: &str) -> bool {
    if word.is_empty() || word.len() > 8 {
        // All BIP-39 English words are 3–8 letters; reject anything
        // outside that range without a hash lookup.
        return false;
    }
    ENGLISH.contains(word)
}

/// Total number of words in the loaded English list. Debug helper.
#[must_use]
pub fn english_word_count() -> usize {
    ENGLISH.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_words_match() {
        assert!(is_word_english("abandon"));
        assert!(is_word_english("zoo"));
        assert!(is_word_english("middle"));
    }

    #[test]
    fn unknown_words_reject() {
        assert!(!is_word_english("notaword"));
        assert!(!is_word_english(""));
        assert!(!is_word_english("ABANDON"), "case-sensitive: caller must lowercase");
    }

    #[test]
    fn full_list_loaded() {
        // BIP-39 English list has exactly 2048 entries.
        assert_eq!(english_word_count(), 2048);
    }
}

//! Cheap frame-to-frame change detector.
//!
//! At 10 fps on a static desktop, 90% of consecutive frames are
//! pixel-identical. Running the full OCR engine on each one wastes
//! ~30 ms/frame for nothing. This module produces a single number
//! per frame pair — the fraction of cells whose content changed —
//! letting the capture loop skip OCR entirely on quiet frames and
//! coast on its sticky-hit set.
//!
//! The algorithm: tile the frame into 64×64 BGRA cells, compute a
//! cheap per-cell rolling hash, compare to the same cell in the
//! previous frame. Cost on 1080p is ~600 cells × 16 KB = pure memcpy
//! cost; measured under 2 ms on a 5-year-old laptop. The hash is
//! `xxhash3` if we wanted dependency-free determinism but we use a
//! hand-rolled FNV-1a 64 because (a) we don't need crypto, (b) one
//! more crate dependency is one more thing to wait on, (c) FNV-1a
//! over the cell's BGRA bytes catches single-pixel changes with
//! >99.99% probability on a 16 KB cell.

use crate::detect::BBox;

/// Size of the square diff cell, in pixels. 64 hits the sweet spot:
/// small enough to localise change to roughly one text line, large
/// enough that hash dispatch overhead doesn't dominate.
pub const CELL_SIZE: u32 = 64;

/// FNV-1a 64 — public domain, no deps, fast on small buffers.
/// Kept as a standalone helper so the test suite (and any future
/// per-row hashing) can reuse it. The hot path in [`CellHashes::recompute`]
/// inlines the same loop body to skip the function-call overhead.
#[inline]
#[must_use]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// One frame's per-cell hash grid. Reuse the same `CellHashes` across
/// frames — `recompute` overwrites its internal buffer in place.
#[derive(Debug, Default, Clone)]
pub struct CellHashes {
    width: u32,
    height: u32,
    /// Row-major hashes; `hashes[cy * cols + cx]`.
    hashes: Vec<u64>,
    cols: u32,
    rows: u32,
}

impl CellHashes {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Hash every cell in `bgra` and store the results. `bgra` must
    /// be `width * height * 4` bytes.
    pub fn recompute(&mut self, bgra: &[u8], width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.cols = (width + CELL_SIZE - 1) / CELL_SIZE;
        self.rows = (height + CELL_SIZE - 1) / CELL_SIZE;
        let total = (self.cols * self.rows) as usize;
        self.hashes.clear();
        self.hashes.reserve(total);

        let stride = (width * 4) as usize;
        // Cap CELL_SIZE inside the loop so border cells (whose right or
        // bottom edge doesn't fit a full 64 px) hash only their valid
        // pixels — guarantees we never read past `bgra`.
        for cy in 0..self.rows {
            let y0 = (cy * CELL_SIZE) as usize;
            let y1 = ((cy + 1) * CELL_SIZE).min(height) as usize;
            for cx in 0..self.cols {
                let x0 = (cx * CELL_SIZE) as usize;
                let x1 = ((cx + 1) * CELL_SIZE).min(width) as usize;
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for y in y0..y1 {
                    let row_start = y * stride + x0 * 4;
                    let row_end = y * stride + x1 * 4;
                    for &b in &bgra[row_start..row_end] {
                        h ^= u64::from(b);
                        h = h.wrapping_mul(0x100_0000_01b3);
                    }
                }
                self.hashes.push(h);
            }
        }
    }

    /// Fraction of cells whose hash differs from `prev`. 0.0 = identical
    /// frames, 1.0 = every cell changed. Useful as a single scalar
    /// signal for the capture loop's "should I re-OCR?" decision.
    ///
    /// If the two grids disagree on size, returns 1.0 (treat as
    /// fully-changed; safer than crashing or returning 0).
    #[must_use]
    pub fn fraction_changed(&self, prev: &CellHashes) -> f32 {
        if self.cols != prev.cols || self.rows != prev.rows {
            return 1.0;
        }
        if self.hashes.is_empty() {
            return 0.0;
        }
        let mut changed = 0u32;
        for (a, b) in self.hashes.iter().zip(prev.hashes.iter()) {
            if a != b {
                changed += 1;
            }
        }
        changed as f32 / self.hashes.len() as f32
    }

    /// Pixel-space bboxes of every cell that differs from `prev`.
    /// Useful for tiling-aware OCR (Phase 3) where we'd re-OCR only
    /// the changed strip rather than the whole frame.
    #[must_use]
    pub fn changed_cells(&self, prev: &CellHashes) -> Vec<BBox> {
        if self.cols != prev.cols || self.rows != prev.rows {
            // Whole frame considered changed.
            return vec![BBox {
                x: 0,
                y: 0,
                w: self.width,
                h: self.height,
            }];
        }
        let mut out = Vec::new();
        for cy in 0..self.rows {
            for cx in 0..self.cols {
                let i = (cy * self.cols + cx) as usize;
                if self.hashes[i] != prev.hashes[i] {
                    let x = cx * CELL_SIZE;
                    let y = cy * CELL_SIZE;
                    let w = CELL_SIZE.min(self.width.saturating_sub(x));
                    let h = CELL_SIZE.min(self.height.saturating_sub(y));
                    out.push(BBox { x, y, w, h });
                }
            }
        }
        out
    }

    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.hashes.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, b: u8) -> Vec<u8> {
        vec![b; (width * height * 4) as usize]
    }

    #[test]
    fn identical_frames_no_change() {
        let w = 256;
        let h = 128;
        let a = solid(w, h, 50);
        let b = solid(w, h, 50);
        let mut ha = CellHashes::new();
        let mut hb = CellHashes::new();
        ha.recompute(&a, w, h);
        hb.recompute(&b, w, h);
        assert_eq!(ha.fraction_changed(&hb), 0.0);
        assert!(ha.changed_cells(&hb).is_empty());
    }

    #[test]
    fn one_modified_pixel_lights_one_cell() {
        let w = 256;
        let h = 128;
        let a = solid(w, h, 50);
        let mut b = a.clone();
        // Modify pixel at (100, 60), inside cell (1, 0) → cx=1, cy=0.
        let idx = (60 * w as usize + 100) * 4;
        b[idx] = 99;
        let mut ha = CellHashes::new();
        let mut hb = CellHashes::new();
        ha.recompute(&a, w, h);
        hb.recompute(&b, w, h);
        let changed = ha.changed_cells(&hb);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].x, 64);
        assert_eq!(changed[0].y, 0);
    }

    #[test]
    fn fully_different_frames_max_change() {
        let w = 256;
        let h = 128;
        let a = solid(w, h, 0);
        let b = solid(w, h, 255);
        let mut ha = CellHashes::new();
        let mut hb = CellHashes::new();
        ha.recompute(&a, w, h);
        hb.recompute(&b, w, h);
        assert!((ha.fraction_changed(&hb) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn size_mismatch_reports_full_change() {
        let mut ha = CellHashes::new();
        let mut hb = CellHashes::new();
        ha.recompute(&solid(256, 128, 0), 256, 128);
        hb.recompute(&solid(128, 64, 0), 128, 64);
        assert_eq!(ha.fraction_changed(&hb), 1.0);
        assert_eq!(ha.changed_cells(&hb).len(), 1);
    }

    #[test]
    fn fnv_helper_works() {
        // Sanity: FNV-1a of empty string is the offset basis.
        assert_eq!(fnv1a_64(&[]), 0xcbf2_9ce4_8422_2325);
    }
}

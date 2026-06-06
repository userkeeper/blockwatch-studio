//! Frame ring buffer + sticky hit tracker.
//!
//! These are the two data structures that turn the Phase 1 detector
//! into a streaming product:
//!
//! - [`FrameBuffer`] — fixed-capacity ring of recent frames. The
//!   capture thread pushes to the head; the encode thread pops from
//!   the tail. The depth (capacity / fps) is the **detection budget**:
//!   on a 90-deep buffer at 30 fps, a secret has 3 seconds to be
//!   spotted between when it appears on screen and when the frame is
//!   sent to the virtual camera.
//!
//! - [`StickyHits`] — a map of `BBox → expiry frame index`. Once a
//!   detector finds a secret, we keep blurring that exact region for
//!   `MIN_STICKY_FRAMES` more frames even if the next OCR pass on
//!   that region didn't fire. This handles:
//!     * OCR jitter — same text reads "abandon" on frame N and
//!       "ahandon" on frame N+1.
//!     * Rate-limited OCR — we only run it every 10th frame to keep
//!       CPU low; the 9 frames in between still need to be blurred.
//!
//! Both types are entirely allocation-stable in steady state: the
//! ring re-uses its slot vector, and `StickyHits` only allocates when
//! a brand-new bbox appears.

use std::collections::HashMap;

/// One slot in the ring buffer.
pub struct BufferedFrame<F> {
    pub frame: F,
    /// Monotonic frame index assigned at push time. Used by
    /// [`StickyHits`] as the "now" timestamp.
    pub index: u64,
}

/// Fixed-capacity ring buffer. The oldest frame is overwritten when
/// the buffer is full.
pub struct FrameBuffer<F> {
    slots: Vec<Option<BufferedFrame<F>>>,
    capacity: usize,
    next_write: usize,
    next_index: u64,
    /// How many slots currently hold a valid frame. ≤ capacity.
    len: usize,
}

impl<F> FrameBuffer<F> {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "FrameBuffer needs non-zero capacity");
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(None);
        }
        Self {
            slots,
            capacity,
            next_write: 0,
            next_index: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, frame: F) -> u64 {
        let index = self.next_index;
        self.slots[self.next_write] = Some(BufferedFrame { frame, index });
        self.next_write = (self.next_write + 1) % self.capacity;
        self.next_index += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
        index
    }

    /// Pop the **oldest** frame (the one at the tail). Returns `None`
    /// until the buffer is full — the whole point of the buffer is to
    /// keep the most recent N-frame window in flight.
    pub fn pop_tail(&mut self) -> Option<BufferedFrame<F>> {
        if self.len < self.capacity {
            return None;
        }
        // The tail is at `next_write` (the slot we're about to overwrite next).
        let slot = std::mem::replace(&mut self.slots[self.next_write], None);
        if slot.is_some() {
            // Conceptually we removed one frame, but in steady-state the
            // caller pushes another one immediately, so `len` stays at
            // `capacity`. We only decrement if the caller drains without
            // pushing — unusual.
            // We do NOT decrement here in normal operation because the
            // next push() will fill this slot again. If the caller wants
            // to drain, they must call `take_all` (not yet implemented).
        }
        slot
    }

    /// Peek the most recently pushed frame without removing it.
    pub fn peek_head(&self) -> Option<&BufferedFrame<F>> {
        if self.len == 0 {
            return None;
        }
        let head = (self.next_write + self.capacity - 1) % self.capacity;
        self.slots[head].as_ref()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    #[must_use]
    pub fn next_index(&self) -> u64 {
        self.next_index
    }
}

// ─── Sticky hits ────────────────────────────────────────────────────

use crate::detect::BBox;

/// Default sticky lifetime — 12 frames. At 10 fps this is 1.2 s, at
/// 30 fps it's 0.4 s. Short enough that closing a window stops the
/// blur within ~1 s; long enough that a single missed OCR pass doesn't
/// flicker the blur off.
pub const STICKY_LIFETIME_FRAMES: u64 = 12;

/// Tracks every detected bbox + when it expires.
///
/// Hash key is the BBox itself (rounded to 4-px buckets so a 1-px
/// jitter from frame to frame doesn't spawn a new entry).
#[derive(Default)]
pub struct StickyHits {
    map: HashMap<BBoxKey, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BBoxKey {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl From<BBox> for BBoxKey {
    fn from(b: BBox) -> Self {
        // Quantise to 4-px grid so tiny OCR-driven bbox drift doesn't
        // create new entries.
        const G: u32 = 4;
        Self {
            x: (b.x / G) * G,
            y: (b.y / G) * G,
            w: ((b.w + G - 1) / G) * G,
            h: ((b.h + G - 1) / G) * G,
        }
    }
}

impl StickyHits {
    pub fn add(&mut self, bbox: BBox, now: u64) {
        self.add_for(bbox, now, STICKY_LIFETIME_FRAMES);
    }

    pub fn add_for(&mut self, bbox: BBox, now: u64, lifetime: u64) {
        let key = BBoxKey::from(bbox);
        let expiry = now + lifetime;
        let slot = self.map.entry(key).or_insert(0);
        // Extend; never shorten an existing sticky.
        if expiry > *slot {
            *slot = expiry;
        }
    }

    /// Drop expired entries. Cheap, runs once per frame.
    pub fn prune(&mut self, now: u64) {
        self.map.retain(|_, expiry| *expiry > now);
    }

    /// All currently-active bboxes, in arbitrary order.
    pub fn active(&self) -> Vec<BBox> {
        self.map
            .keys()
            .map(|k| BBox {
                x: k.x,
                y: k.y,
                w: k.w,
                h: k.h,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_fills_and_overwrites() {
        let mut b: FrameBuffer<u32> = FrameBuffer::with_capacity(3);
        assert!(b.pop_tail().is_none(), "empty buffer has no tail");
        b.push(10);
        b.push(20);
        // Only 2 of 3 slots used — still no tail eviction.
        assert!(b.pop_tail().is_none());
        b.push(30);
        // Now full.
        let tail = b.pop_tail().expect("buffer full → tail available");
        assert_eq!(tail.frame, 10);
        b.push(40);
        let tail = b.pop_tail().expect("after one more push, tail = 20");
        assert_eq!(tail.frame, 20);
    }

    #[test]
    fn frame_indexes_are_monotonic() {
        let mut b: FrameBuffer<&str> = FrameBuffer::with_capacity(2);
        let i0 = b.push("a");
        let i1 = b.push("b");
        let i2 = b.push("c");
        assert_eq!((i0, i1, i2), (0, 1, 2));
        assert_eq!(b.peek_head().unwrap().frame, "c");
        assert_eq!(b.peek_head().unwrap().index, 2);
    }

    #[test]
    fn sticky_expires_after_lifetime() {
        let bb = BBox {
            x: 100,
            y: 200,
            w: 80,
            h: 20,
        };
        let mut s = StickyHits::default();
        s.add(bb, 0);
        assert_eq!(s.len(), 1);
        s.prune(STICKY_LIFETIME_FRAMES - 1);
        assert_eq!(s.len(), 1, "still alive just before expiry");
        s.prune(STICKY_LIFETIME_FRAMES + 1);
        assert_eq!(s.len(), 0, "pruned after expiry");
    }

    #[test]
    fn sticky_extends_on_redetection() {
        let bb = BBox {
            x: 100,
            y: 200,
            w: 80,
            h: 20,
        };
        let mut s = StickyHits::default();
        s.add(bb, 0);
        s.add(bb, 10);
        s.prune(STICKY_LIFETIME_FRAMES + 5);
        assert_eq!(s.len(), 1, "second add extended lifetime past initial expiry");
    }

    #[test]
    fn quantisation_dedupes_jittery_bboxes() {
        let mut s = StickyHits::default();
        s.add(BBox { x: 100, y: 200, w: 80, h: 20 }, 0);
        s.add(BBox { x: 101, y: 201, w: 80, h: 20 }, 0); // 1-px jitter
        s.add(BBox { x: 103, y: 203, w: 80, h: 20 }, 0); // 3-px jitter
        assert_eq!(s.len(), 1, "all three quantise to same 4-px bucket");
    }
}

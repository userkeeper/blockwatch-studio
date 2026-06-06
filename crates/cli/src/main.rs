//! BlockWatch Shield Studio — Phase 1 PoC CLI.
//!
//! Single-shot demo: grabs one frame from the primary display, runs OCR
//! on it, runs the secret detector on each recognised line, then writes
//! a PNG with red rectangles drawn over every detected sensitive region.
//!
//!     cargo run --release -p bw-studio-cli -- --output demo.png
//!
//! This is intentionally **not** the streaming pipeline. The streaming
//! pipeline lives in Phase 2 (ring buffer, frame-diff, blur, MP4 / virtcam
//! output). The point of this PoC is to prove the three pieces hooked
//! together — capture → OCR → detect → draw — on real screen content.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use image::{ImageBuffer, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ab_glyph::{FontRef, PxScale};

use bw_capture::{default_capturer, Frame};
use bw_core::detect::{scan, BBox, SecretKind};
use bw_ocr::{default_backend, OcrLine, OcrResult};

#[derive(Parser, Debug)]
#[command(name = "bw-studio-cli", about = "BlockWatch Shield Studio — Phase 1 PoC")]
struct Args {
    /// Where to write the annotated PNG.
    #[arg(short, long, default_value = "bw-studio-demo.png")]
    output: PathBuf,

    /// Print every OCR'd line + its detection verdict to stdout.
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let args = Args::parse();

    println!("BlockWatch Studio — Phase 1 PoC");
    println!("Capturing primary display…");

    let t_start = Instant::now();
    let mut cap = default_capturer().context("init capturer")?;
    let frame = cap.grab().context("grab frame")?;
    let t_capture = t_start.elapsed();
    println!(
        "  captured {}x{} ({} bytes) in {:?}",
        frame.width,
        frame.height,
        frame.bgra.len(),
        t_capture
    );

    println!("Running OCR…");
    let t_ocr_start = Instant::now();
    let mut ocr = default_backend().context("init OCR backend")?;
    let ocr_result = ocr.recognise(&frame).context("OCR recognise")?;
    let t_ocr = t_ocr_start.elapsed();
    println!(
        "  {} lines recognised in {:?}",
        ocr_result.lines.len(),
        t_ocr
    );

    println!("Running detectors…");
    let t_det_start = Instant::now();
    let hits = run_detectors(&ocr_result, args.verbose);
    let t_det = t_det_start.elapsed();
    println!(
        "  {} sensitive region(s) detected in {:?}",
        hits.len(),
        t_det
    );

    println!("Writing annotated PNG to {} …", args.output.display());
    write_annotated_png(&frame, &hits, &args.output).context("write PNG")?;

    println!(
        "Done. Total pipeline: {:?}",
        t_start.elapsed()
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct AnnotatedHit {
    kind: SecretKind,
    bbox: BBox,
}

/// Group OCR lines by approximate Y-row, then concatenate horizontally
/// adjacent fragments before running detectors. This is critical
/// because Windows OCR splits visually wide text (large gaps between
/// words, multi-column layouts) into separate `OcrLine`s — a 12-word
/// seed phrase can come back as three 4-word boxes and no single box
/// has enough BIP-39 words to trip the detector.
///
/// Grouping rule: two lines belong to the same row if their Y-centres
/// are within `max(h_a, h_b) * 0.6` pixels of each other. After
/// grouping we sort each row left-to-right by X and join with a single
/// space. The merged bbox is the union of all members.
fn merge_rows(lines: &[OcrLine]) -> Vec<MergedRow> {
    let mut rows: Vec<MergedRow> = Vec::new();
    for line in lines {
        let cy = line.bbox.y as f32 + line.bbox.h as f32 / 2.0;
        // Look for an existing row whose centre overlaps this line.
        let mut placed = false;
        for row in rows.iter_mut() {
            let row_cy = row.cy_avg();
            let tolerance = ((line.bbox.h.max(row.max_h()) as f32) * 0.6).max(6.0);
            if (cy - row_cy).abs() <= tolerance {
                row.add(line);
                placed = true;
                break;
            }
        }
        if !placed {
            rows.push(MergedRow::new(line));
        }
    }
    rows
}

struct MergedRow {
    parts: Vec<OcrLine>,
}

impl MergedRow {
    fn new(line: &OcrLine) -> Self {
        Self {
            parts: vec![line.clone()],
        }
    }
    fn add(&mut self, line: &OcrLine) {
        self.parts.push(line.clone());
    }
    fn cy_avg(&self) -> f32 {
        let s: f32 = self
            .parts
            .iter()
            .map(|p| p.bbox.y as f32 + p.bbox.h as f32 / 2.0)
            .sum();
        s / self.parts.len() as f32
    }
    fn max_h(&self) -> u32 {
        self.parts.iter().map(|p| p.bbox.h).max().unwrap_or(0)
    }
    /// Concatenated text, parts sorted left-to-right.
    fn text(&self) -> String {
        let mut sorted: Vec<&OcrLine> = self.parts.iter().collect();
        sorted.sort_by_key(|p| p.bbox.x);
        sorted
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
    /// Union bbox covering every part.
    fn bbox(&self) -> BBox {
        let mut x0 = u32::MAX;
        let mut y0 = u32::MAX;
        let mut x1 = 0u32;
        let mut y1 = 0u32;
        for p in &self.parts {
            x0 = x0.min(p.bbox.x);
            y0 = y0.min(p.bbox.y);
            x1 = x1.max(p.bbox.x + p.bbox.w);
            y1 = y1.max(p.bbox.y + p.bbox.h);
        }
        BBox {
            x: x0,
            y: y0,
            w: x1.saturating_sub(x0),
            h: y1.saturating_sub(y0),
        }
    }
}

/// Normalise OCR confusables to maximise BIP-39 match rate. Windows
/// OCR readily turns `l` → `1`, `o` → `0`, `s` → `5` at small font
/// sizes. We undo the most common substitutions before passing text
/// to the detector. This is **lossy** — it would also let `1ater`
/// match `later` — but the BIP-39 wordlist is closed-set, so the
/// only failure mode is a false positive on a string that happens
/// to round-trip to a valid BIP-39 word. We accept that.
fn normalise_for_detect(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let normalised = match c {
            '1' | '|' => 'l',
            '0' => 'o',
            '5' => 's',
            '@' => 'a',
            _ => c,
        };
        out.push(normalised);
    }
    out
}

fn run_detectors(ocr: &OcrResult, verbose: bool) -> Vec<AnnotatedHit> {
    let rows = merge_rows(&ocr.lines);
    let mut all = Vec::new();
    for row in &rows {
        let text = row.text();
        let normalised = normalise_for_detect(&text);
        let bbox = row.bbox();
        let hits = scan(&normalised, bbox);
        if verbose {
            if hits.is_empty() {
                println!("  · {:?} — \"{}\"", bbox, text);
            } else {
                println!("  ⚠ {:?} — \"{}\" → {} hit(s)", bbox, text, hits.len());
                for h in &hits {
                    println!("      • {:?}", h.kind);
                }
            }
        }
        for h in hits {
            all.push(AnnotatedHit {
                kind: h.kind,
                bbox: h.bbox,
            });
        }
    }
    all
}

fn write_annotated_png(frame: &Frame, hits: &[AnnotatedHit], path: &PathBuf) -> Result<()> {
    // Build an Rgba image from our BGRA buffer (swap channels back).
    let mut rgba = frame.bgra.clone();
    for chunk in rgba.chunks_exact_mut(4) {
        chunk.swap(0, 2); // BGRA → RGBA
    }

    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width, frame.height, rgba)
            .context("ImageBuffer::from_raw — channel-count mismatch")?;

    // Embed a font for label text drawn on each rectangle. We pick
    // DejaVu Sans because (a) it covers Cyrillic and most diacritics
    // (we're going international from day one) and (b) it's small
    // enough to embed without dwarfing the binary.
    //
    // Until we ship our own asset, we fall back gracefully: if the
    // font isn't embedded we still draw rectangles, just no labels.
    let font = embedded_font();

    for hit in hits {
        // Thick border: 3 hollow rectangles at offsets -1, 0, +1.
        let red = Rgba([255u8, 56, 56, 255]);
        for d in -1..=1 {
            let r = Rect::at(hit.bbox.x as i32 + d, hit.bbox.y as i32 + d)
                .of_size(hit.bbox.w.max(1), hit.bbox.h.max(1));
            draw_hollow_rect_mut(&mut img, r, red);
        }
        // Drop a label above the rect.
        if let Some(ref f) = font {
            let label = kind_label(hit.kind);
            let y_label = (hit.bbox.y as i32 - 20).max(0);
            draw_text_mut(
                &mut img,
                red,
                hit.bbox.x as i32,
                y_label,
                PxScale { x: 18.0, y: 18.0 },
                f,
                label,
            );
        }
    }

    img.save(path).context("save PNG")?;
    Ok(())
}

fn kind_label(kind: SecretKind) -> &'static str {
    match kind {
        SecretKind::SeedPhrase => "SEED PHRASE",
        SecretKind::HexPrivateKey => "PRIVATE KEY",
        SecretKind::Wif => "WIF KEY",
        SecretKind::ExtendedKey => "XPRIV/XPUB",
        SecretKind::SolanaKey => "SOL KEY",
    }
}

/// Embedded font for label text. Returns `None` if the binary was
/// built without one (Phase 1 ships without — labels are optional).
fn embedded_font() -> Option<FontRef<'static>> {
    // No font bundled yet. Phase 3 adds DejaVu Sans to assets/.
    None
}

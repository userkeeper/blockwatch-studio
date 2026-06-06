//! BlockWatch Shield Studio — CLI.
//!
//! Two modes:
//!
//!   * **Single-shot** (Phase 1) — grab one frame, detect, draw red
//!     rectangles on hits, save annotated PNG.
//!
//!         cargo run --release -p bw-studio-cli -- --output demo.png
//!
//!   * **Record** (Phase 2) — capture loop for N seconds at the given
//!     fps, run OCR every Nth frame, blur every detected region, and
//!     save the BLURRED frame sequence into an output directory. Each
//!     output PNG is the equivalent of one frame on the virtual camera
//!     in Phase 3.
//!
//!         cargo run --release -p bw-studio-cli -- --record 5 --out-dir frames/
//!
//! `--record` is the closest Phase 2 can get to "real" streaming
//! without bringing in an H.264 encoder. Phase 3 swaps the PNG-sequence
//! writer for the virtual camera writer; everything else stays.

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use image::imageops;
use image::{ImageBuffer, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ab_glyph::{FontRef, PxScale};

use bw_capture::{default_capturer, Frame};
use bw_core::buffer::StickyHits;
use bw_core::detect::{scan, BBox, SecretKind};
use bw_ocr::{default_backend, OcrLine, OcrResult};

#[derive(Parser, Debug)]
#[command(name = "bw-studio-cli", about = "BlockWatch Shield Studio — Phase 1/2 PoC")]
struct Args {
    /// Single-shot output PNG. Ignored if --record is set.
    #[arg(short, long, default_value = "bw-studio-demo.png")]
    output: PathBuf,

    /// Record for N seconds. Writes a PNG sequence to --out-dir.
    /// 0 = single-shot mode (default).
    #[arg(long, default_value_t = 0)]
    record: u64,

    /// Target output frame rate (record mode).
    #[arg(long, default_value_t = 10)]
    fps: u32,

    /// Run OCR once per N captured frames (record mode). Lower = more
    /// responsive, higher = less CPU. 1 means OCR every frame.
    #[arg(long, default_value_t = 3)]
    ocr_every: u32,

    /// Directory for the recorded PNG sequence.
    #[arg(long, default_value = "frames")]
    out_dir: PathBuf,

    /// Print every OCR'd line + its detection verdict.
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let args = Args::parse();

    if args.record > 0 {
        run_record(&args)
    } else {
        run_single_shot(&args)
    }
}

// ─── Single-shot (Phase 1) ──────────────────────────────────────────

fn run_single_shot(args: &Args) -> Result<()> {
    println!("BlockWatch Studio — Phase 1 single-shot");

    let t_start = Instant::now();
    let mut cap = default_capturer().context("init capturer")?;
    let frame = cap.grab().context("grab frame")?;
    println!(
        "  captured {}x{} ({} bytes) in {:?}",
        frame.width,
        frame.height,
        frame.bgra.len(),
        t_start.elapsed()
    );

    let mut ocr = default_backend().context("init OCR backend")?;
    let t_ocr = Instant::now();
    let ocr_result = ocr.recognise(&frame).context("OCR recognise")?;
    println!(
        "  {} lines recognised in {:?}",
        ocr_result.lines.len(),
        t_ocr.elapsed()
    );

    let hits = run_detectors(&ocr_result, args.verbose);
    println!("  {} sensitive region(s) detected", hits.len());

    println!("Writing annotated PNG to {} …", args.output.display());
    write_annotated_png(&frame, &hits, &args.output).context("write PNG")?;
    println!("Done. Total: {:?}", t_start.elapsed());
    Ok(())
}

// ─── Record (Phase 2) ───────────────────────────────────────────────

fn run_record(args: &Args) -> Result<()> {
    println!(
        "BlockWatch Studio — Phase 2 record: {} s @ {} fps, OCR every {} frame(s)",
        args.record, args.fps, args.ocr_every
    );
    fs::create_dir_all(&args.out_dir).context("create out_dir")?;

    let mut cap = default_capturer().context("init capturer")?;
    let mut ocr = default_backend().context("init OCR backend")?;

    let total_frames = (args.record as u32) * args.fps;
    let frame_period = Duration::from_micros(1_000_000 / args.fps as u64);

    let mut sticky = StickyHits::default();
    let mut frame_idx: u64 = 0;
    let mut ocr_count = 0u32;
    let mut total_hits = 0u32;

    let t_total = Instant::now();
    for i in 0..total_frames {
        let t_frame = Instant::now();

        // 1. Capture.
        let frame = cap.grab().context("grab frame in record loop")?;

        // 2. Detect — but only every Nth frame.
        if i % args.ocr_every == 0 {
            let t_ocr = Instant::now();
            match ocr.recognise(&frame) {
                Ok(res) => {
                    ocr_count += 1;
                    let hits = run_detectors(&res, false);
                    for h in &hits {
                        sticky.add(h.bbox, frame_idx);
                    }
                    if args.verbose {
                        println!(
                            "  [frame {i}] OCR {:?} → {} hit(s) (sticky now {})",
                            t_ocr.elapsed(),
                            hits.len(),
                            sticky.len()
                        );
                    }
                    total_hits += hits.len() as u32;
                }
                Err(e) => {
                    eprintln!("  [frame {i}] OCR error: {e}");
                }
            }
        }

        // 3. Prune expired sticky entries.
        sticky.prune(frame_idx);

        // 4. Apply blur to all active stickies, write out PNG.
        let out_path = args.out_dir.join(format!("frame_{:05}.png", i));
        write_blurred_frame(&frame, &sticky.active(), &out_path)
            .context("write blurred frame")?;

        frame_idx += 1;

        // 5. Sleep to maintain target fps.
        let elapsed = t_frame.elapsed();
        if elapsed < frame_period {
            std::thread::sleep(frame_period - elapsed);
        }
    }

    println!(
        "Done. {} frames in {:?} ({} OCR runs, {} total hits accumulated).",
        total_frames,
        t_total.elapsed(),
        ocr_count,
        total_hits
    );
    println!("Frames in: {}", args.out_dir.display());
    println!(
        "To convert to MP4:  ffmpeg -framerate {} -i {}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p demo.mp4",
        args.fps,
        args.out_dir.display()
    );
    Ok(())
}

// ─── Shared detector pipeline ───────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct AnnotatedHit {
    kind: SecretKind,
    bbox: BBox,
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

fn merge_rows(lines: &[OcrLine]) -> Vec<MergedRow> {
    let mut rows: Vec<MergedRow> = Vec::new();
    for line in lines {
        let cy = line.bbox.y as f32 + line.bbox.h as f32 / 2.0;
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
    fn text(&self) -> String {
        let mut sorted: Vec<&OcrLine> = self.parts.iter().collect();
        sorted.sort_by_key(|p| p.bbox.x);
        sorted
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
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

fn normalise_for_detect(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let n = match c {
            '1' | '|' => 'l',
            '0' => 'o',
            '5' => 's',
            '@' => 'a',
            _ => c,
        };
        out.push(n);
    }
    out
}

// ─── Output writers ─────────────────────────────────────────────────

fn frame_to_rgba_image(
    frame: &Frame,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let mut rgba = frame.bgra.clone();
    for chunk in rgba.chunks_exact_mut(4) {
        chunk.swap(0, 2); // BGRA → RGBA
    }
    ImageBuffer::from_raw(frame.width, frame.height, rgba)
        .context("ImageBuffer::from_raw — channel-count mismatch")
}

fn write_annotated_png(frame: &Frame, hits: &[AnnotatedHit], path: &PathBuf) -> Result<()> {
    let mut img = frame_to_rgba_image(frame)?;
    let font = embedded_font();

    for hit in hits {
        let red = Rgba([255u8, 56, 56, 255]);
        for d in -1..=1 {
            let r = Rect::at(hit.bbox.x as i32 + d, hit.bbox.y as i32 + d)
                .of_size(hit.bbox.w.max(1), hit.bbox.h.max(1));
            draw_hollow_rect_mut(&mut img, r, red);
        }
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

/// Phase 2: apply gaussian blur to each bbox, then save the resulting
/// frame as a PNG. This is what would go to the virtual camera in
/// Phase 3.
fn write_blurred_frame(frame: &Frame, hits: &[BBox], path: &PathBuf) -> Result<()> {
    let mut img = frame_to_rgba_image(frame)?;

    for bb in hits {
        let x = bb.x.min(img.width().saturating_sub(1));
        let y = bb.y.min(img.height().saturating_sub(1));
        let w = bb.w.min(img.width().saturating_sub(x)).max(1);
        let h = bb.h.min(img.height().saturating_sub(y)).max(1);

        // Inflate the rect so blur extends well past glyph edges and
        // covers user margin around the text (cursor halo, line height
        // padding). Bigger pad horizontally because text lines tend to
        // be long but thin.
        const PAD_X: u32 = 24;
        const PAD_Y: u32 = 16;
        let px = x.saturating_sub(PAD_X);
        let py = y.saturating_sub(PAD_Y);
        let pw = (w + 2 * PAD_X).min(img.width() - px);
        let ph = (h + 2 * PAD_Y).min(img.height() - py);

        // Copy the sub-rect out, blur, paste back. `image::imageops::blur`
        // runs on CPU — slower than wgpu but no GPU init cost, and at
        // 10 fps PoC we have budget. σ=25 produces a heavy frost so
        // glyph silhouettes are gone, not just smeared.
        let sub = imageops::crop(&mut img, px, py, pw, ph).to_image();
        let blurred = imageops::blur(&sub, 25.0);
        imageops::replace(&mut img, &blurred, px as i64, py as i64);
    }

    img.save(path).context("save blurred PNG")?;
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

fn embedded_font() -> Option<FontRef<'static>> {
    None
}

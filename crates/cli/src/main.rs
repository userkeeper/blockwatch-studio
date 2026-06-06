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

use bw_capture::window_info::{classify as classify_window, get_foreground_window};
use bw_capture::{default_capturer, Frame};
use bw_core::buffer::StickyHits;
use bw_core::detect::{scan, BBox, SecretKind};
use bw_core::frame_diff::CellHashes;
use bw_core::qr::{is_sensitive, scan_qrs_bgra};
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

    /// Skip OCR when fewer than this fraction of 64×64 cells changed
    /// since the previous OCR pass. 0.0 disables the optimisation
    /// (always OCR), 1.0 effectively disables OCR after frame 0.
    /// Default 0.0 — be safe, always OCR every Nth frame (controlled
    /// by --ocr-every). User can re-enable the optimisation if CPU
    /// is a concern.
    #[arg(long, default_value_t = 0.0)]
    diff_skip_threshold: f32,

    /// Directory for the recorded PNG sequence.
    #[arg(long, default_value = "frames")]
    out_dir: PathBuf,

    /// Live-preview mode. Opens a window mirroring the primary display
    /// with every detected secret blurred in real time. Runs until the
    /// window is closed (Esc or close button). Ignores --record and
    /// --output. This is the Phase 2 deliverable — same pipeline that
    /// will feed the virtual camera in Phase 3, just displayed in a
    /// window for now.
    #[arg(long, default_value_t = false)]
    preview: bool,

    /// Downscale the preview window to this width. The pipeline still
    /// runs at native resolution; only the displayed copy is scaled,
    /// because mirroring a 4K display into a 4K window crushes a lot
    /// of laptops.
    #[arg(long, default_value_t = 1280)]
    preview_width: u32,

    /// Print every OCR'd line + its detection verdict.
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// Dataset-collection mode. Opens a live preview of the primary
    /// display; press `C` to save the current frame as a PNG into
    /// --out-dir with a class-prefixed filename. Used to build the
    /// training set for the YOLOv8 vision model (see ADR-002, Phase 1).
    ///
    /// Example:
    ///     --collect wallet_receive_popup --out-dir dataset/images/train/
    #[arg(long)]
    collect: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let args = Args::parse();

    if let Some(ref class_label) = args.collect {
        run_collect(&args, class_label)
    } else if args.preview {
        run_preview(&args)
    } else if args.record > 0 {
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

    let mut hits = run_detectors(&ocr_result, args.verbose);

    // QR scan.
    let t_qr = Instant::now();
    let qrs = scan_qrs_bgra(&frame.bgra, frame.width, frame.height);
    let mut qr_hits = 0;
    for q in &qrs {
        if is_sensitive(q.kind) {
            hits.push(AnnotatedHit {
                kind: SecretKind::SeedPhrase, // reusing label for now
                bbox: q.bbox,
            });
            qr_hits += 1;
        }
    }
    println!(
        "  {} text region(s) + {} sensitive QR(s) in {:?}",
        hits.len() - qr_hits,
        qr_hits,
        t_qr.elapsed()
    );

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
    let mut ocr_skipped = 0u32;
    let mut total_hits = 0u32;

    // Frame-diff state: we hash the frame on which we last ran OCR, and
    // skip the next OCR pass if the screen barely changed since.
    let mut last_ocr_hashes = CellHashes::new();
    let mut curr_hashes = CellHashes::new();
    // Counter of consecutive frame-diff skips. Forces an OCR re-run
    // after N=3 skips so stickies on a truly static screen get
    // refreshed before they expire on STICKY_LIFETIME_FRAMES.
    let mut consecutive_skips: u32 = 0;
    const MAX_CONSECUTIVE_SKIPS: u32 = 3;

    let t_total = Instant::now();
    for i in 0..total_frames {
        let t_frame = Instant::now();

        // 1. Capture.
        let frame = cap.grab().context("grab frame in record loop")?;

        // 1b. Foreground-window check — informational. See run_preview
        // for the same comment: we do NOT sticky the full window rect,
        // OCR / QR detectors handle surgical sub-region blur.
        if args.verbose {
            if let Ok(Some(info)) = get_foreground_window() {
                let verdict = classify_window(&info);
                if verdict.is_sensitive() {
                    println!(
                        "  [frame {i}] foreground context: {:?} \"{}\"",
                        verdict, info.title
                    );
                }
            }
        }

        // 2. Detect — but only every Nth frame AND only if the screen
        // actually changed enough to be worth re-OCRing.
        if i % args.ocr_every == 0 {
            // Frame-diff gate. On the very first OCR pass we skip the
            // comparison (no previous hashes yet) and always OCR.
            // After MAX_CONSECUTIVE_SKIPS skips in a row, we force an
            // OCR pass even if frame-diff says "static" — otherwise
            // stickies on a pixel-static screen expire and the secret
            // is unblurred.
            let should_ocr = if last_ocr_hashes.is_empty() {
                true
            } else if consecutive_skips >= MAX_CONSECUTIVE_SKIPS {
                if args.verbose {
                    println!("  [frame {i}] force OCR after {consecutive_skips} skips");
                }
                true
            } else {
                let t_diff = Instant::now();
                curr_hashes.recompute(&frame.bgra, frame.width, frame.height);
                let changed = curr_hashes.fraction_changed(&last_ocr_hashes);
                if args.verbose && changed < args.diff_skip_threshold {
                    println!(
                        "  [frame {i}] diff {:.2}% in {:?} → skip OCR ({}/{})",
                        changed * 100.0,
                        t_diff.elapsed(),
                        consecutive_skips + 1,
                        MAX_CONSECUTIVE_SKIPS
                    );
                }
                changed >= args.diff_skip_threshold
            };

            if should_ocr {
                consecutive_skips = 0;
                let t_ocr = Instant::now();
                match ocr.recognise(&frame) {
                    Ok(res) => {
                        ocr_count += 1;
                        let hits = run_detectors(&res, args.verbose);
                        for h in &hits {
                            sticky.add(h.bbox, frame_idx);
                        }
                        // QR scan piggy-backs on the same "we just
                        // re-analysed this frame" budget. Same
                        // sticky set, so blur path is identical.
                        let t_qr = Instant::now();
                        let qrs = scan_qrs_bgra(&frame.bgra, frame.width, frame.height);
                        let mut qr_sensitive = 0u32;
                        for q in &qrs {
                            if is_sensitive(q.kind) {
                                sticky.add(q.bbox, frame_idx);
                                qr_sensitive += 1;
                            }
                        }
                        // Snapshot the hashes we just OCR'd against.
                        last_ocr_hashes.recompute(&frame.bgra, frame.width, frame.height);
                        if args.verbose {
                            println!(
                                "  [frame {i}] OCR {:?} → {} hit(s), QR {:?} → {} sensitive (sticky now {})",
                                t_ocr.elapsed(),
                                hits.len(),
                                t_qr.elapsed(),
                                qr_sensitive,
                                sticky.len()
                            );
                        }
                        total_hits += hits.len() as u32 + qr_sensitive;
                    }
                    Err(e) => {
                        eprintln!("  [frame {i}] OCR error: {e}");
                    }
                }
            } else {
                ocr_skipped += 1;
                consecutive_skips += 1;
                // We DO NOT extend sticky lifetimes on frame-diff skip.
                // Doing so was the cause of the "ghost blur" bug —
                // closing a small UI element produced <0.5% screen
                // change, frame-diff skipped OCR, sticky was extended
                // forever. Stickies now expire on their natural
                // STICKY_LIFETIME_FRAMES schedule. If the screen is
                // pixel-static, frame-diff will skip K cycles, sticky
                // will expire, and the *next* substantial change will
                // re-trigger OCR.
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
        "Done. {} frames in {:?} ({} OCR runs, {} OCR skipped via frame-diff, {} total hits accumulated).",
        total_frames,
        t_total.elapsed(),
        ocr_count,
        ocr_skipped,
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

        // Run scan twice: once on the normalised text (good for BIP-39
        // and other patterns where OCR routinely misreads l→1, o→0),
        // once on the raw text (preserves case + digits for API-key
        // patterns that REQUIRE [0-9A-Z] — AWS, GitHub PAT, etc.).
        // Normalise is destructive: if OCR correctly read `AKIA1OSF…`
        // (real AWS key with digit 1), normalise turns it into
        // `AKIAlOSF…` and AWS_KEY_RE no longer matches.
        let mut hits = scan(&normalised, bbox);
        if normalised != text {
            for h in scan(&text, bbox) {
                if !hits.contains(&h) {
                    hits.push(h);
                }
            }
        }
        // Third pass: collapse whitespace and re-scan. Catches the
        // case where OCR inserts a phantom space mid-token
        // (`sk-abc def…` instead of `sk-abcdef…`), which breaks
        // every API-key regex.
        let collapsed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        if collapsed.len() >= 16 && collapsed != text {
            for h in scan(&collapsed, bbox) {
                if !hits.contains(&h) {
                    hits.push(h);
                }
            }
        }
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

    // Second pass: vertical stitch of hex-only rows. Wallet UIs (Phantom,
    // MetaMask) routinely wrap a 64-hex address across 2-3 lines, so a
    // single row never has enough hex chars to fire HEX_PRIV_RE. We
    // walk rows top-to-bottom, find runs of "pure-hex" rows that are
    // vertically adjacent, concatenate their hex content, and if the
    // total length matches an address shape (40 / 64 hex chars, with
    // optional `0x` prefix) we add every contributing row's bbox as
    // a SeedPhrase-tagged hit (label reused to surface "secret blurred"
    // — kind enum extension is a future cleanup).
    let mut sorted_rows: Vec<(usize, BBox, String)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.bbox(), r.text()))
        .collect();
    sorted_rows.sort_by_key(|t| t.1.y);

    let mut i = 0;
    while i < sorted_rows.len() {
        if let Some(_) = row_as_hex_chunk(&sorted_rows[i].2) {
            // Start of a candidate run. Greedily extend while the
            // next row is also pure-hex AND vertically adjacent.
            let mut run_end = i + 1;
            let mut total_hex = row_as_hex_chunk(&sorted_rows[i].2).unwrap().len();
            while run_end < sorted_rows.len() {
                let prev = &sorted_rows[run_end - 1].1;
                let curr = &sorted_rows[run_end].1;
                let v_gap = curr.y as i32 - (prev.y as i32 + prev.h as i32);
                // Allow up to 1× line-height of vertical gap.
                let allowed_gap = prev.h.max(curr.h) as i32;
                if v_gap > allowed_gap || v_gap < -(curr.h as i32) {
                    break;
                }
                let Some(chunk) = row_as_hex_chunk(&sorted_rows[run_end].2) else {
                    break;
                };
                total_hex += chunk.len();
                run_end += 1;
            }
            // Run is [i, run_end). Check if total hex looks like an
            // address: 40 (EVM), 64 (Sui/Aptos/priv key). We accept
            // ±1 to absorb OCR drop-out at a chunk boundary.
            let looks_like_address = matches!(total_hex, 39..=41 | 63..=65);
            if looks_like_address && run_end - i >= 2 {
                if verbose {
                    println!(
                        "  ⚠ vertical stitch [{}..{}): {} hex chars → secret",
                        i, run_end, total_hex
                    );
                }
                for r in &sorted_rows[i..run_end] {
                    all.push(AnnotatedHit {
                        kind: SecretKind::HexPrivateKey,
                        bbox: r.1,
                    });
                }
            }
            i = run_end;
        } else {
            i += 1;
        }
    }

    all
}

/// If `text` is composed only of hex characters (plus an optional
/// `0x` prefix and ignoring whitespace), return the concatenated hex
/// string. Otherwise return `None`. Used by the vertical-stitch
/// pass to identify rows that *might* be part of a wrapped address.
fn row_as_hex_chunk(text: &str) -> Option<String> {
    // Drop the `0x` if present (only at the very start).
    let t = text.trim();
    let body = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    let mut out = String::new();
    for c in body.chars() {
        if c.is_ascii_whitespace() {
            continue;
        }
        if c.is_ascii_hexdigit() {
            out.push(c);
        } else {
            return None;
        }
    }
    // Need at least 8 hex chars to bother — otherwise it's just "ff" type tokens.
    if out.len() >= 8 {
        Some(out)
    } else {
        None
    }
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
        SecretKind::AwsAccessKey => "AWS KEY",
        SecretKind::GithubToken => "GITHUB TOKEN",
        SecretKind::StripeKey => "STRIPE KEY",
        SecretKind::LlmApiKey => "API KEY",
        SecretKind::SlackToken => "SLACK TOKEN",
        SecretKind::TwilioSid => "TWILIO SID",
        SecretKind::JsonWebToken => "JWT",
        SecretKind::HighEntropyToken => "TOKEN",
        SecretKind::EnvLineValue => "ENV",
    }
}

fn embedded_font() -> Option<FontRef<'static>> {
    None
}

// ─── Dataset collector (Phase 1 of ML migration) ────────────────────

/// Live capture loop with manual frame saving. Used to assemble the
/// training set for the YOLOv8 popup detector — open whatever window
/// represents the target class (a Phantom receive popup, a Notepad
/// open on `.env`, etc.), press `C`, repeat. Files are saved as
/// `<class>_<unix-millis>.png` so they sort by capture time.
fn run_collect(args: &Args, class_label: &str) -> Result<()> {
    use minifb::{Key, Window, WindowOptions};

    println!(
        "BlockWatch Studio — dataset collector\n  class: {}\n  out:   {}\nPress C to save current frame, Esc to exit.",
        class_label,
        args.out_dir.display(),
    );
    std::fs::create_dir_all(&args.out_dir).context("create out_dir")?;

    let mut cap = default_capturer().context("init capturer")?;
    let probe = cap.grab().context("probe frame")?;
    let src_w = probe.width;
    let src_h = probe.height;
    let dst_w = args.preview_width.min(src_w);
    let dst_h = (src_h as f32 * (dst_w as f32 / src_w as f32)) as u32;

    let mut window = Window::new(
        &format!("BlockWatch Studio — collect: {}", class_label),
        dst_w as usize,
        dst_h as usize,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("minifb window: {e}"))?;
    window.set_target_fps(args.fps as usize);

    let mut pixel_buf: Vec<u32> = vec![0u32; (dst_w * dst_h) as usize];
    let mut frame = probe;
    let mut saved = 0u32;
    // Debounce so a single C-keypress saves exactly one frame instead
    // of spamming N consecutive frames while the key is held.
    let mut c_was_down = false;

    loop {
        if !window.is_open() || window.is_key_down(Key::Escape) {
            break;
        }

        // Save on C press (edge-triggered).
        let c_is_down = window.is_key_down(Key::C);
        if c_is_down && !c_was_down {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let filename = format!("{}_{}.png", class_label, ts);
            let path = args.out_dir.join(&filename);

            let img = frame_to_rgba_image(&frame)?;
            img.save(&path).context("save dataset PNG")?;
            saved += 1;
            println!("  ✓ saved [{}] {}", saved, filename);
        }
        c_was_down = c_is_down;

        // Render the live preview UNALTERED — we deliberately don't
        // run detection here; the point is to capture ground truth
        // for labelling.
        let img = frame_to_rgba_image(&frame)?;
        downscale_into_argb(&img, dst_w, dst_h, &mut pixel_buf);
        window
            .update_with_buffer(&pixel_buf, dst_w as usize, dst_h as usize)
            .map_err(|e| anyhow::anyhow!("window update: {e}"))?;

        frame = match cap.grab() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("capture error: {e} (exiting)");
                break;
            }
        };
    }

    println!("Collector closed. {saved} frame(s) saved to {}", args.out_dir.display());
    Ok(())
}

// ─── Live preview (Phase 2) ─────────────────────────────────────────

/// Open a window mirroring the primary display, run the capture + OCR
/// + blur pipeline at the requested fps, push each blurred frame to
/// the window. Runs until the window is closed.
///
/// Architecture mirrors `run_record` 1:1 — the only difference is
/// where the output frame goes (window pixel buffer vs PNG file).
/// This makes it the cleanest possible Phase 3 swap-target for the
/// virtual-camera writer: replace one function call.
fn run_preview(args: &Args) -> Result<()> {
    use minifb::{Key, Window, WindowOptions};

    println!(
        "BlockWatch Studio — Phase 2 preview @ {} fps, OCR every {} frame(s). Esc / close to exit.",
        args.fps, args.ocr_every
    );

    let mut cap = default_capturer().context("init capturer")?;
    let mut ocr = default_backend().context("init OCR backend")?;

    // Grab one frame to learn the display resolution.
    let probe = cap.grab().context("probe frame")?;
    let src_w = probe.width;
    let src_h = probe.height;
    let dst_w = args.preview_width.min(src_w);
    let dst_h = (src_h as f32 * (dst_w as f32 / src_w as f32)) as u32;

    let mut window = Window::new(
        "BlockWatch Shield Studio — live preview",
        dst_w as usize,
        dst_h as usize,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("minifb window: {e}"))?;
    window.set_target_fps(args.fps as usize);

    let mut sticky = StickyHits::default();
    let mut frame_idx: u64 = 0;
    let mut last_ocr_hashes = CellHashes::new();
    let mut curr_hashes = CellHashes::new();
    let mut consecutive_skips: u32 = 0;
    const MAX_CONSECUTIVE_SKIPS_PREVIEW: u32 = 3;
    // Scratch buffer for the down-scaled ARGB output. Allocated once.
    let mut pixel_buf: Vec<u32> = vec![0u32; (dst_w * dst_h) as usize];

    let t_start = Instant::now();
    let mut frames_drawn: u64 = 0;
    // The probe frame is "frame 0", reuse it.
    let mut frame = probe;

    loop {
        if !window.is_open() || window.is_key_down(Key::Escape) {
            break;
        }

        // 0. Foreground-window check — informational only.
        // We DO NOT auto-sticky the whole window rect — that produces
        // an enormous blur halo and ruins the UX. The classifier is
        // a CONTEXT hint for verbose logging today; in Phase 1 of the
        // vision migration it becomes one of the input features fed
        // to the YOLOv8 detector to bias confidence inside known
        // sensitive windows. Surgical sub-region blur (QR + text)
        // continues to come from the OCR/QR detectors below.
        if args.verbose {
            if let Ok(Some(info)) = get_foreground_window() {
                let verdict = classify_window(&info);
                if verdict.is_sensitive() {
                    println!(
                        "  [frame {frame_idx}] foreground context: {:?} \"{}\"",
                        verdict, info.title
                    );
                }
            }
        }

        // 1. Detect (rate-limited + frame-diff gated).
        if frame_idx % args.ocr_every as u64 == 0 {
            let should_ocr = if last_ocr_hashes.is_empty() {
                true
            } else if consecutive_skips >= MAX_CONSECUTIVE_SKIPS_PREVIEW {
                true
            } else {
                curr_hashes.recompute(&frame.bgra, frame.width, frame.height);
                curr_hashes.fraction_changed(&last_ocr_hashes) >= args.diff_skip_threshold
            };
            if should_ocr {
                consecutive_skips = 0;
                if let Ok(res) = ocr.recognise(&frame) {
                    let hits = run_detectors(&res, args.verbose);
                    for h in &hits {
                        sticky.add(h.bbox, frame_idx);
                    }
                    let qrs = scan_qrs_bgra(&frame.bgra, frame.width, frame.height);
                    for q in &qrs {
                        if is_sensitive(q.kind) {
                            sticky.add(q.bbox, frame_idx);
                        }
                    }
                    last_ocr_hashes.recompute(&frame.bgra, frame.width, frame.height);
                }
            } else {
                consecutive_skips += 1;
            }
            // (No extend-on-skip — see comment in run_record.)
        }
        sticky.prune(frame_idx);

        // 2. Blur the captured frame in-place (clone first since we
        // need the raw BGRA for hashing on the next iteration).
        let mut img = frame_to_rgba_image(&frame)?;
        for bb in sticky.active() {
            blur_bbox_inplace(&mut img, bb);
        }

        // 3. Downscale to the preview window size + pack as 0xAARRGGBB
        // u32 pixels (minifb's expected format).
        downscale_into_argb(&img, dst_w, dst_h, &mut pixel_buf);
        window
            .update_with_buffer(&pixel_buf, dst_w as usize, dst_h as usize)
            .map_err(|e| anyhow::anyhow!("window update: {e}"))?;
        frames_drawn += 1;
        frame_idx += 1;

        // 4. Grab the NEXT frame for the next iteration. Doing this
        // last means the work above overlaps with whatever DDA needs
        // to do internally for the next acquisition.
        frame = match cap.grab() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("capture error: {e} (exiting)");
                break;
            }
        };
    }

    let elapsed = t_start.elapsed();
    let fps_avg = frames_drawn as f64 / elapsed.as_secs_f64();
    println!(
        "Preview closed. {} frames in {:?} ({:.1} fps).",
        frames_drawn, elapsed, fps_avg
    );
    Ok(())
}

/// Blur one bbox inside `img` in place. Mirrors the padding rules from
/// `write_blurred_frame` so preview and record-mode look identical.
fn blur_bbox_inplace(img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, bb: BBox) {
    let x = bb.x.min(img.width().saturating_sub(1));
    let y = bb.y.min(img.height().saturating_sub(1));
    let w = bb.w.min(img.width().saturating_sub(x)).max(1);
    let h = bb.h.min(img.height().saturating_sub(y)).max(1);
    const PAD_X: u32 = 24;
    const PAD_Y: u32 = 16;
    let px = x.saturating_sub(PAD_X);
    let py = y.saturating_sub(PAD_Y);
    let pw = (w + 2 * PAD_X).min(img.width() - px);
    let ph = (h + 2 * PAD_Y).min(img.height() - py);
    let sub = imageops::crop(img, px, py, pw, ph).to_image();
    let blurred = imageops::blur(&sub, 25.0);
    imageops::replace(img, &blurred, px as i64, py as i64);
}

/// Nearest-neighbour downscale of an RGBA image into a flat u32 buffer
/// (one u32 per pixel, packed as 0x00RRGGBB; minifb ignores the alpha
/// byte). Fast enough at 1080p → 1280-wide preview: ~3 ms.
fn downscale_into_argb(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    dst_w: u32,
    dst_h: u32,
    dst: &mut [u32],
) {
    debug_assert_eq!(dst.len(), (dst_w * dst_h) as usize);
    let sw = src.width();
    let sh = src.height();
    for dy in 0..dst_h {
        let sy = dy * sh / dst_h;
        let src_row_start = (sy * sw * 4) as usize;
        let dst_row_start = (dy * dst_w) as usize;
        for dx in 0..dst_w {
            let sx = dx * sw / dst_w;
            let i = src_row_start + (sx * 4) as usize;
            let r = src.as_raw()[i] as u32;
            let g = src.as_raw()[i + 1] as u32;
            let b = src.as_raw()[i + 2] as u32;
            dst[dst_row_start + dx as usize] = (r << 16) | (g << 8) | b;
        }
    }
}

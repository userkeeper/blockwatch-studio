"""
Synthetic dataset generator for BlockWatch Studio's YOLOv8 model.

This script renders fake screenshots that look like real `.env` files
in editors, terminal credential exports, and seed-phrase reveal
screens, and writes YOLO-format labels for each. Output goes into
`../dataset/images/train/` + `../dataset/labels/train/`.

Why synthetic data:
  - We get hundreds of varied, perfectly-labelled examples in minutes
    instead of hours of manual screenshot taking.
  - The model learns the VISUAL SHAPE of credentials (UPPER_SNAKE_CASE,
    `=`, long alphanumeric value) rather than memorising specific keys.
  - We can vary font, colour, window size, theme — far more
    augmentation than a human would produce by hand.

What this script does NOT cover:
  - `wallet_receive_popup` (class 0): real wallet UIs have brand-
    specific layouts and chrome. Synthetic rendering of those
    produces poor generalisation. Collect those via `--collect`.
  - `portfolio_balance` (class 5): same reason.

Run:
    cd training/
    python -m pip install pillow
    python synth_data.py --count 500
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ─── Class IDs (must match dataset.yaml + bw_vision::ObjectClass) ────
ENV_FILE_EDITOR = 3
CREDENTIAL_TERMINAL = 4
SEED_PHRASE_REVEAL = 1

# ─── Output paths ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
IMG_DIR = ROOT / "dataset" / "images" / "train"
LBL_DIR = ROOT / "dataset" / "labels" / "train"

# ─── Visual variations ──────────────────────────────────────────────
THEMES = [
    # (bg, fg, accent)  — light, dark, dim
    ("#ffffff", "#1f1f1f", "#0066cc"),
    ("#1e1e1e", "#d4d4d4", "#569cd6"),
    ("#282c34", "#abb2bf", "#61afef"),
    ("#fdf6e3", "#586e75", "#268bd2"),
    ("#272822", "#f8f8f2", "#a6e22e"),
]

# Common monospace font candidates (PIL will use the first that loads).
MONO_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cascadiacode.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
]


def load_mono_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in MONO_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


# ─── Fake key generators ────────────────────────────────────────────

def rnd_hex(n: int) -> str:
    return "".join(random.choices(string.hexdigits.lower(), k=n))


def rnd_alnum(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def fake_aws() -> str:
    return f"AWS_ACCESS_KEY_ID=AKIA{rnd_alnum(16).upper()}"


def fake_gh() -> str:
    return f"GH_TOKEN=ghp_{rnd_alnum(36)}"


def fake_stripe() -> str:
    return f"STRIPE_SECRET=sk_live_{rnd_alnum(24)}"


def fake_openai() -> str:
    return f"OPENAI_API_KEY=sk-{rnd_alnum(48)}"


def fake_anthropic() -> str:
    return f"ANTHROPIC_API_KEY=sk-ant-api03-{rnd_alnum(95)}-aaaaaaaa"


def fake_db_url() -> str:
    user = rnd_alnum(8).lower()
    pwd = rnd_alnum(16)
    host = random.choice(["db.example.com", "127.0.0.1", "prod-db.aws"])
    return f"DATABASE_URL=postgresql://{user}:{pwd}@{host}:5432/app"


def fake_jwt() -> str:
    return f"JWT_SECRET=eyJ{rnd_alnum(20)}.eyJ{rnd_alnum(20)}.{rnd_alnum(30)}"


def fake_twilio() -> str:
    return f"TWILIO_SID=AC{rnd_hex(32)}"


def fake_generic() -> str:
    name = random.choice([
        "API_KEY", "SECRET_KEY", "AUTH_TOKEN", "SESSION_SECRET",
        "ENCRYPTION_KEY", "WEBHOOK_SECRET", "CLIENT_SECRET",
    ])
    return f"{name}={rnd_alnum(random.randint(20, 60))}"


def random_env_line() -> str:
    return random.choice([
        fake_aws, fake_gh, fake_stripe, fake_openai, fake_anthropic,
        fake_db_url, fake_jwt, fake_twilio, fake_generic,
    ])()


# BIP-39 — a small embedded list of 50 common words is enough to
# render a fake-looking seed phrase. We don't need every real word.
BIP39_SAMPLE = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb",
    "abstract", "absurd", "abuse", "access", "accident", "account", "ace",
    "acid", "across", "act", "action", "actor", "actress", "actual",
    "adapt", "add", "addict", "address", "adjust", "admit", "adult",
    "advance", "advice", "aerobic", "affair", "afford", "afraid",
    "again", "age", "agent", "agree", "ahead", "aim", "air", "airport",
    "aisle", "alarm", "album", "alcohol", "alert", "alien", "all",
    "alley",
]


def random_seed_phrase(words: int = 12) -> str:
    return " ".join(random.choices(BIP39_SAMPLE, k=words))


# ─── Renderers ──────────────────────────────────────────────────────

def render_env_editor(idx: int) -> tuple[Path, Path, int]:
    """Render a fake editor showing a `.env` file. Returns (image,
    label, class_id)."""
    w, h = 1280, 720
    bg, fg, accent = random.choice(THEMES)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = load_mono_font(random.randint(14, 18))

    # Window chrome
    draw.rectangle([0, 0, w, 32], fill=accent)
    draw.text((10, 8), ".env — myproject", fill="#ffffff", font=font)

    # Body — 5-12 lines of fake env content
    n_lines = random.randint(5, 12)
    lines = [random_env_line() for _ in range(n_lines)]
    y = 50
    body_top = y
    line_height = font.size + 8 if hasattr(font, "size") else 24
    for line in lines:
        # Occasional comment line for realism
        if random.random() < 0.15:
            draw.text((20, y), f"# {random.choice(['production', 'staging', 'do not commit'])}",
                      fill="#888888", font=font)
            y += line_height
        draw.text((20, y), line, fill=fg, font=font)
        y += line_height
    body_bottom = y

    # Label box: the entire .env body area
    x_center = ((20 + 800) / 2) / w
    y_center = ((body_top + body_bottom) / 2) / h
    box_w = 800 / w
    box_h = (body_bottom - body_top) / h

    img_path = IMG_DIR / f"synth_env_{idx:05d}.png"
    lbl_path = LBL_DIR / f"synth_env_{idx:05d}.txt"
    img.save(img_path)
    lbl_path.write_text(
        f"{ENV_FILE_EDITOR} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
    )
    return img_path, lbl_path, ENV_FILE_EDITOR


def render_terminal(idx: int) -> tuple[Path, Path, int]:
    w, h = 1280, 720
    bg, fg, accent = ("#0c0c0c", "#cccccc", "#16c60c")
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = load_mono_font(random.randint(13, 16))

    # PowerShell-like prompt
    prompt = "PS C:\\Users\\dev\\app>"
    y = 20
    line_height = font.size + 6 if hasattr(font, "size") else 22

    n_exports = random.randint(4, 10)
    body_top = y
    for _ in range(n_exports):
        line = "$env:" + random_env_line().replace("=", " = '") + "'"
        draw.text((10, y), prompt, fill=accent, font=font)
        draw.text((10 + len(prompt) * 9, y), line, fill=fg, font=font)
        y += line_height
    body_bottom = y

    x_center = 0.5
    y_center = ((body_top + body_bottom) / 2) / h
    box_w = (w - 20) / w
    box_h = (body_bottom - body_top + 20) / h

    img_path = IMG_DIR / f"synth_term_{idx:05d}.png"
    lbl_path = LBL_DIR / f"synth_term_{idx:05d}.txt"
    img.save(img_path)
    lbl_path.write_text(
        f"{CREDENTIAL_TERMINAL} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
    )
    return img_path, lbl_path, CREDENTIAL_TERMINAL


def render_seed_reveal(idx: int) -> tuple[Path, Path, int]:
    """A wallet-style seed-phrase reveal screen: 12 words in a grid."""
    w, h = 1280, 720
    bg, fg, accent = random.choice([
        ("#1a1a2e", "#eaeaea", "#7c3aed"),
        ("#ffffff", "#1f1f1f", "#6d28d9"),
        ("#0f172a", "#e2e8f0", "#3b82f6"),
    ])
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    title_font = load_mono_font(28)
    word_font = load_mono_font(22)

    draw.text((w / 2 - 200, 60), "Your Recovery Phrase", fill=fg, font=title_font)
    draw.text((w / 2 - 280, 110),
              "Write these 12 words down. Never share them.",
              fill="#999999", font=word_font)

    words = random.choices(BIP39_SAMPLE, k=12)
    # 3 columns × 4 rows grid
    grid_x, grid_y = 280, 200
    cell_w, cell_h = 240, 70
    for i, word in enumerate(words):
        col = i % 3
        row = i // 3
        x = grid_x + col * cell_w
        y = grid_y + row * cell_h
        # word number
        draw.rectangle([x, y, x + 200, y + 50], outline=accent, width=2)
        draw.text((x + 10, y + 14), f"{i+1}. {word}", fill=fg, font=word_font)

    # Label: the full grid area
    x_center = (grid_x + 3 * cell_w / 2) / w
    y_center = (grid_y + 2 * cell_h) / h
    box_w = (3 * cell_w + 20) / w
    box_h = (4 * cell_h + 20) / h

    img_path = IMG_DIR / f"synth_seed_{idx:05d}.png"
    lbl_path = LBL_DIR / f"synth_seed_{idx:05d}.txt"
    img.save(img_path)
    lbl_path.write_text(
        f"{SEED_PHRASE_REVEAL} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
    )
    return img_path, lbl_path, SEED_PHRASE_REVEAL


# ─── Driver ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count", type=int, default=500,
        help="total number of synthetic examples to generate (split equally per class)",
    )
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    per_class = max(1, args.count // 3)
    print(f"Generating {per_class * 3} synthetic examples ({per_class} per class)")

    counters = {"env": 0, "term": 0, "seed": 0}
    for i in range(per_class):
        render_env_editor(i)
        counters["env"] += 1
    for i in range(per_class):
        render_terminal(i)
        counters["term"] += 1
    for i in range(per_class):
        render_seed_reveal(i)
        counters["seed"] += 1

    total = sum(counters.values())
    print(f"\nDone. Generated {total} examples:")
    for k, v in counters.items():
        print(f"  {k:8s} {v}")
    print(f"\nImages: {IMG_DIR}")
    print(f"Labels: {LBL_DIR}")
    print(
        "\nNext step:\n"
        "  1. Collect real wallet_receive_popup + portfolio_balance examples via --collect\n"
        "  2. cd training/ && python -m pip install ultralytics\n"
        "  3. python train_yolov8.py\n"
    )


if __name__ == "__main__":
    main()

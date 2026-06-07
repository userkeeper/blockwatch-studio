"""
Download promotional screenshots of crypto wallet apps from the public
iTunes Search API and the App Store screenshot CDN.

This script does NOT scrape — every endpoint it hits is a public API
or CDN that Apple exposes for app discovery. The screenshots are
official marketing assets that any developer/user can view on the
App Store today.

What you get:
    dataset/raw_wallets/<bundle_id>/<i>.png

Each wallet's bundle returns 5-10 screenshots from their App Store
listing. Across the bundle list below (~12 wallets) you should get
60-120 real, high-resolution wallet UI screenshots — including
receive popups, send screens, balance dashboards, seed reveal flows.

Run:
    cd training/
    python -m pip install requests
    python scrape_wallet_screenshots.py

Next steps (manual):
    1. Run LabelImg over dataset/raw_wallets/, draw bounding boxes
       around the QR + address text for wallet_receive_popup, around
       the seed grid for seed_phrase_reveal, etc.
    2. Move labelled images + .txt files into dataset/images/train/
       + dataset/labels/train/ alongside the synthetic examples.
    3. Run split_dataset.py and train_yolov8.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

# Wallets to fetch. Each entry is (search_term, expected_bundle_prefix).
# The bundle prefix is used to verify we picked the right app — iTunes
# search occasionally returns clones with the right name.
WALLETS = [
    ("Phantom Solana", "app.phantom"),
    ("MetaMask", "io.metamask"),
    ("Trust Wallet", "com.trustwallet"),
    ("Tonkeeper", "com.ton-keeper"),
    ("Backpack Wallet", "com.backpack"),
    ("Rainbow Wallet", "me.rainbow"),
    ("Solflare", "com.solflare"),
    ("Exodus", "com.exodus"),
    ("Ledger Live", "com.ledger.live"),
    ("Coinbase Wallet", "org.toshi"),
    ("Rabby Wallet", "com.debank.rabbymobile"),
    ("Tangem", "com.tangem.tangemcards"),
]

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "dataset" / "raw_wallets"


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; bw-scraper/1.0)"})
    with urlopen(req, timeout=15) as resp:
        return json.load(resp)


def download_binary(url: str, dst: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; bw-scraper/1.0)"})
    with urlopen(req, timeout=15) as resp:
        dst.write_bytes(resp.read())


def fetch_wallet(search_term: str, expected_bundle_prefix: str) -> int:
    """Return the number of screenshots saved."""
    url = (
        f"https://itunes.apple.com/search?term={quote(search_term)}"
        f"&country=us&entity=software&limit=5"
    )
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"  ⚠ search failed: {e}")
        return 0

    # Pick the first result whose bundleId matches our expected prefix.
    apps = data.get("results", [])
    pick = next(
        (a for a in apps if a.get("bundleId", "").startswith(expected_bundle_prefix)),
        None,
    )
    if pick is None:
        # Fall back to the top result.
        pick = apps[0] if apps else None
    if pick is None:
        print(f"  ⚠ no results")
        return 0

    bundle = pick.get("bundleId", "unknown").replace(".", "_")
    name = pick.get("trackName", "unknown")
    screens = pick.get("screenshotUrls", []) + pick.get("ipadScreenshotUrls", [])
    if not screens:
        print(f"  · {name} ({bundle}): no screenshots in listing")
        return 0

    dst_dir = OUT_DIR / bundle
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, url in enumerate(screens):
        dst = dst_dir / f"{i:02d}.png"
        if dst.exists():
            saved += 1
            continue
        try:
            download_binary(url, dst)
            saved += 1
        except Exception as e:
            print(f"    ! {url} → {e}")
        time.sleep(0.2)  # be polite

    print(f"  ✓ {name} ({bundle}): {saved} screenshot(s) saved")
    return saved


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching wallet screenshots into {OUT_DIR}\n")
    total = 0
    for term, prefix in WALLETS:
        print(f"[{term}]")
        total += fetch_wallet(term, prefix)

    print(f"\nDone. {total} screenshot(s) downloaded across {len(WALLETS)} wallets.")
    print("\nNext step:")
    print("  1. Open LabelImg, point at dataset/raw_wallets/")
    print("  2. Draw boxes for wallet_receive_popup / seed_phrase_reveal / etc.")
    print("  3. Move labelled .png + .txt files into dataset/images/train/ + labels/train/")
    print("  4. cd training/ && python split_dataset.py && python train_yolov8.py")


if __name__ == "__main__":
    main()

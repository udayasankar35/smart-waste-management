#!/usr/bin/env python3
"""
Llama 3.2 Model Downloader
Downloads quantised GGUF weights from Hugging Face into module/models/

Usage:
    python download_model.py          # Download 1B (recommended, ~750 MB)
    python download_model.py 3b       # Download 3B (~2.0 GB)
    python download_model.py --check  # Check if model is already present

Bharat AI-SoC Challenge 2026–27 — PS2
"""

import os
import sys
import time
import hashlib

# ─── Model definitions ──────────────────────────────────────────────────────────

MODELS = {
    "1b": {
        "name": "Llama-3.2-1B-Instruct Q4_K_M",
        "description": "Recommended for SoC & laptops (< 1.2 GB RAM)",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "url": (
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/"
            "resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ),
        "size_mb": 750,
    },
    "3b": {
        "name": "Llama-3.2-3B-Instruct Q4_K_M",
        "description": "Higher accuracy for 8 GB+ PCs",
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "url": (
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/"
            "resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        ),
        "size_mb": 2000,
    },
}


# ─── Download helper ─────────────────────────────────────────────────────────────

def download_with_progress(url: str, dest: str, expected_mb: int = 0) -> bool:
    """Download a file with a console progress bar."""
    import urllib.request

    start_time = time.time()

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            elapsed = time.time() - start_time
            speed = (downloaded / (1024 * 1024)) / max(elapsed, 0.1)
            sys.stdout.write(
                f"\r  ⬇  {pct:5.1f}%  |  {mb_done:7.1f} MB / {mb_total:.1f} MB  |  {speed:.1f} MB/s   "
            )
        else:
            mb_done = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  ⬇  {mb_done:.1f} MB downloaded …")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
        print()  # newline after progress bar
        return True
    except Exception as exc:
        print(f"\n❌ Download failed: {exc}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────────

def main():
    # Parse arguments
    choice = "1b"
    check_only = False

    for arg in sys.argv[1:]:
        a = arg.lower().strip("-")
        if a in ("3b", "3"):
            choice = "3b"
        elif a in ("1b", "1"):
            choice = "1b"
        elif a in ("check", "status"):
            check_only = True
        elif a in ("help", "h"):
            print(__doc__)
            sys.exit(0)

    model = MODELS[choice]
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    dest_file = os.path.join(models_dir, model["filename"])

    print()
    print("=" * 64)
    print("  Llama 3.2 Model Downloader")
    print("  Bharat AI-SoC Challenge 2026–27 — PS2")
    print("=" * 64)
    print()
    print(f"  Model      : {model['name']}")
    print(f"  Description: {model['description']}")
    print(f"  Size       : ~{model['size_mb']} MB")
    print(f"  Destination: {dest_file}")
    print()

    # Check existing
    if os.path.isfile(dest_file):
        size_mb = os.path.getsize(dest_file) / (1024 * 1024)
        print(f"  ✅ Model already exists ({size_mb:.1f} MB)")
        print(f"     Path: {dest_file}")
        print()
        print("  To re-download, delete the file and run this script again.")
        return

    if check_only:
        print("  ❌ Model NOT found. Run without --check to download.")
        return

    # Confirm
    print(f"  This will download ~{model['size_mb']} MB. Ensure you have enough disk space")
    print("  and a stable internet connection.")
    print()

    try:
        answer = input("  Proceed with download? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return

    if answer and answer not in ("y", "yes", ""):
        print("  Cancelled.")
        return

    print()
    print(f"  Downloading {model['name']} …")
    print(f"  URL: {model['url']}")
    print()

    success = download_with_progress(model["url"], dest_file, model["size_mb"])

    if success and os.path.isfile(dest_file):
        final_size = os.path.getsize(dest_file) / (1024 * 1024)
        print()
        print(f"  ✅ Download complete!  ({final_size:.1f} MB)")
        print(f"     Saved to: {dest_file}")
        print()
        print("  Next steps:")
        print("    1. Start the app:  python web_app.py")
        print("    2. The chatbot will now use real Llama 3.2 1B inference!")
        print()
    else:
        # Clean up partial download
        if os.path.isfile(dest_file):
            os.remove(dest_file)
        print()
        print("  ❌ Download failed. Please check your internet connection and try again.")
        print(f"     You can also manually download from:\n     {model['url']}")
        print(f"     And place it at: {dest_file}")


if __name__ == "__main__":
    main()

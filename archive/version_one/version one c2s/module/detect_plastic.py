#!/usr/bin/env python3
"""
Plastic Detection CLI
Detect plastic waste in images using the trained YOLOv8 model.

Usage:
    # Detect plastic in a single image
    python detect_plastic.py --image path/to/image.jpg

    # Detect plastic in all images inside a folder
    python detect_plastic.py --folder path/to/folder

    # Save annotated output images to a custom directory
    python detect_plastic.py --folder test_images --output results

    # Adjust confidence threshold
    python detect_plastic.py --image photo.jpg --conf 0.5
"""

import argparse
import os
import sys
import glob
import logging
import time

sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_CONFIG
from waste_detector import WasteDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def collect_images(path: str) -> list[str]:
    """Collect image file paths from a file or folder."""
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        images = []
        for entry in sorted(os.listdir(path)):
            ext = os.path.splitext(entry)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(path, entry))
        return images

    # Try glob pattern
    matched = glob.glob(path)
    return [f for f in matched if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]


def detect_in_images(
    image_paths: list[str],
    output_dir: str | None,
    confidence: float,
    show: bool,
) -> dict:
    """Run plastic detection on a list of images.

    Returns a summary dict with counts.
    """
    import cv2

    # Configure detector
    config = DEFAULT_CONFIG.detection
    config.confidence_threshold = confidence

    detector = WasteDetector(config)

    if not detector.load_model():
        logger.error("❌ Failed to load YOLOv8 model (yolo/weights/weights.pt)")
        sys.exit(1)

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_plastic = 0
    total_images = len(image_paths)
    results_summary = []

    print()
    print("=" * 60)
    print("  🔍 PLASTIC DETECTION — YOLOv8")
    print("=" * 60)
    print(f"  Model      : yolo/weights/weights.pt")
    print(f"  Confidence : {confidence:.0%}")
    print(f"  Images     : {total_images}")
    if output_dir:
        print(f"  Output     : {output_dir}/")
    print("=" * 60)
    print()

    for idx, image_path in enumerate(image_paths, 1):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"  ⚠️  Could not read: {image_path}")
            continue

        # Run detection
        start = time.time()
        waste_frame = detector.detect(image, idx)
        elapsed_ms = (time.time() - start) * 1000

        # Filter only plastic detections
        plastic_detections = [
            d for d in waste_frame.detections if d.class_name == "plastic"
        ]
        num_plastic = len(plastic_detections)
        total_plastic += num_plastic

        # Status icon
        status = "✅ PLASTIC FOUND" if num_plastic > 0 else "❌ No plastic"
        basename = os.path.basename(image_path)

        print(
            f"  [{idx:3d}/{total_images}]  {basename:30s}  "
            f"{status}  ({num_plastic} detection{'s' if num_plastic != 1 else ''})  "
            f"[{elapsed_ms:.0f}ms]"
        )

        # Print details for each detection
        for det in plastic_detections:
            print(
                f"           └─ PLASTIC  {det.confidence:.0%}  "
                f"bbox=({int(det.bbox[0])},{int(det.bbox[1])},{int(det.bbox[2])},{int(det.bbox[3])})"
            )

        # Save annotated image
        if output_dir or show:
            annotated = detector.draw_detections(image, waste_frame)

            if output_dir:
                name, ext = os.path.splitext(basename)
                out_path = os.path.join(output_dir, f"{name}_detected{ext}")
                cv2.imwrite(out_path, annotated)

            if show:
                cv2.imshow("Plastic Detection | Press any key", annotated)
                cv2.waitKey(0)

        results_summary.append({
            "file": basename,
            "plastic_count": num_plastic,
            "detections": [d.to_dict() for d in plastic_detections],
        })

    if show:
        import cv2
        cv2.destroyAllWindows()

    # Print summary
    print()
    print("─" * 60)
    print(f"  📊 SUMMARY")
    print(f"     Images scanned    : {total_images}")
    print(f"     Plastic detected  : {total_plastic} object{'s' if total_plastic != 1 else ''}")
    images_with = sum(1 for r in results_summary if r["plastic_count"] > 0)
    print(f"     Images with plastic: {images_with}/{total_images}")
    if output_dir:
        print(f"     Annotated images  : {output_dir}/")
    print("─" * 60)
    print()

    return {
        "total_images": total_images,
        "total_plastic": total_plastic,
        "images_with_plastic": images_with,
        "results": results_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Detect plastic waste in images using trained YOLOv8 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_plastic.py --image test_images/Plastic.jpeg
  python detect_plastic.py --folder test_images
  python detect_plastic.py --folder test_images --output results
  python detect_plastic.py --image photo.jpg --conf 0.5 --show
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--folder", type=str, help="Path to a folder containing images")

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for annotated images (default: results/)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each annotated image (press any key to advance)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save annotated output images",
    )

    args = parser.parse_args()

    # Collect images
    path = args.image or args.folder
    images = collect_images(path)

    if not images:
        print(f"❌ No images found at: {path}")
        print(f"   Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(1)

    output_dir = None if args.no_save else args.output

    detect_in_images(
        image_paths=images,
        output_dir=output_dir,
        confidence=args.conf,
        show=args.show,
    )


if __name__ == "__main__":
    main()

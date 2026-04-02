"""
generate_synth_word_data.py

Generates synthetic training data for the word CNN by compositing word
templates from data/templates/ onto random background crops.

Templates are greyscale JPGs with dark backgrounds — pixel brightness is
used directly as the alpha channel (black = fully transparent, white = fully
opaque), so no manual transparency work is needed beforehand.

Output goes to data/word_training_data/{class_name}/synth_NNNNN.png.
The WordDataset.resize_pad() call at train time handles the final 32×256
normalisation, so output size here just needs to be reasonable.

Usage:
    python generate_synth_word_data.py                    # 300 samples/class
    python generate_synth_word_data.py --samples 500
    python generate_synth_word_data.py --append           # add to existing synth files
    python generate_synth_word_data.py --classes iron gold tin
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


TEMPLATE_DIRS = [
    "data/templates/resources",
    "data/templates/deposits",
]
BG_DIR = "data/synth_data/backgrounds"
OUTPUT_DIR = "data/word_training_data"


def load_templates(template_dirs: list[Path]) -> dict[str, np.ndarray]:
    """
    Load all JPG/PNG templates from the given directories as greyscale.
    If a class appears in multiple dirs, the first one wins.
    Returns {class_name: grey_array}.
    """
    templates: dict[str, np.ndarray] = {}
    for d in template_dirs:
        for path in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
            name = path.stem
            if name not in templates:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[name] = img
    return templates


def load_backgrounds(bg_dir: Path) -> list[np.ndarray]:
    """Load all background images as BGR arrays."""
    bgs = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in sorted(bg_dir.glob(pattern)):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                bgs.append(img)
    return bgs


def random_crop(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return a random (h×w) BGR crop, tiling the image if necessary."""
    ih, iw = image.shape[:2]
    if ih < h or iw < w:
        reps_h = (h // ih) + 2
        reps_w = (w // iw) + 2
        image = np.tile(image, (reps_h, reps_w, 1))
        ih, iw = image.shape[:2]
    y = random.randint(0, ih - h)
    x = random.randint(0, iw - w)
    return image[y : y + h, x : x + w].copy()


def composite(
    template_grey: np.ndarray,
    bg_bgr: np.ndarray,
    scale_range: tuple[float, float] = (2.0, 3.5),
    pad_v: int = 4,
    pad_h: int = 6,
) -> np.ndarray:
    """
    Alpha-composite a word template over a background patch.

    Steps:
      1. Scale the template to a random height in scale_range × original height.
      2. Use the greyscale pixel value as the alpha channel
         (dark pixels become transparent, bright pixels become opaque).
      3. Crop a background patch the same size as the padded template canvas.
      4. Alpha-blend the white text over the background in greyscale.

    Returns a greyscale uint8 image.
    """
    src_h, src_w = template_grey.shape

    # --- scale ---
    scale = random.uniform(*scale_range)
    new_h = max(4, round(src_h * scale))
    new_w = max(4, round(src_w * scale))
    resized = cv2.resize(template_grey, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- canvas size ---
    canvas_h = new_h + 2 * pad_v
    canvas_w = new_w + 2 * pad_h

    # --- background crop (greyscale) ---
    bg_crop = random_crop(bg_bgr, canvas_h, canvas_w)
    bg_grey = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- alpha from brightness: bright text = opaque, dark = transparent ---
    alpha = resized.astype(np.float32) / 255.0  # [0, 1]

    # Place template into canvas with padding
    result = bg_grey.copy()
    y0, x0 = pad_v, pad_h
    y1, x1 = y0 + new_h, x0 + new_w

    text_brightness = resized.astype(np.float32)
    result[y0:y1, x0:x1] = (
        alpha * text_brightness + (1.0 - alpha) * bg_grey[y0:y1, x0:x1]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def augment(img: np.ndarray) -> np.ndarray:
    """Apply random photometric augmentations to a greyscale image."""
    # Brightness / contrast
    contrast = random.uniform(0.75, 1.30)
    brightness = random.randint(-25, 25)
    img = np.clip(contrast * img.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

    # Gaussian noise (50% chance)
    if random.random() < 0.5:
        sigma = random.uniform(2.0, 10.0)
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Gaussian blur (30% chance)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


def next_synth_index(class_dir: Path) -> int:
    """Return the next available synth index for --append mode."""
    existing = list(class_dir.glob("synth_*.png"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        try:
            indices.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return (max(indices) + 1) if indices else 0


def generate_class(
    name: str,
    template: np.ndarray,
    backgrounds: list[np.ndarray],
    class_dir: Path,
    n_samples: int,
    start_index: int,
) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        bg = random.choice(backgrounds)
        img = composite(template, bg)
        img = augment(img)
        cv2.imwrite(str(class_dir / f"synth_{start_index + i:05d}.png"), img)
    print(f"  {name:20s}  {n_samples} images  ->  {class_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic word training images for the word CNN."
    )
    parser.add_argument(
        "--template-dirs", nargs="+",
        default=TEMPLATE_DIRS,
        help="Directories containing word template images (default: data/templates/resources + deposits)",
    )
    parser.add_argument(
        "--bg-dir", default=BG_DIR,
        help=f"Background images directory (default: {BG_DIR})",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Root word_training_data folder (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--samples", type=int, default=300,
        help="Number of synthetic samples per class (default: 300)",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Limit generation to specific class names (default: all templates)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing synth_*.png files instead of starting from index 0",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    template_dirs = [Path(d) for d in args.template_dirs]
    bg_dir = Path(args.bg_dir)
    output_dir = Path(args.output_dir)

    print("Loading word templates...")
    templates = load_templates(template_dirs)
    if not templates:
        raise RuntimeError(f"No templates found in {template_dirs}")
    print(f"  Found {len(templates)} class(es): {sorted(templates)}")

    print("Loading backgrounds...")
    backgrounds = load_backgrounds(bg_dir)
    if not backgrounds:
        raise RuntimeError(f"No background images found in {bg_dir}")
    print(f"  Loaded {len(backgrounds)} background(s)")

    target_classes = args.classes if args.classes else sorted(templates)

    print(f"\nGenerating {args.samples} samples per class...")
    print("-" * 60)

    for name in target_classes:
        if name not in templates:
            print(f"  {name:20s}  SKIP — no template found")
            continue
        class_dir = output_dir / name
        start_index = next_synth_index(class_dir) if args.append else 0
        generate_class(name, templates[name], backgrounds, class_dir, args.samples, start_index)

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

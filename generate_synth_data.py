"""
generate_synth_data.py

Generates synthetic training data for the digit CNN by alpha-compositing
transparent digit templates (data/synth_data/*.png) onto random background
crops (data/synth_data/backgrounds/), then saving 28x28 grayscale images
into data/training_data/{class}/.

Usage:
    python generate_synth_data.py                         # 300 samples/class, all chars
    python generate_synth_data.py --samples 500           # 500 samples/class
    python generate_synth_data.py --chars 0123456789      # digits only
    python generate_synth_data.py --append                # add to existing synth files
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# Maps special chars to their directory names (matching cnn/dataset.py)
CHAR_DIR_MAP = {
    ".": "dot",
    "-": "dash",
    "%": "percent",
    ",": "comma",
}

# Maps template filename stems to character
TEMPLATE_FILES = {
    "0": "0.png", "1": "1.png", "2": "2.png", "3": "3.png", "4": "4.png",
    "5": "5.png", "6": "6.png", "7": "7.png", "8": "8.png", "9": "9.png",
    ".": "dot.png", "%": "percent.png", "-": "dash.png", ",": "comma.png",
}


def load_templates(synth_dir: Path) -> dict[str, np.ndarray]:
    """Load RGBA digit template PNGs. Returns {char: BGRA array}."""
    templates = {}
    for char, filename in TEMPLATE_FILES.items():
        path = synth_dir / filename
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Warning: could not read {path}")
            continue
        # Ensure BGRA (add alpha channel if missing)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        templates[char] = img
    return templates


def load_backgrounds(bg_dir: Path) -> list[np.ndarray]:
    """Load all background images from bg_dir as BGR arrays."""
    backgrounds = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in sorted(bg_dir.glob(pattern)):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                backgrounds.append(img)
    return backgrounds


def random_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Return a random (size x size) BGR crop from image, tiling if necessary."""
    h, w = image.shape[:2]
    # Tile so the image is at least `size` in each dimension
    if h < size or w < size:
        reps_h = (size // h) + 2
        reps_w = (size // w) + 2
        image = np.tile(image, (reps_h, reps_w, 1))
        h, w = image.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return image[y : y + size, x : x + size].copy()


def composite(
    digit_bgra: np.ndarray,
    bg_bgr: np.ndarray,
    target_size: int = 28,
    scale_range: tuple[float, float] = (0.65, 0.95),
    max_jitter: int = 3,
) -> np.ndarray:
    """
    Alpha-composite a transparent digit onto a background patch.

    1. Crop a (target_size x target_size) patch from bg_bgr randomly.
    2. Scale the digit to a random fraction of target_size (aspect-preserving).
    3. Place the digit near-centre with a small random jitter.
    4. Alpha-blend digit over background.
    5. Return grayscale (target_size x target_size) uint8 image.
    """
    canvas = random_crop(bg_bgr, target_size).astype(np.float32)

    # --- scale digit (preserve aspect ratio, height drives scaling) ---
    scale = random.uniform(*scale_range)
    digit_h = max(4, int(target_size * scale))
    src_h, src_w = digit_bgra.shape[:2]
    digit_w = max(4, int(src_w * digit_h / src_h))
    digit_w = min(digit_w, target_size)  # don't exceed canvas width
    digit_resized = cv2.resize(digit_bgra, (digit_w, digit_h), interpolation=cv2.INTER_AREA)

    # --- split channels ---
    b, g, r, a = cv2.split(digit_resized)
    digit_bgr = cv2.merge([b, g, r]).astype(np.float32)
    alpha = a.astype(np.float32) / 255.0  # (digit_h, digit_w)

    # --- centre + jitter placement ---
    pad_h = target_size - digit_h
    pad_w = target_size - digit_w
    dy = pad_h // 2 + random.randint(-min(max_jitter, pad_h // 2), min(max_jitter, max(0, pad_h - pad_h // 2)))
    dx = pad_w // 2 + random.randint(-min(max_jitter, pad_w // 2), min(max_jitter, max(0, pad_w - pad_w // 2)))
    dy = max(0, min(dy, target_size - digit_h))
    dx = max(0, min(dx, target_size - digit_w))

    # --- alpha blend ---
    y1, y2 = dy, dy + digit_h
    x1, x2 = dx, dx + digit_w
    alpha_3ch = alpha[:, :, np.newaxis]  # broadcast over BGR
    canvas[y1:y2, x1:x2] = alpha_3ch * digit_bgr + (1.0 - alpha_3ch) * canvas[y1:y2, x1:x2]

    result = np.clip(canvas, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


def augment(img: np.ndarray) -> np.ndarray:
    """Apply random photometric augmentations to a grayscale image."""
    # Random brightness / contrast
    contrast = random.uniform(0.75, 1.30)
    brightness = random.randint(-25, 25)
    img = np.clip(contrast * img.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

    # Gaussian noise (50% chance)
    if random.random() < 0.5:
        sigma = random.uniform(2.0, 12.0)
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Gaussian blur (30% chance)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


def next_synth_index(class_dir: Path) -> int:
    """Return the next available synth index (max existing + 1, or 0)."""
    if not class_dir.exists():
        return 0
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
    char: str,
    digit_bgra: np.ndarray,
    backgrounds: list[np.ndarray],
    output_dir: Path,
    n_samples: int,
    start_index: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        bg = random.choice(backgrounds)
        img = composite(digit_bgra, bg)
        img = augment(img)
        filename = f"synth_{start_index + i:05d}.png"
        cv2.imwrite(str(output_dir / filename), img)
    print(f"  [{char!r:>3}]  {n_samples} images  ->  {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training images for the digit CNN."
    )
    parser.add_argument(
        "--synth-dir", default="data/synth_data",
        help="Folder containing digit template PNGs and backgrounds/ subfolder (default: data/synth_data)",
    )
    parser.add_argument(
        "--output-dir", default="data/training_data",
        help="Root training_data folder (default: data/training_data)",
    )
    parser.add_argument(
        "--samples", type=int, default=300,
        help="Number of synthetic samples to generate per class (default: 300)",
    )
    parser.add_argument(
        "--chars", default="0123456789.%",
        help="Characters to generate (default: '0123456789.%%')",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing synth_*.png files instead of overwriting from index 0",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    synth_dir = Path(args.synth_dir)
    output_dir = Path(args.output_dir)
    bg_dir = synth_dir / "backgrounds"

    print("Loading digit templates...")
    templates = load_templates(synth_dir)
    if not templates:
        raise RuntimeError(f"No templates found in {synth_dir}")
    print(f"  Loaded {len(templates)} template(s): {sorted(templates)}")

    print("Loading backgrounds...")
    backgrounds = load_backgrounds(bg_dir)
    if not backgrounds:
        raise RuntimeError(f"No background images found in {bg_dir}")
    print(f"  Loaded {len(backgrounds)} background(s)")

    print(f"\nGenerating {args.samples} samples per class for chars: {args.chars!r}")
    print("-" * 60)

    missing = []
    for char in args.chars:
        if char not in templates:
            missing.append(char)
            print(f"  [{char!r:>3}]  SKIP — no template PNG found")
            continue

        dir_name = CHAR_DIR_MAP.get(char, char)
        class_dir = output_dir / dir_name
        start_index = next_synth_index(class_dir) if args.append else 0
        generate_class(char, templates[char], backgrounds, class_dir, args.samples, start_index)

    print("-" * 60)
    if missing:
        print(f"Skipped {len(missing)} char(s) with no template: {missing}")
    print("Done.")


if __name__ == "__main__":
    main()

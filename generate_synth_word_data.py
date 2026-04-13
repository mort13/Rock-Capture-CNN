"""
generate_synth_word_data.py

Generates synthetic training data for the word CNN by compositing word
templates from data/templates/ onto random background crops.

Templates can be either:
  - Transparent PNG files (with alpha channel) — the transparency is used directly
  - Greyscale images — pixel brightness is used as the alpha channel

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
SUFFIX_DIR = "data/templates/suffix"


def load_templates(template_dirs: list[Path]) -> dict[str, dict]:
    """
    Load all JPG/PNG templates from the given directories.
    Returns {class_name: {
        'image': rgba_or_grey_array,
        'has_alpha': bool
    }}.
    
    Transparent PNGs are loaded with alpha channel (4 channels BGRA).
    Greyscale images are loaded without alpha (will use brightness as alpha).
    """
    templates: dict[str, dict] = {}
    for d in template_dirs:
        for path in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
            name = path.stem
            if name not in templates:
                # Load with potential alpha channel
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    has_alpha = img.ndim == 3 and img.shape[2] == 4
                    templates[name] = {
                        'image': img,
                        'has_alpha': has_alpha
                    }
    return templates


def load_suffix_mapping(suffix_dir: Path) -> dict[str, list[str]]:
    """
    Load suffix_mapping.json from the suffix directory.
    Returns {class_name: ["ore", ...]} for words that have suffixes.
    Words absent from the mapping will never receive a suffix.
    """
    import json
    mapping_path = suffix_dir / "suffix_mapping.json"
    if not mapping_path.exists():
        return {}
    with open(mapping_path, encoding="utf-8") as f:
        raw = json.load(f)
    # Normalise: ensure every value is a list
    result: dict[str, list[str]] = {}
    for word, val in raw.items():
        result[word] = val if isinstance(val, list) else [val]
    return result


def load_suffix_templates(suffix_dir: Path) -> dict[str, dict]:
    """
    Load transparent PNG suffix images (e.g. ore.png, raw.png).
    Returns {suffix_name: {'image': bgra_array, 'has_alpha': True}}.
    """
    templates: dict[str, dict] = {}
    for path in sorted(suffix_dir.glob("*.png")):
        if path.stem == "suffix_mapping":
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            has_alpha = img.ndim == 3 and img.shape[2] == 4
            templates[path.stem] = {"image": img, "has_alpha": has_alpha}
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
    template_data: dict,
    bg_bgr: np.ndarray,
    scale_range: tuple[float, float] = (2.0, 3.5),
    pad_v: int = 4,
    pad_h: int = 6,
) -> np.ndarray:
    """
    Alpha-composite a word template over a background patch.

    For transparent PNGs: uses the alpha channel directly.
    For greyscale images: uses pixel brightness as alpha channel.

    Steps:
      1. Scale the template to a random height in scale_range × original height.
      2. Use alpha channel (for PNG) or greyscale brightness (for JPG) as transparency.
      3. Crop a background patch the same size as the padded template canvas.
      4. Alpha-blend the text over the background in greyscale.

    Returns a greyscale uint8 image.
    """
    template_img = template_data['image']
    has_alpha = template_data['has_alpha']
    
    # Extract the image part (strip alpha if present)
    if has_alpha:
        template_bgr = template_img[:, :, :3]
        template_grey = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        alpha_channel = template_img[:, :, 3].astype(np.float32) / 255.0
    else:
        # For greyscale, convert to grey and use brightness as alpha
        if template_img.ndim == 3:
            template_grey = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            template_grey = template_img
        alpha_channel = template_grey.astype(np.float32) / 255.0

    src_h, src_w = template_grey.shape

    # --- scale ---
    scale = random.uniform(*scale_range)
    new_h = max(4, round(src_h * scale))
    new_w = max(4, round(src_w * scale))
    resized = cv2.resize(template_grey, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    alpha_resized = cv2.resize(alpha_channel, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- canvas size ---
    canvas_h = new_h + 2 * pad_v
    canvas_w = new_w + 2 * pad_h

    # --- background crop (greyscale) ---
    bg_crop = random_crop(bg_bgr, canvas_h, canvas_w)
    bg_grey = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Place template into canvas with padding
    result = bg_grey.copy()
    y0, x0 = pad_v, pad_h
    y1, x1 = y0 + new_h, x0 + new_w

    text_brightness = resized.astype(np.float32)
    result[y0:y1, x0:x1] = (
        alpha_resized * text_brightness + (1.0 - alpha_resized) * bg_grey[y0:y1, x0:x1]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def composite_with_suffix(
    template_data: dict,
    suffix_data: dict,
    bg_bgr: np.ndarray,
    scale_range: tuple[float, float] = (2.0, 3.5),
    gap: int = 6,
    pad_v: int = 4,
    pad_h: int = 6,
) -> np.ndarray:
    """
    Alpha-composite a word template followed by a suffix template over a
    background patch, placed side-by-side with a small gap.

    Both templates are scaled by the same random factor so the suffix
    matches the word height.  The result is greyscale uint8.
    """
    scale = random.uniform(*scale_range)

    def _prepare(data: dict):
        img = data["image"]
        has_alpha = data["has_alpha"]
        if has_alpha:
            grey = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha = img[:, :, 3].astype(np.float32) / 255.0
        else:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            alpha = grey.astype(np.float32) / 255.0
        h, w = grey.shape
        new_h = max(4, round(h * scale))
        new_w = max(4, round(w * scale))
        grey_r = cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha_r = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return grey_r, alpha_r

    word_grey, word_alpha = _prepare(template_data)
    suf_grey, suf_alpha = _prepare(suffix_data)

    wh, ww = word_grey.shape
    sh, sw = suf_grey.shape

    canvas_h = max(wh, sh) + 2 * pad_v
    canvas_w = ww + gap + sw + 2 * pad_h

    bg_crop = random_crop(bg_bgr, canvas_h, canvas_w)
    result = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Vertically centre each element
    def _blit(grey, alpha, x_off):
        h, w = grey.shape
        y0 = pad_v + (canvas_h - 2 * pad_v - h) // 2
        y1 = y0 + h
        x1 = x_off + w
        result[y0:y1, x_off:x1] = (
            alpha * grey.astype(np.float32)
            + (1.0 - alpha) * result[y0:y1, x_off:x1]
        )

    _blit(word_grey, word_alpha, pad_h)
    _blit(suf_grey, suf_alpha, pad_h + ww + gap)

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
    template_data: dict,
    backgrounds: list[np.ndarray],
    class_dir: Path,
    n_samples: int,
    start_index: int,
    suffix_templates: dict[str, dict] | None = None,
    suffix_options: list[str] | None = None,
) -> None:
    """
    Generate n_samples images for `name`.  When suffix_templates and
    suffix_options are provided, exactly half the samples will be plain and
    half will have a randomly chosen suffix composited to the right.
    """
    class_dir.mkdir(parents=True, exist_ok=True)

    use_suffix = (
        suffix_templates is not None
        and suffix_options is not None
        and len(suffix_options) > 0
        and all(s in suffix_templates for s in suffix_options)
    )

    plain_count = n_samples // 2 if use_suffix else n_samples
    suffix_count = n_samples - plain_count if use_suffix else 0

    for i in range(plain_count):
        bg = random.choice(backgrounds)
        img = composite(template_data, bg)
        img = augment(img)
        cv2.imwrite(str(class_dir / f"synth_{start_index + i:05d}.png"), img)

    for j in range(suffix_count):
        bg = random.choice(backgrounds)
        chosen_suffix = random.choice(suffix_options)
        img = composite_with_suffix(template_data, suffix_templates[chosen_suffix], bg)
        img = augment(img)
        cv2.imwrite(str(class_dir / f"synth_{start_index + plain_count + j:05d}.png"), img)

    suffix_note = f" (+{suffix_count} w/ suffix)" if use_suffix else ""
    print(f"  {name:20s}  {n_samples} images{suffix_note}  ->  {class_dir}")


def generate_empty_class(
    backgrounds: list[np.ndarray],
    class_dir: Path,
    n_samples: int,
    start_index: int,
    canvas_h: int = 40,
    canvas_w: int = 200,
) -> None:
    """Generate empty field samples (background-only, no template)."""
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        bg = random.choice(backgrounds)
        # Crop a random patch from background (no template composited)
        img = random_crop(bg, canvas_h, canvas_w)
        # Convert to greyscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply augmentation
        img = augment(img)
        cv2.imwrite(str(class_dir / f"synth_{start_index + i:05d}.png"), img)
    print(f"  {'empty':20s}  {n_samples} images  ->  {class_dir}")


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
        help="Limit generation to specific class names, or 'empty' for empty fields (default: all templates)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing synth_*.png files instead of starting from index 0",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--suffix-dir", default=SUFFIX_DIR,
        help=f"Directory containing suffix PNG templates and suffix_mapping.json (default: {SUFFIX_DIR})",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    template_dirs = [Path(d) for d in args.template_dirs]
    bg_dir = Path(args.bg_dir)
    output_dir = Path(args.output_dir)
    suffix_dir = Path(args.suffix_dir)

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

    print("Loading suffix templates...")
    suffix_templates = load_suffix_templates(suffix_dir)
    suffix_mapping = load_suffix_mapping(suffix_dir)
    if suffix_templates:
        print(f"  Found suffix images: {sorted(suffix_templates)}")
        print(f"  Suffix mapping covers {len(suffix_mapping)} word(s)")
    else:
        print("  No suffix templates found — plain words only")

    target_classes = args.classes if args.classes else sorted(templates)

    print(f"\nGenerating {args.samples} samples per class...")
    print("-" * 60)

    for name in target_classes:
        # Handle empty class specially
        if name == 'empty':
            class_dir = output_dir / 'empty'
            start_index = next_synth_index(class_dir) if args.append else 0
            generate_empty_class(backgrounds, class_dir, args.samples, start_index)
            continue
        
        if name not in templates:
            print(f"  {name:20s}  SKIP — no template found")
            continue
        class_dir = output_dir / name
        start_index = next_synth_index(class_dir) if args.append else 0
        options = suffix_mapping.get(name)  # None if word has no suffix
        generate_class(
            name, templates[name], backgrounds, class_dir,
            args.samples, start_index,
            suffix_templates=suffix_templates if options else None,
            suffix_options=options,
        )

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

"""
generate_synth_strip_data.py

Generates synthetic training strips for the CRNN digit sequence model by
compositing single-glyph templates (data/synth_data/*.png) side-by-side onto
random background crops, then saving 32-px-tall grayscale strip images into
data/strip_training_data/.

Three formats are generated in equal proportion:
  decimal   {1,3}.xx   e.g.  "1.25"  "42.50"  "100.00"
  percent   {1,2}%     e.g.  "5%"    "75%"
  integer   {1,6}      e.g.  "7"     "42"     "100000"

Output filename encodes the label:   <label>_<n:06d>.png
  e.g.  42.50_000001.png

Usage:
    python generate_synth_strip_data.py                   # 300 per format
    python generate_synth_strip_data.py --samples 500
    python generate_synth_strip_data.py --append          # add to existing
    python generate_synth_strip_data.py --seed 42
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


STRIP_H = 32
STRIP_W = 256

# Template file mapping (same source as generate_synth_data.py)
TEMPLATE_FILES = {
    "0": "0.png", "1": "1.png", "2": "2.png", "3": "3.png", "4": "4.png",
    "5": "5.png", "6": "6.png", "7": "7.png", "8": "8.png", "9": "9.png",
    ".": "dot.png",
    "%": "percent.png",
}


# ── Template / background loading ─────────────────────────────────────────────

def load_templates(synth_dir: Path) -> dict[str, np.ndarray]:
    """Load BGRA glyph templates. Returns {char: BGRA array}."""
    templates: dict[str, np.ndarray] = {}
    for char, filename in TEMPLATE_FILES.items():
        path = synth_dir / filename
        if not path.exists():
            print(f"  Warning: template not found: {path}")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Warning: could not read {path}")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        templates[char] = img
    return templates


def load_backgrounds(bg_dir: Path) -> list[np.ndarray]:
    """Load all background images as BGR arrays."""
    backgrounds: list[np.ndarray] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in sorted(bg_dir.glob(pattern)):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                backgrounds.append(img)
    return backgrounds


def random_bg_crop(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Return a random (target_h × target_w) BGR crop, tiling if necessary."""
    h, w = image.shape[:2]
    if h < target_h or w < target_w:
        reps_h = (target_h // h) + 2
        reps_w = (target_w // w) + 2
        image = np.tile(image, (reps_h, reps_w, 1))
        h, w = image.shape[:2]
    y = random.randint(0, h - target_h)
    x = random.randint(0, w - target_w)
    return image[y: y + target_h, x: x + target_w].copy()


# ── Sequence generation helpers ───────────────────────────────────────────────

def _random_decimal() -> str:
    """Return a random decimal string matching {1,3}.xx."""
    int_part = random.randint(0, 999)
    frac_part = random.randint(0, 99)
    return f"{int_part}.{frac_part:02d}"


def _random_percent() -> str:
    """Return a random percentage string matching {1,2}%."""
    return f"{random.randint(0, 99)}%"


def _random_integer() -> str:
    """Return a random integer string matching {1,6}."""
    n_digits = random.randint(1, 6)
    # First digit non-zero for multi-digit numbers
    if n_digits == 1:
        return str(random.randint(0, 9))
    first = str(random.randint(1, 9))
    rest = "".join(str(random.randint(0, 9)) for _ in range(n_digits - 1))
    return first + rest


def _random_decimal_percent() -> str:
    """Return a random decimal percentage string matching {1,2}.{2}%."""
    int_part = random.randint(0, 99)
    frac_part = random.randint(0, 99)
    return f"{int_part}.{frac_part:02d}%"


FORMAT_GENERATORS = [_random_decimal, _random_percent, _random_integer, _random_decimal_percent]


# ── Strip compositing ─────────────────────────────────────────────────────────

def composite_strip(
    sequence: str,
    templates: dict[str, np.ndarray],
    bg: np.ndarray,
    strip_h: int = STRIP_H,
    scale_range: tuple[float, float] = (0.65, 0.95),
    gap_range: tuple[int, int] = (0, 2),
) -> np.ndarray | None:
    """
    Alpha-composite glyphs for *sequence* side-by-side onto a background strip.

    Returns a grayscale (strip_h × W) image, or None if a required template
    is missing.
    
    Special handling for dot (.): scaled to ~40% of digit height and positioned
    at the bottom (baseline) instead of vertically centered.
    """
    # Shared glyph height for this strip (for digits and %)
    glyph_h = max(4, int(strip_h * random.uniform(*scale_range)))

    # Collect scaled glyphs with metadata (height, baseline_offset)
    glyphs: list[tuple[np.ndarray, int]] = []  # (image, y_offset_from_bottom)
    for ch in sequence:
        tmpl = templates.get(ch)
        if tmpl is None:
            return None  # required template missing
        src_h, src_w = tmpl.shape[:2]
        
        if ch == ".":
            # Dot: scale to ~40% of digit height, position at baseline
            dot_h = max(2, round(glyph_h * 0.4))
            dot_w = max(1, round(src_w * dot_h / src_h))
            dot_img = cv2.resize(tmpl, (dot_w, dot_h), interpolation=cv2.INTER_AREA)
            # y_offset_from_bottom: how many pixels up from the bottom to place the top of the dot
            # If glyph_h is 20 and dot_h is 8, we want the dot sitting near the bottom, e.g. offset 3px up
            y_offset_from_bottom = max(0, glyph_h - dot_h - round(glyph_h * 0.1))
            glyphs.append((dot_img, y_offset_from_bottom))
        else:
            # Regular glyph (digit or %): scale to full height, center vertically
            scaled_w = max(1, round(src_w * glyph_h / src_h))
            scaled_img = cv2.resize(tmpl, (scaled_w, glyph_h), interpolation=cv2.INTER_AREA)
            glyphs.append((scaled_img, 0))  # y_offset 0 means centre it

    # Compute total width
    gaps = [random.randint(*gap_range) for _ in range(len(glyphs) - 1)]
    total_w = sum(g.shape[1] for g, _ in glyphs) + sum(gaps)

    # Canvas: background crop
    canvas_w = max(total_w + 2, 16)  # at least a tiny pad
    canvas_bgr = random_bg_crop(bg, canvas_w, strip_h).astype(np.float32)

    # Vertical offset: centre digit glyphs, special handling for dot
    y_off = (strip_h - glyph_h) // 2

    x = 0
    for idx, (glyph, y_offset_from_bottom) in enumerate(glyphs):
        gw = glyph.shape[1]
        gh = glyph.shape[0]
        b, g_ch, r, a = cv2.split(glyph)
        digit_bgr = cv2.merge([b, g_ch, r]).astype(np.float32)
        alpha = a.astype(np.float32) / 255.0
        alpha_3 = alpha[:, :, np.newaxis]

        # Calculate y position: for dots with y_offset_from_bottom, position at bottom
        # For regular glyphs (y_offset_from_bottom=0), use centred y_off
        if y_offset_from_bottom > 0:
            # Dot case: position so its top is at (strip_h - glyph_h + y_offset_from_bottom)
            glyph_y_top = strip_h - glyph_h + y_offset_from_bottom
        else:
            # Regular glyph: centred
            glyph_y_top = y_off

        y1 = max(0, glyph_y_top)
        y2 = min(strip_h, glyph_y_top + gh)
        x1 = x
        x2 = min(canvas_w, x + gw)
        
        # Clipped region in glyph space
        glyph_y1 = max(0, -glyph_y_top) if glyph_y_top < 0 else 0
        glyph_y2 = glyph_y1 + (y2 - y1)
        glyph_x1 = 0
        glyph_x2 = (x2 - x1)
        
        canvas_bgr[y1:y2, x1:x2] = (
            alpha_3[glyph_y1:glyph_y2, glyph_x1:glyph_x2] * digit_bgr[glyph_y1:glyph_y2, glyph_x1:glyph_x2]
            + (1.0 - alpha_3[glyph_y1:glyph_y2, glyph_x1:glyph_x2]) * canvas_bgr[y1:y2, x1:x2]
        )
        x += gw
        if idx < len(gaps):
            x += gaps[idx]

    result = np.clip(canvas_bgr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(img: np.ndarray) -> np.ndarray:
    """Apply random photometric augmentations to a grayscale strip."""
    contrast = random.uniform(0.75, 1.30)
    brightness = random.randint(-25, 25)
    img = np.clip(contrast * img.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        sigma = random.uniform(2.0, 10.0)
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


# ── Index helpers ─────────────────────────────────────────────────────────────

def next_strip_index(output_dir: Path) -> int:
    """Return the next available strip index (max existing + 1, or 0)."""
    if not output_dir.exists():
        return 0
    indices = []
    for p in output_dir.glob("*_*.png"):
        # filename: <label>_<n>.png — split only on the last underscore
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                indices.append(int(parts[1]))
            except ValueError:
                pass
    return (max(indices) + 1) if indices else 0


# ── Main generation ───────────────────────────────────────────────────────────

def generate_strips(
    synth_dir: Path,
    output_dir: Path,
    n_per_format: int,
    start_index: int = 0,
) -> None:
    templates = load_templates(synth_dir)
    bg_dir = synth_dir / "backgrounds"
    backgrounds = load_backgrounds(bg_dir)

    missing = [c for c in "0123456789.%" if c not in templates]
    if missing:
        raise RuntimeError(f"Missing glyph templates for: {missing}")
    if not backgrounds:
        raise RuntimeError(f"No background images found in {bg_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    idx = start_index
    for fmt_gen in FORMAT_GENERATORS:
        generated = 0
        attempts = 0
        max_attempts = n_per_format * 10
        while generated < n_per_format and attempts < max_attempts:
            attempts += 1
            seq = fmt_gen()
            bg = random.choice(backgrounds)
            strip = composite_strip(seq, templates, bg)
            if strip is None:
                continue
            strip = augment(strip)
            # Escape '.' in filenames: use 'p' separator to avoid confusion
            # Keep label exactly as the text string — dots are valid in filenames
            fname = f"{seq}_{idx:06d}.png"
            cv2.imwrite(str(output_dir / fname), strip)
            idx += 1
            generated += 1

        fmt_name = fmt_gen.__name__.replace("_random_", "")
        print(f"  [{fmt_name:>8}]  {generated} strips  ->  {output_dir}")

    print(f"  Total: {idx - start_index} strips written.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic strip training data for the CRNN digit model."
    )
    parser.add_argument(
        "--synth-dir", default="data/synth_data",
        help="Folder with glyph templates and backgrounds/ (default: data/synth_data)",
    )
    parser.add_argument(
        "--output-dir", default="data/strip_training_data",
        help="Output folder for strip images (default: data/strip_training_data)",
    )
    parser.add_argument(
        "--samples", type=int, default=300,
        help="Number of strips per format (default: 300, total ~900)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Add to existing strips instead of overwriting",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    synth_dir = Path(args.synth_dir)
    output_dir = Path(args.output_dir)

    start_index = next_strip_index(output_dir) if args.append else 0

    if not args.append and output_dir.exists():
        # Remove existing synth strips only
        removed = 0
        for p in output_dir.glob("*_*.png"):
            p.unlink()
            removed += 1
        if removed:
            print(f"  Removed {removed} existing strips from {output_dir}")

    print(f"Generating {args.samples} strips per format (start index: {start_index}) ...")
    generate_strips(synth_dir, output_dir, args.samples, start_index)


if __name__ == "__main__":
    main()

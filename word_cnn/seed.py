"""
Seed word training data from existing template images.

Copies each template from data/templates/{subdir}/ into
data/word_training_data/{label}/template.{ext} so the word CNN
can train even with just one image per class.
"""

import shutil
from pathlib import Path

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def seed_from_templates(
    template_dirs: list[str | Path],
    output_dir: str | Path,
) -> dict[str, int]:
    """
    Copy template images into word_training_data class folders.

    Parameters
    ----------
    template_dirs : list of directories containing labelled templates
        (e.g. ["data/templates/resources", "data/templates/deposits"]).
    output_dir : root of the word training data folder
        (e.g. "data/word_training_data").

    Returns
    -------
    dict mapping class name → number of images copied for that class.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for tdir in template_dirs:
        tdir = Path(tdir)
        if not tdir.is_dir():
            continue
        for img_path in sorted(tdir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            label = img_path.stem  # e.g. "titanium"
            class_dir = out / label
            class_dir.mkdir(parents=True, exist_ok=True)
            dest = class_dir / f"template{img_path.suffix}"
            if not dest.exists():
                shutil.copy2(img_path, dest)
            counts[label] = counts.get(label, 0) + 1
    return counts

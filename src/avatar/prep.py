"""Normalise the cropped image and get a background mask (cached in work/<stem>/mask.png)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import WORK_DIR, WORK_SIZE, die, open_image, rel, warn

MODEL = "u2net_human_seg"  # rembg model trained to segment people


def normalise(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.LANCZOS))
    # Contrast stretch on lightness only, so colours don't shift.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(l, (1, 99))
    lab[..., 0] = np.clip((l - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Edge-preserving smoothing: removes skin and fabric texture, keeps outlines.
    return cv2.bilateralFilter(rgb, d=9, sigmaColor=35, sigmaSpace=7)


def _rembg(img: Image.Image) -> tuple[Image.Image, str] | None:
    try:
        from rembg import new_session, remove
    except ImportError:
        warn("rembg is not installed (uv sync --extra bg): no mask, so only the keep mode is available")
        return None
    print(f"mask: segmenting with rembg/{MODEL} (the first run downloads the model)...")
    img = img.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    mask = remove(img, session=new_session(MODEL), only_mask=True, post_process_mask=True)
    return mask.convert("L"), f"rembg/{MODEL}"


def _from_file(path: Path) -> Image.Image:
    """An image with transparency (e.g. Preview's 'Remove Background' export) used as the mask."""
    if not path.is_file():
        die(f"{rel(path)} not found")
    img = open_image(path)
    if img.width != img.height:
        die(f"{rel(path)} is {img.width}×{img.height}; it must be the same square crop as the input image")
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        return img.convert("RGBA").getchannel("A")
    warn(f"{rel(path)} has no transparency; treating it as a greyscale mask (white = subject)")
    return img.convert("L")


def _clean(mask: Image.Image) -> Image.Image:
    m = np.asarray(mask.resize((WORK_SIZE, WORK_SIZE), Image.LANCZOS))
    # Keep only the largest connected region (drops stray bits), soften the edge.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(labels == largest, m, 0).astype(np.uint8)
    return Image.fromarray(cv2.GaussianBlur(m, (0, 0), 1.0))


def mask_path(stem: str) -> Path:
    return WORK_DIR / stem / "mask.png"


def get_mask(src: Path, img: Image.Image, stem: str, *, mask_from: Path | None, redo: bool) -> np.ndarray | None:
    """Return the mask (uint8, 255 = subject) or None. Reused from work/<stem>/mask.png when still valid."""
    path = mask_path(stem)
    if mask_from:
        mask, how = _from_file(mask_from), rel(mask_from)
    elif path.exists() and not redo and path.stat().st_mtime >= src.stat().st_mtime:
        print(f"mask: reusing {rel(path)} (--redo-mask to regenerate)")
        return np.asarray(Image.open(path).convert("L"))
    else:
        result = _rembg(img)
        if result is None:
            return None
        mask, how = result
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _clean(mask)
    cleaned.save(path)
    print(f"mask: {how} -> {rel(path)}")
    return np.asarray(cleaned)

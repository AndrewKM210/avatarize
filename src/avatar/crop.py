"""Crop assets/original/<image> to a square assets/cropped/<image>."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .common import CROPPED_DIR, ORIGINAL_DIR, die, missing_image_msg, open_image, rel, resolve_image, warn

DETECT_SIZE = 800  # face detection runs on a downscaled copy


def detect_faces(img: Image.Image) -> list[tuple[float, float, float]]:
    """Faces as (centre_x, centre_y, height) in original pixels, sorted left to right.

    scikit-image's LBP frontal-face cascade: a classic detector, not a generative model.
    """
    from skimage import data
    from skimage.feature import Cascade

    scale = DETECT_SIZE / max(img.size)
    small = img.convert("RGB").resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    detector = Cascade(data.lbp_frontal_face_cascade_filename())
    side = min(small.size)
    found = detector.detect_multi_scale(
        img=np.asarray(small), scale_factor=1.15, step_ratio=1, min_size=(32, 32), max_size=(side, side)
    )
    faces = [
        ((f["c"] + f["width"] / 2) / scale, (f["r"] + f["height"] / 2) / scale, f["height"] / scale) for f in found
    ]
    # The cascade often reports the same face several times at slightly different scales: merge those.
    merged: list[tuple[float, float, float]] = []
    for f in sorted(faces, key=lambda f: -f[2]):
        if all(abs(f[0] - m[0]) > m[2] / 2 or abs(f[1] - m[1]) > m[2] / 2 for m in merged):
            merged.append(f)
    return sorted(merged, key=lambda f: f[0])


def auto_box(img: Image.Image, zoom: float, face_index: int | None, pad: bool) -> tuple[int, int, int]:
    w, h = img.size
    max_side = min(w, h)
    faces = detect_faces(img)
    if not faces:
        warn("no face detected, falling back to a centred square crop (use --box to set one)")
        return (w - max_side) // 2, (h - max_side) // 2, max_side
    if face_index is not None:
        if not 1 <= face_index <= len(faces):
            die(f"--face {face_index}: found {len(faces)} face(s)")
        cx, cy, fh = faces[face_index - 1]
    else:
        cx, cy, fh = max(faces, key=lambda f: f[2])
    if len(faces) > 1:
        listing = ", ".join(f"{i}: x={f[0]:.0f}" for i, f in enumerate(faces, 1))
        chosen = face_index or 1 + faces.index(max(faces, key=lambda f: f[2]))
        print(f"found {len(faces)} faces, left to right ({listing}); using {chosen}. Pick another with --face N")
    side = round(fh * zoom) if pad else min(round(fh * zoom), max_side)
    centre_y = cy + 0.12 * side  # face a little above centre so the crown and shoulders fit
    x, y = round(cx - side / 2), round(centre_y - side / 2)
    if pad:
        return x, y, side
    cx_, cy_ = max(0, min(x, w - side)), max(0, min(y, h - side))
    if max(abs(cx_ - x), abs(cy_ - y)) > 0.05 * side:
        print("note: the face is close to the photo edge, so the crop was shifted and the face is off-centre.\n"
              "      --pad keeps it centred and fills the part past the photo edge with black")
    return cx_, cy_, side


def _padded_crop(img: Image.Image, x: int, y: int, side: int) -> Image.Image:
    """Crop that may extend past the photo; the missing part is black."""
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[:2]
    left, top = max(0, -x), max(0, -y)
    right, bottom = max(0, x + side - w), max(0, y + side - h)
    a = np.pad(a, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)
    return Image.fromarray(a[y + top : y + top + side, x + left : x + left + side])


def crop(image: str, box: tuple[int, int, int] | None, zoom: float, face: int | None, pad: bool,
         out: str | None, force: bool) -> None:
    src = resolve_image(image, ORIGINAL_DIR)
    if not src.is_file():
        die(missing_image_msg(src, ORIGINAL_DIR))
    dst = CROPPED_DIR / (out or src.name)
    if dst.exists() and not force:
        die(f"{rel(dst)} already exists. Pass --force to overwrite it, or --out to pick another name.")

    img = open_image(src)
    w, h = img.size
    if box:
        x, y, side = box
        if not pad and (x < 0 or y < 0 or x + side > w or y + side > h):
            die(f"box {x},{y},{side} falls outside the {w}×{h} image (add --pad to fill the rest with black)")
    else:
        x, y, side = auto_box(img, zoom, face, pad)

    outside = x < 0 or y < 0 or x + side > w or y + side > h
    cropped = _padded_crop(img, x, y, side) if outside else img.crop((x, y, x + side, y + side))
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    jpeg = dst.suffix.lower() in {".jpg", ".jpeg"}
    if jpeg and cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
    cropped.save(dst, **({"quality": 95} if jpeg else {}))  # no exif= argument: metadata is not copied
    print(f"cropped {rel(src)} ({w}×{h}) at box {x},{y},{side} -> {rel(dst)} ({side}×{side})"
          + (", padded with black" if outside else ""))
    if side < 1024:
        warn(f"the crop is only {side}px; it will be upscaled to 1024px, so expect softer results")

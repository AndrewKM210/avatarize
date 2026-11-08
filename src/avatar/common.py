"""Shared paths, image loading, palette and external-tool helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = ROOT / "assets" / "original"
CROPPED_DIR = ROOT / "assets" / "cropped"
WORK_DIR = ROOT / "work"
OUT_DIR = ROOT / "out"
PALETTE_FILE = ROOT / "palette.toml"

DEFAULT_IMAGE = "profile.jpeg"
WORK_SIZE = 1024
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
MODES = ("keep", "simplify", "remove")


# ---------------------------------------------------------------- messages

def die(msg: str):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- images

def resolve_image(arg: str, folder: Path) -> Path:
    """A bare filename is looked up in `folder`; anything with a directory part is used as a path."""
    p = Path(arg).expanduser()
    if p.parent != Path(".") or p.is_absolute():
        return p.resolve()
    if (folder / p).is_file():
        return folder / p
    # Extension optional: `profile_balloon` matches profile_balloon.jpg.
    stem = p.stem if p.suffix.lower() in IMAGE_EXTS else p.name
    matches = [folder / n for n in list_images(folder) if Path(n).stem == stem]
    if len(matches) > 1:
        die(f"'{arg}' matches several files in {rel(folder)}/: {', '.join(m.name for m in matches)}. "
            "Give the full filename.")
    return matches[0] if matches else folder / p


def list_images(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def missing_image_msg(path: Path, folder: Path) -> str:
    found = list_images(folder)
    return f"{rel(path)} not found. Images in {rel(folder)}/: {', '.join(found) if found else '(none)'}"


def open_image(path: Path) -> Image.Image:
    """Open an image with its EXIF orientation applied (phone photos are often stored rotated)."""
    return ImageOps.exif_transpose(Image.open(path))


def load_cropped(arg: str) -> tuple[Path, Image.Image]:
    """Load the square image every stage after `crop` starts from. Exits with a hint otherwise."""
    path = resolve_image(arg, CROPPED_DIR)
    if not path.is_file():
        hint = ""
        originals = [n for n in list_images(ORIGINAL_DIR) if Path(n).stem == path.stem]
        if originals:
            hint = f"\nAn original exists. Crop it first:  uv run avatar crop --image {originals[0]}"
        die(missing_image_msg(path, CROPPED_DIR) + hint)
    img = open_image(path)
    if img.width != img.height:
        die(
            f"{rel(path)} is {img.width}×{img.height}, not square.\n"
            f"Crop it first:  uv run avatar crop --image {path.name}\n"
            f"or replace it with a square version in {rel(CROPPED_DIR)}/"
        )
    return path, img


def stem_of(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------- palette

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        die(f"invalid colour '{h}', expected #RRGGBB")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c)) for c in rgb))


def load_palette() -> dict:
    if not PALETTE_FILE.is_file():
        die(f"{rel(PALETTE_FILE)} not found")
    with PALETTE_FILE.open("rb") as f:
        data = tomllib.load(f)
    themes = data.get("themes", {})
    if not themes:
        die(f"{rel(PALETTE_FILE)} defines no [themes.*] tables")
    for name, t in themes.items():
        for key in ("accent", "paper"):
            if key not in t:
                die(f"{rel(PALETTE_FILE)}: [themes.{name}] is missing '{key}'")
        t["accent_rgb"] = hex_to_rgb(t["accent"])
        t["paper_rgb"] = hex_to_rgb(t["paper"])
    data.setdefault("preview", {})
    data["preview"].setdefault("light_ui", "#FFFFFF")
    data["preview"].setdefault("dark_ui", "#0D1117")
    return data


# ---------------------------------------------------------------- external tools

def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    # `go install` puts binaries in $(go env GOPATH)/bin, which non-login shells may not have on PATH.
    candidates = []
    gopath = os.environ.get("GOPATH")
    if not gopath and shutil.which("go"):
        try:
            gopath = subprocess.run(["go", "env", "GOPATH"], capture_output=True, text=True).stdout.strip()
        except OSError:
            gopath = None
    if gopath:
        candidates.append(Path(gopath) / "bin" / name)
    candidates.append(Path.home() / "go" / "bin" / name)
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def require_tool(name: str, install_hint: str) -> str:
    path = find_tool(name)
    if not path:
        die(f"'{name}' not found. Install it: {install_hint}")
    return path  # type: ignore[return-value]


def svg_to_png(svg: Path, png: Path, size: int) -> None:
    resvg = require_tool("resvg", "brew install resvg")
    subprocess.run([resvg, "-w", str(size), "-h", str(size), str(svg), str(png)], check=True)

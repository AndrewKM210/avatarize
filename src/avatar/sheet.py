"""Contact sheet: each variant as a large square plus 48/32 px circles on light and dark UI backgrounds."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import hex_to_rgb

SMALL_SIZES = (48, 32)
PAD, LABEL_H, STRIP_H, TITLE_H = 16, 22, 64, 30
SHEET_BG, TEXT = (236, 236, 236), (40, 40, 40)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def circle(img: Image.Image, size: int) -> Image.Image:
    """Circle crop at `size` px, antialiased by masking at 4x and downsampling."""
    big = size * 4
    rgba = img.resize((big, big), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (big, big), 0)
    ImageDraw.Draw(m).ellipse((0, 0, big - 1, big - 1), fill=255)
    rgba.putalpha(m)
    return rgba.resize((size, size), Image.LANCZOS)


def _strip(img: Image.Image, width: int, bg) -> Image.Image:
    s = Image.new("RGB", (width, STRIP_H), bg)
    x = 12
    for size in SMALL_SIZES:
        c = circle(img, size)
        s.paste(c, (x, (STRIP_H - size) // 2), c)
        x += size + 16
    return s


def _cell(path: Path, label: str, size: int, light_ui, dark_ui, font) -> Image.Image:
    img = Image.open(path).convert("RGB")
    c = Image.new("RGB", (size, LABEL_H + size + 2 * STRIP_H), SHEET_BG)
    ImageDraw.Draw(c).text((0, 3), label, fill=TEXT, font=font)
    big = img.resize((size, size), Image.LANCZOS)
    ImageDraw.Draw(big).ellipse((0, 0, size - 1, size - 1), outline=(255, 255, 255), width=1)  # circle guide
    c.paste(big, (0, LABEL_H))
    c.paste(_strip(img, size, light_ui), (0, LABEL_H + size))
    c.paste(_strip(img, size, dark_ui), (0, LABEL_H + size + STRIP_H))
    return c


def build(groups: list[tuple[str, list[tuple[str, Path]]]], dst: Path, palette: dict, title: str,
          note: str = "", size: int = 320, cols: int = 6) -> None:
    """groups: [(row title, [(cell label, png path), ...]), ...]; long rows wrap after `cols` cells.
    `note` is printed under the title (e.g. the command that reproduces a cell)."""
    light_ui = hex_to_rgb(palette["preview"]["light_ui"])
    dark_ui = hex_to_rgb(palette["preview"]["dark_ui"])
    font, title_font, head_font = _font(13), _font(18), _font(22)
    cols = min(cols, max(len(items) for _, items in groups))
    cell_h = LABEL_H + size + 2 * STRIP_H

    lines = []
    for name, items in groups:
        for start in range(0, len(items), cols):
            lines.append((name if start == 0 else None, items[start : start + cols]))
    width = PAD + cols * (size + PAD)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(width, 2 * PAD + int(max(measure.textlength(title, font=head_font),
                                         measure.textlength(note, font=title_font))))
    head_h = 36 + (TITLE_H if note else 0)
    height = PAD + head_h + sum((TITLE_H if t else 0) + cell_h + PAD for t, _ in lines)
    sheet = Image.new("RGB", (width, height), SHEET_BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, PAD), title, fill=TEXT, font=head_font)
    if note:
        draw.text((PAD, PAD + 36), note, fill=(90, 90, 90), font=title_font)

    y = PAD + head_h
    for name, items in lines:
        if name:
            draw.text((PAD, y + 5), name, fill=TEXT, font=title_font)
            y += TITLE_H
        for i, (label, path) in enumerate(items):
            sheet.paste(_cell(path, label, size, light_ui, dark_ui, font), (PAD + i * (size + PAD), y))
        y += cell_h + PAD
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)

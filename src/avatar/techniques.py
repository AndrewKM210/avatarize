"""Stylisation techniques and background modes. All deterministic image processing, no generative models.

Each technique declares its parameters: a default (used by `overview` and `make`) and optionally
a sweep (the values `explore` tries when you don't pass --set).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

from .common import WORK_SIZE, hex_to_rgb, require_tool, rgb_to_hex, svg_to_png

S = WORK_SIZE


# ============================================================ registry

@dataclass
class Param:
    name: str
    default: object
    help: str
    sweep: tuple = ()
    choices: tuple = ()

    def parse(self, raw: str):
        if self.choices and raw not in self.choices:
            raise ValueError(f"{self.name} must be one of {', '.join(map(str, self.choices))}")
        return type(self.default)(raw)


@dataclass
class Technique:
    name: str
    two_tone: bool  # uses the theme's accent + paper colours in every mode
    summary: str
    params: list[Param]
    fn: Callable = field(repr=False, default=None)
    modes: tuple = ("keep", "simplify", "remove")

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}


@dataclass
class Ctx:
    img: np.ndarray  # RGB uint8, S×S, background mode already applied
    mask: np.ndarray | None
    theme: dict
    mode: str
    dst: Path  # output path without suffix
    tmp: Path
    bg_hex: str | None  # flat background colour in remove mode (full-colour techniques)


# ============================================================ helpers

def luminance(rgb) -> float:
    r, g, b = (c / 255 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def accent_is_darker(theme: dict) -> bool:
    return luminance(theme["accent_rgb"]) < luminance(theme["paper_rgb"])


def gray01(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def svg_doc(body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" viewBox="0 0 {S} {S}">\n'
            f"{body}\n</svg>\n")


def write_svg(c: Ctx, body: str) -> Path:
    svg, png = c.dst.with_suffix(".svg"), c.dst.with_suffix(".png")
    svg.write_text(svg_doc(body))
    svg_to_png(svg, png, S)
    return png


def potrace_layer(binary: np.ndarray, color: str, tmp: Path, turdsize: int) -> str:
    """Trace a boolean mask (True = fill); return its <g> element in 0..S user units."""
    potrace = require_tool("potrace", "brew install potrace")
    pbm, svg = tmp / "layer.pbm", tmp / "layer.svg"
    Image.fromarray(np.where(binary, 0, 255).astype(np.uint8)).convert("1").save(pbm)  # potrace traces black
    subprocess.run([potrace, str(pbm), "-s", "-o", str(svg), "--turdsize", str(turdsize), "--alphamax", "1.0",
                    "--opttolerance", "0.3", "--flat", "--color", color], check=True)
    m = re.search(r"<g\b.*?</g>", svg.read_text(), re.S)
    return m.group(0) if m else ""


def remove_specks(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def kmeans_lab(lab: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(k, len(lab)))
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(
        lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 4, cv2.KMEANS_PP_CENTERS
    )
    return labels.ravel(), centers


# ============================================================ background modes

def simplify_background(rgb: np.ndarray, mask: np.ndarray, bands: int = 3) -> np.ndarray:
    """Flatten the background to a few smooth colour bands; keep the subject untouched."""
    m = (mask.astype(np.float32) / 255.0)[..., None]
    bg_w = 1.0 - m
    # Fill the subject area with the surrounding background colour (normalised convolution).
    num = cv2.GaussianBlur(rgb.astype(np.float32) * bg_w, (0, 0), 28)
    den = cv2.GaussianBlur(bg_w, (0, 0), 28)[..., None]
    filled = np.clip(np.where(den > 1e-3, num / np.maximum(den, 1e-3), rgb), 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(filled, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    bg_px = mask.reshape(-1) < 128
    _, centers = kmeans_lab(lab[bg_px] if bg_px.any() else lab, bands)
    labels = ((lab[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    order = np.argsort(centers[:, 0])
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    labels = cv2.medianBlur(rank[labels].reshape(S, S).astype(np.uint8), 41)
    band_rgb = cv2.cvtColor(centers[order][None].astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
    return np.clip(rgb * m + band_rgb[labels].astype(np.float32) * (1 - m), 0, 255).astype(np.uint8)


def flat_background(rgb: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    m = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(rgb * m + np.array(color, np.float32) * (1 - m), 0, 255).astype(np.uint8)


def mode_input(mode: str, subject: np.ndarray, mask: np.ndarray | None, theme: dict, two_tone: bool,
               cache: dict) -> np.ndarray:
    if mode == "keep":
        return subject
    if mode == "simplify":
        if "simplify" not in cache:
            cache["simplify"] = simplify_background(subject, mask)
        return cache["simplify"]
    if two_tone:  # background must come out as plain paper
        return flat_background(subject, mask, (255, 255, 255) if accent_is_darker(theme) else (0, 0, 0))
    return flat_background(subject, mask, theme["accent_rgb"])


# ============================================================ techniques

PRIMITIVE_SHAPES = {"combo": 0, "triangle": 1, "rect": 2, "circle": 4, "bezier": 6, "ellipse": 7, "polygon": 8}


def t_primitive(c: Ctx, p: dict) -> Path:
    exe = require_tool("primitive", "go install github.com/fogleman/primitive@latest")
    src = c.tmp / "primitive_in.png"
    Image.fromarray(c.img).save(src)
    png = c.dst.with_suffix(".png")
    cmd = [exe, "-i", str(src), "-o", str(c.dst.with_suffix(".svg")), "-o", str(png), "-n", str(p["shapes"]),
           "-m", str(PRIMITIVE_SHAPES[p["shape"]]), "-a", str(p["alpha"]), "-r", "256", "-s", str(S)]
    if c.bg_hex:
        cmd += ["-bg", c.bg_hex.lstrip("#")]
    subprocess.run(cmd, check=True, capture_output=True)
    return png


def t_lowpoly(c: Ctx, p: dict) -> Path:
    rng = np.random.default_rng(p["seed"])
    n = p["points"]
    gray = cv2.GaussianBlur(cv2.cvtColor(c.img, cv2.COLOR_RGB2GRAY), (0, 0), 1.5)
    ey, ex = np.nonzero(cv2.Canny(gray, 40, 120))
    n_edge = min(int(n * p["edges"]), len(ex))
    idx = rng.choice(len(ex), n_edge, replace=False) if n_edge else np.array([], int)
    pts = [np.column_stack([ex[idx], ey[idx]]).astype(np.float64), rng.uniform(0, S - 1, (n - n_edge, 2))]
    border = np.linspace(0, S - 1, 9)
    pts.append(np.array([(x, y) for x in border for y in (0, S - 1)] + [(x, y) for y in border for x in (0, S - 1)],
                        float))
    if c.mask is not None:  # points along the silhouette keep the outline crisp
        contours, _ = cv2.findContours((c.mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
            pts.append(cnt[:: max(1, len(cnt) // max(60, n // 8))].astype(np.float64))
    points = np.unique(np.vstack(pts).round(1), axis=0)
    tri = Delaunay(points)

    yy, xx = np.mgrid[0:S, 0:S]
    simplex = tri.find_simplex(np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5]))
    ok = simplex >= 0
    nt = len(tri.simplices)
    flat = c.img.reshape(-1, 3).astype(np.float64)
    counts = np.bincount(simplex[ok], minlength=nt).astype(np.float64)
    colors = np.stack([np.bincount(simplex[ok], weights=flat[ok, ch], minlength=nt) for ch in range(3)], 1)
    for i in np.nonzero(counts == 0)[0]:  # slivers with no pixel centre: sample the centroid
        cx, cy = np.clip(points[tri.simplices[i]].mean(0).astype(int), 0, S - 1)
        colors[i], counts[i] = c.img[cy, cx], 1
    colors /= counts[:, None]

    polys = []
    for s, col in zip(tri.simplices, colors):
        hx = rgb_to_hex(col)
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points[s])
        polys.append(f'<polygon points="{coords}" fill="{hx}" stroke="{hx}" stroke-width="0.8" stroke-linejoin="round"/>')
    return write_svg(c, "\n".join(polys))


def t_posterize(c: Ctx, p: dict) -> Path:
    smooth = cv2.medianBlur(cv2.bilateralFilter(c.img, 9, 60, 9), p["smooth"] | 1)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    if c.mask is None:
        labels, centers = kmeans_lab(lab, p["colors"])
    else:  # subject and background quantised separately, so the face keeps all its tones
        fg = c.mask.reshape(-1) > 127
        lf, cf = kmeans_lab(lab[fg], p["colors"])
        kb = 1 if c.mode == "remove" else p["bg_colors"]
        lb, cb = kmeans_lab(lab[~fg], kb)
        if c.mode == "remove" and c.bg_hex:  # exact palette colour, not a blurred average
            cb[:] = cv2.cvtColor(np.uint8([[hex_to_rgb(c.bg_hex)]]), cv2.COLOR_RGB2LAB)[0, 0]
        labels = np.empty(len(lab), np.int64)
        labels[fg], labels[~fg] = lf, lb + len(cf)
        centers = np.vstack([cf, cb])

    order = np.argsort(centers[:, 0])  # dark -> light
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    labels = cv2.medianBlur(rank[labels.reshape(S, S)].astype(np.uint8), 5)
    colors = cv2.cvtColor(centers[order][None].astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
    # Stacked layers: darkest colour as the base, then layer i covers every label >= i. Each layer
    # is a superset of the ones above it, so traced layers never leave gaps between them.
    body = [f'<rect width="{S}" height="{S}" fill="{rgb_to_hex(colors[0])}"/>']
    body += [potrace_layer(labels >= i, rgb_to_hex(colors[i]), c.tmp, p["specks"]) for i in range(1, len(colors))]
    return write_svg(c, "\n".join(body))


def t_lineart(c: Ctx, p: dict) -> Path:
    g = gray01(cv2.bilateralFilter(c.img, 9, 50, 9))
    sigma = p["sigma"]
    dog = cv2.GaussianBlur(g, (0, 0), sigma) - 0.98 * cv2.GaussianBlur(g, (0, 0), sigma * 1.6)  # XDoG, tau form
    ink = dog < -p["threshold"]
    if p["thicken"]:
        k = 2 * p["thicken"] + 1
        ink = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    if accent_is_darker(c.theme) and p["shadow"] > 0:  # solid ink for the deepest shadows (e.g. a dark shirt)
        ink |= cv2.GaussianBlur(g, (0, 0), 2.0) < p["shadow"]
    ink = remove_specks(ink, p["specks"])
    if c.mask is not None and c.mode != "keep":  # no lines on the flattened/removed background
        ink &= cv2.dilate((c.mask > 127).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    body = [f'<rect width="{S}" height="{S}" fill="{c.theme["paper"]}"/>',
            potrace_layer(ink, c.theme["accent"], c.tmp, 2)]
    return write_svg(c, "\n".join(body))


TECHNIQUES: dict[str, Technique] = {t.name: t for t in [
    Technique("primitive", False, "photo rebuilt from overlapping shapes by hill climbing (fogleman/primitive)", [
        Param("shapes", 200, "number of shapes", sweep=(100, 200, 400)),
        Param("shape", "triangle", "shape type", sweep=("triangle", "ellipse", "combo"),
              choices=tuple(PRIMITIVE_SHAPES)),
        Param("alpha", 128, "shape opacity 1-255"),
    ], t_primitive),
    Technique("lowpoly", False, "Delaunay triangles on edge points, each filled with its mean colour", [
        Param("points", 800, "number of triangulation points", sweep=(300, 600, 1000, 1500)),
        Param("edges", 0.65, "fraction of points placed on edges (more = sharper features)"),
        Param("seed", 0, "random seed for point placement"),
    ], t_lowpoly),
    Technique("posterize", False, "k-means colour reduction traced to smooth vector shapes", [
        Param("colors", 4, "colours for the subject", sweep=(3, 4, 5, 6)),
        Param("bg_colors", 3, "colours for the background (keep/simplify modes)"),
        Param("smooth", 7, "median smoothing before quantising (odd, larger = simpler shapes)"),
        Param("specks", 25, "drop shapes smaller than this many pixels"),
    ], t_posterize),
    Technique("lineart", True, "ink drawing from an XDoG edge filter, traced to vector", [
        Param("sigma", 1.8, "edge scale (larger = fewer, broader lines)", sweep=(1.2, 1.8, 2.6)),
        Param("threshold", 0.002, "edge sensitivity (smaller = more lines)"),
        Param("thicken", 1, "extra line thickness in px", sweep=(0, 1, 2)),
        Param("shadow", 0.12, "light theme: fill areas darker than this solid (0 = off)"),
        Param("specks", 12, "drop marks smaller than this many pixels"),
    ], t_lineart, modes=("keep", "remove")),  # simplify would equal remove: lines are drawn on the subject only
]}

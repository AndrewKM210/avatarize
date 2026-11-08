"""avatar: turn a photo into a stylised avatar.

  uv run avatar crop     [--image NAME]                     optional: square-crop assets/original/NAME
  uv run avatar overview [--image NAME]                     sheet: every technique, background and theme
  uv run avatar explore  TECHNIQUE [--set P=a,b ...]        sheet: one technique with different settings
  uv run avatar make     TECHNIQUE --background B --theme T [--set P=v ...]   the final PNG + SVG
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sheet
from .common import DEFAULT_IMAGE, MODES, OUT_DIR, WORK_DIR, die, load_cropped, load_palette, rel, warn
from .crop import crop
from .prep import get_mask, mask_path, normalise
from .techniques import TECHNIQUES, Ctx, Technique, mode_input

# ============================================================ jobs and the render cache
#
# Every render is stored in work/<image>/renders/ under a name that encodes all its settings and colours.
# Sheets reuse renders that are still valid, and `make` copies the exact render the sheet showed. That
# matters for Primitive, which is random: re-rendering it would give a different picture.


@dataclass
class Inputs:
    src: Path
    stem: str
    subject: np.ndarray
    mask: np.ndarray | None
    palette: dict
    valid_after: float  # renders older than this (photo or mask changed) are stale


@dataclass
class Job:
    tech: Technique
    mode: str
    theme: str | None  # None: full-colour technique in a mode without a theme colour
    params: dict


def slug(params: dict) -> str:
    # No dots in file names: Path.with_suffix() would treat "sigma-1.8" as a suffix.
    return "_".join(f"{k}-{v}".replace(".", "p") for k, v in params.items())


def render_path(inp: Inputs, j: Job) -> Path:
    colours = ""
    if j.theme:
        t = inp.palette["themes"][j.theme]
        colours = f"{j.theme}-{t['accent']}-{t['paper']}".replace("#", "")
    name = "_".join(x for x in (j.tech.name, j.mode, colours, slug(j.params)) if x)
    return WORK_DIR / inp.stem / "renders" / name


def run_jobs(inp: Inputs, jobs: list[Job], fresh: bool) -> list[Path]:
    """Render (or reuse) every job; return the path of each render without suffix."""
    themes = inp.palette["themes"]
    first_theme = next(iter(themes))
    cache: dict = {}
    out, t_start = [], time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, j in enumerate(jobs, 1):
            dst = render_path(inp, j)
            png = dst.with_suffix(".png")
            what = " · ".join(x for x in (j.tech.name, j.mode, j.theme, slug(j.params)) if x)
            out.append(dst)
            if not fresh and png.is_file() and png.stat().st_mtime >= inp.valid_after:
                print(f"  [{i:>2}/{len(jobs)}] {what}  (reused)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            for old in (png, dst.with_suffix(".svg")):
                old.unlink(missing_ok=True)
            theme = themes[j.theme or first_theme]
            img = mode_input(j.mode, inp.subject, inp.mask, theme, j.tech.two_tone, cache)
            bg_hex = theme["accent"] if (j.mode == "remove" and not j.tech.two_tone) else None
            t0 = time.time()
            j.tech.fn(Ctx(img, inp.mask, theme, j.mode, dst, Path(tmpdir), bg_hex), j.params)
            print(f"  [{i:>2}/{len(jobs)}] {what}  ({time.time() - t0:.1f}s)")
    print(f"rendered in {time.time() - t_start:.0f}s")
    return out


# ============================================================ argument handling

def parse_list(value: str, allowed, what: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        die(f"unknown {what} {', '.join(bad)} (choose from {', '.join(allowed)})")
    return items


def get_technique(name: str | None) -> Technique:
    if name not in TECHNIQUES:
        die(f"unknown technique '{name}' (choose from {', '.join(TECHNIQUES)})")
    return TECHNIQUES[name]


def parse_sets(tech: Technique, items: list[str] | None) -> dict[str, list]:
    """--set name=v1,v2 ... -> {name: [values]} for the parameters given."""
    values = {}
    for item in items or []:
        name, eq, raw = item.partition("=")
        if not eq or not raw:
            die(f"--set expects param=value[,value...], got '{item}'")
        try:
            param = tech.param(name)
        except KeyError:
            die(f"{tech.name} has no parameter '{name}' (has: {', '.join(p.name for p in tech.params)})")
        try:
            values[name] = [param.parse(v.strip()) for v in raw.split(",") if v.strip()]
        except ValueError as e:
            die(f"--set {item}: {e}")
    return values


def load_inputs(args) -> Inputs:
    src, img = load_cropped(args.image)
    mask = get_mask(src, img, src.stem, mask_from=args.mask_from, redo=args.redo_mask)
    mp = mask_path(src.stem)
    valid_after = max(src.stat().st_mtime, mp.stat().st_mtime if mask is not None and mp.exists() else 0)
    return Inputs(src, src.stem, normalise(img), mask, load_palette(), valid_after)


def themes_for(tech: Technique, mode: str, themes: list[str]) -> list[str | None]:
    """Full-colour techniques only use a theme for the flat background in remove mode."""
    return themes if (tech.two_tone or mode == "remove") else [None]


def select_modes(requested: str | None, tech: Technique, mask) -> list[str]:
    modes = parse_list(requested, MODES, "background") if requested else list(MODES)
    modes = [m for m in modes if m in tech.modes]
    if mask is None and any(m != "keep" for m in modes):
        modes = [m for m in modes if m == "keep"]
    return modes


def select_themes(requested: str | None, palette: dict) -> list[str]:
    names = list(palette["themes"])
    return parse_list(requested, names, "theme") if requested else names


def row_name(j: Job) -> str:
    return " · ".join(x for x in (j.mode, j.theme) if x)


def write_sheet(inp: Inputs, jobs: list[Job], renders: list[Path], group_of, label_of, dst: Path,
                title: str, note: str, open_it: bool) -> None:
    groups: dict[str, list] = {}
    for j, r in zip(jobs, renders):
        groups.setdefault(group_of(j), []).append((label_of(j), r.with_suffix(".png")))
    sheet.build(list(groups.items()), dst, inp.palette, title, note)
    print(f"sheet: {rel(dst)}")
    if open_it:
        subprocess.run(["open", str(dst)], check=False)


def list_techniques() -> None:
    print("Techniques and their parameters (--set NAME=VALUE):\n")
    for t in TECHNIQUES.values():
        colours = "theme colours" if t.two_tone else "photo colours"
        print(f"{t.name}: {t.summary} ({colours}; backgrounds: {', '.join(t.modes)})")
        for p in t.params:
            sweep = f"   explore tries: {', '.join(map(str, p.sweep))}" if p.sweep else ""
            choices = f" [{'|'.join(p.choices)}]" if p.choices else ""
            print(f"    {p.name:<10} default {str(p.default):<9} {p.help}{choices}{sweep}")
        print()


# ============================================================ commands

def cmd_crop(args) -> None:
    crop(args.image, args.box, args.zoom, args.face, args.pad, args.out, args.force)


def cmd_overview(args) -> None:
    inp = load_inputs(args)
    themes = select_themes(args.theme, inp.palette)
    techs = [TECHNIQUES[t] for t in parse_list(args.only, TECHNIQUES, "technique")] if args.only \
        else list(TECHNIQUES.values())
    jobs = [Job(tech, mode, theme, tech.defaults())
            for tech in techs
            for mode in select_modes(args.background, tech, inp.mask)
            for theme in themes_for(tech, mode, themes)]
    if not jobs:
        die("nothing to render with these options")
    print(f"{rel(inp.src)}: {len(jobs)} renders")
    renders = run_jobs(inp, jobs, args.fresh)
    write_sheet(inp, jobs, renders, lambda j: j.tech.name, row_name, OUT_DIR / inp.stem / "overview.png",
                f"{inp.src.name} · overview (default settings)",
                f"explore a row:  uv run avatar explore <technique> --image {args.image} "
                f"--background <background> --theme <theme>", args.open)


def cmd_explore(args) -> None:
    if not args.technique:
        list_techniques()
        print("Explore one with:  uv run avatar explore <technique> [--set param=v1,v2 ...]")
        return
    tech = get_technique(args.technique)
    values = {p.name: list(p.sweep) if p.sweep else [p.default] for p in tech.params}
    values.update(parse_sets(tech, args.set))
    swept = [n for n, v in values.items() if len(v) > 1]
    combos = [dict(zip(values, c)) for c in itertools.product(*values.values())]

    inp = load_inputs(args)
    themes = select_themes(args.theme, inp.palette)
    modes = select_modes(args.background, tech, inp.mask)
    if not modes:
        die(f"{tech.name} supports the backgrounds: {', '.join(tech.modes)}")
    jobs = [Job(tech, mode, theme, params)
            for mode in modes for theme in themes_for(tech, mode, themes) for params in combos]
    defaults = tech.defaults()

    def label(j: Job) -> str:  # every setting that differs from the default: paste it after --set
        return " ".join(f"{n}={v}" for n, v in j.params.items() if n in swept or v != defaults[n]) or "(defaults)"

    print(f"{rel(inp.src)}: {tech.name}, varying {', '.join(swept) or 'nothing'}: {len(jobs)} renders")
    renders = run_jobs(inp, jobs, args.fresh)
    write_sheet(inp, jobs, renders, row_name, label, OUT_DIR / inp.stem / f"explore-{tech.name}.png",
                f"{inp.src.name} · {tech.name}",
                f"make one:  uv run avatar make {tech.name} --image {args.image} "
                f"--background <background> --theme <theme> --set <label>", args.open)


def cmd_make(args) -> None:
    if not args.technique:
        list_techniques()
        print("Make one with:  uv run avatar make <technique> --background <background> --theme <theme> "
              "[--set param=value ...]")
        return
    tech = get_technique(args.technique)
    params = tech.defaults()
    for name, vals in parse_sets(tech, args.set).items():
        if len(vals) != 1:
            die(f"make takes one value per parameter, got {name}={','.join(map(str, vals))}")
        params[name] = vals[0]
    if args.background not in tech.modes:
        die(f"{tech.name} supports the backgrounds: {', '.join(tech.modes)}")

    inp = load_inputs(args)
    if args.background != "keep" and inp.mask is None:
        die(f"--background {args.background} needs a background mask (install rembg or pass --mask-from)")
    names = list(inp.palette["themes"])
    theme = args.theme or names[0]
    if theme not in names:
        die(f"unknown theme '{theme}' (choose from {', '.join(names)})")
    theme = themes_for(tech, args.background, [theme])[0]
    if args.theme and theme is None:
        warn(f"{tech.name} keeps the photo colours with --background {args.background}, so --theme is ignored")

    job = Job(tech, args.background, theme, params)
    (render,) = run_jobs(inp, [job], args.fresh)
    changed = {k: v for k, v in params.items() if v != tech.defaults()[k]}
    name = "_".join(x for x in (tech.name, args.background, theme, slug(changed)) if x)
    out = OUT_DIR / inp.stem
    out.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        if render.with_suffix(suffix).is_file():
            shutil.copyfile(render.with_suffix(suffix), out / (name + suffix))
            print(f"wrote {rel(out / (name + suffix))}")
    if args.open:
        subprocess.run(["open", str(out / (name + ".png"))], check=False)


# ============================================================ argument parsing

def parse_box(s: str) -> tuple[int, int, int]:
    try:
        x, y, size = (int(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected X,Y,SIZE as integers, e.g. 1100,500,1840")
    if size <= 0:
        raise argparse.ArgumentTypeError("SIZE must be positive")
    return x, y, size


def main(argv: list[str] | None = None) -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog="avatar", description=__doc__, formatter_class=fmt)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    techs = ", ".join(TECHNIQUES)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--image", default=DEFAULT_IMAGE,
                        help=f"filename (extension optional) in assets/cropped/, or a path (default: {DEFAULT_IMAGE})")
    common.add_argument("--open", action="store_true", help="open the result when done")
    common.add_argument("--fresh", action="store_true",
                        help="render again instead of reusing earlier renders (Primitive gives a new random result)")
    m = common.add_argument_group("background mask (cached in work/<image>/mask.png)")
    m.add_argument("--mask-from", type=Path, metavar="PNG",
                   help="use a background-removed PNG (e.g. from Preview) instead of rembg")
    m.add_argument("--redo-mask", action="store_true", help="compute the mask again")

    sheets = argparse.ArgumentParser(add_help=False)
    sheets.add_argument("--background", metavar="LIST", help="comma list of keep,simplify,remove (default: all)")
    sheets.add_argument("--theme", metavar="LIST", help="comma list of themes in palette.toml (default: all)")

    p = sub.add_parser("crop", help="square-crop an original photo into assets/cropped/", formatter_class=fmt,
                       description="Crop assets/original/<image> to a square assets/cropped/<image>.\n"
                                   "Optional: you can also place your own square crop in assets/cropped/.")
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help=f"filename (extension optional) in assets/original/, or a path (default: {DEFAULT_IMAGE})")
    p.add_argument("--zoom", type=float, default=2.6,
                   help="crop side as a multiple of the face height (default 2.6; larger = looser)")
    p.add_argument("--face", type=int, metavar="N", help="which face to crop around, counted left to right")
    p.add_argument("--pad", action="store_true",
                   help="let the crop extend past the photo edge, filling it with black")
    p.add_argument("--box", type=parse_box, metavar="X,Y,SIZE", help="explicit square crop in original pixels (negative values: --box=-100,0,1800 --pad)")
    p.add_argument("--out", metavar="NAME", help="output filename in assets/cropped/ (default: same as input)")
    p.add_argument("--force", action="store_true", help="overwrite an existing crop")
    p.set_defaults(func=cmd_crop)

    p = sub.add_parser("overview", parents=[common, sheets], formatter_class=fmt,
                       help="sheet with every technique × background × theme",
                       description="Render every technique in every background and theme with default settings,\n"
                                   "and put them side by side in out/<image>/overview.png.")
    p.add_argument("--only", metavar="LIST", help=f"comma list of techniques: {techs}")
    p.set_defaults(func=cmd_overview)

    p = sub.add_parser("explore", parents=[common, sheets], formatter_class=fmt,
                       help="sheet with one technique at different settings (no technique: list settings)",
                       description="Render one technique with different settings and put them side by side in\n"
                                   "out/<image>/explore-<technique>.png. Without --set it tries each\n"
                                   "technique's main settings; run without a technique to list them all.\n\n"
                                   "examples:\n"
                                   "  uv run avatar explore primitive --background remove --theme light\n"
                                   "  uv run avatar explore posterize --set colors=4,5 smooth=5,9,13\n"
                                   "  uv run avatar explore primitive --set shape=ellipse shapes=300,600")
    p.add_argument("technique", nargs="?", help=techs)
    p.add_argument("--set", nargs="+", action="extend", metavar="PARAM=V1[,V2]",
                   help="fix a setting (one value) or try several (comma list)")
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("make", parents=[common], formatter_class=fmt,
                       help="the final avatar: one PNG (1024 px) + SVG",
                       description="Make the final avatar as out/<image>/<name>.png (1024 px) and .svg (when the\n"
                                   "technique is vector). Reuses the exact render a sheet showed when one exists.\n\n"
                                   "example:\n"
                                   "  uv run avatar make posterize --background remove --theme light --set colors=5")
    p.add_argument("technique", nargs="?", help=techs)
    p.add_argument("--background", default="keep", choices=MODES, help="background mode (default: keep)")
    p.add_argument("--theme", help="theme in palette.toml (default: the first one)")
    p.add_argument("--set", nargs="+", action="extend", metavar="PARAM=VALUE", help="settings (default: defaults)")
    p.set_defaults(func=cmd_make)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

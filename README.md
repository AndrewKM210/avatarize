# avatarize

A small command-line tool that turns a photo into a stylised avatar using classic image processing, without generative AI. It supports four styles: shapes fitted with [Primitive](https://github.com/fogleman/primitive), low-poly triangles, posterised colour shapes and line art. Each style can keep the photo's background, simplify it or replace it with a flat colour from your palette.

You compare the options on contact sheets, which also show each picture as a small circle like it appears on GitHub or X. Then you export the one you like as a PNG and an SVG.

## Setup

### 1. Prerequisites

- macOS with [Homebrew](https://brew.sh)
- Git (optional, only if Primitive needs the fallback build in step 3)

### 2. Install system tools

From the repo root:

```sh
brew bundle
```

This installs everything in `Brewfile`: `uv`, `go`, `potrace` and `resvg`.

Check:

```sh
uv --version && go version && potrace --version && resvg --version
```

### 3. Install Primitive

```sh
go install github.com/fogleman/primitive@latest
```

The binary lands in `$(go env GOPATH)/bin`. Add it to your `PATH` if it isn't there already (zsh):

```sh
echo 'export PATH="$PATH:$(go env GOPATH)/bin"' >> ~/.zshrc
source ~/.zshrc
primitive 2>&1 | head -3   # prints usage
```

If `go install` fails with a module error, build from source:

```sh
git clone https://github.com/fogleman/primitive.git /tmp/primitive
cd /tmp/primitive
go mod init github.com/fogleman/primitive && go mod tidy
go build -o "$(go env GOPATH)/bin/primitive" .
cd -
```

### 4. Install Python dependencies

```sh
uv sync --extra bg      # dependencies + the `avatar` command; `bg` adds rembg (background removal)
```

Without `--extra bg` everything works except the `simplify` and `remove` background modes (unless you supply a mask with `--mask-from`). On first use, `rembg` downloads its model (`u2net_human_seg` for people, about 170 MB; `isnet-general-use` for `--subject general`, about 180 MB) to `~/.rembg/models/`.

### 5. Set your colours

Edit `palette.toml`. Each theme (`light`, `dark`) has an `accent` (flat background when the background is removed; ink for two-colour techniques) and a `paper` (background for two-colour techniques).

## Usage

Everything runs through one command, `uv run avatar <command>`, and every command has `--help`. `--image` takes a filename in `assets/cropped/` (the extension is optional) and defaults to `profile.jpeg`.

### 1. Add a photo

- **Already square?** Put it in `assets/cropped/`.
- **Not square?** Put it in `assets/original/` and crop it:

  ```sh
  uv run avatar crop --image profile # finds your face and crops around it
  uv run avatar crop --image profile --zoom 3 # looser crop (default 2.6)
  uv run avatar crop --image profile --face 1 # several faces: pick one, counted left to right
  uv run avatar crop --image profile --face 1 --pad --zoom 2.2 # face near the edge: see below
  uv run avatar crop --image profile --box 236,475,806 --force # exact square x,y,size; overwrite
  uv run avatar crop --image van --no-face # not a person: largest centred square
  ```

  If your face is close to the edge of the photo, a square crop can't be centred on it, so `crop` shifts the square and says so. `--pad` keeps your face centred instead and fills the part past the edge with black. With `--background remove`, the black is replaced by the flat background colour, but your shoulder is cut off in a straight line where the photo ends.

### 2. Compare all techniques

```sh
uv run avatar overview --image profile --open
```

This renders every technique (`primitive`, `lowpoly`, `posterize`, `lineart`) with default settings, in every background (`keep`, `simplify`, `remove`) and theme (`light`, `dark` from `palette.toml`), side by side in `out/<image>/overview.png`. Each picture also appears at 48 and 32 px in a circle, on a light and a dark page.

Narrow it down with `--only posterize,lineart`, `--background remove` or `--theme light`.

### 3. Explore one technique

```sh
uv run avatar explore # list techniques and their settings
uv run avatar explore primitive --image profile --background remove --theme light --open
uv run avatar explore posterize --image profile --set colors=4,5 smooth=5,9,13
```

This renders one technique with different settings, side by side in `out/<image>/explore-<technique>.png`. Without `--set` it tries the technique's main settings; `--set name=a,b` tries the values you give, and a single value fixes that setting. Each sheet row is a background and theme, and each label lists the settings that differ from the defaults.

### 4. Make the avatar

```sh
uv run avatar make posterize --image profile --background remove --theme light --set colors=5 smooth=9
```

Copy the row (`--background`, `--theme`) and label (`--set`) of the picture you like from the sheet. This writes `out/<image>/<name>.png` (1024 px) and, for every technique, `<name>.svg`.

Every render is kept in `work/<image>/renders/`, so `make` gives you the exact picture the sheet showed. That matters for `primitive`, which is random: add `--fresh` to any command to render again instead.

### Background mask

The `simplify` and `remove` backgrounds need a mask of the subject. It is computed with rembg on first use and cached in `work/<image>/mask-<subject>.png`; it is recomputed when the cropped photo changes or with `--redo-mask`.

By default the mask uses a model trained on people, which keeps everyone in the photo, so with someone else next to you, check the `remove` column of the overview. For anything else (a pet, a van, a plant), add `--subject general` to `overview`, `explore` and `make`. It uses a general model that keeps the main object (about 180 MB, downloaded on first use):

```sh
uv run avatar overview --image van --subject general
uv run avatar make lineart --image van --subject general --background remove --theme light
```

A landscape has no subject to cut out, so use the `keep` background for it.

If the mask is wrong, make one in Preview: open the cropped photo, Tools > Remove Background, erase anything else with the selection tools, File > Export as PNG. Then pass it once (it replaces the cached mask):

```sh
uv run avatar overview --image profile --mask-from ~/Desktop/profile-nobg.png
```

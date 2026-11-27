# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<3", "Pillow>=10.1,<13"]
# ///
"""Render code-labeled fuse-bead charts and previews from the same grid."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

METRIC = np.array([1, 1.12, 1.12])
PAPER, INK = "#fbfaf6", "#292421"


def lab(rgb):
    srgb = np.asarray(rgb, dtype=np.float64) / 255
    linear = np.where(srgb > .04045, ((srgb + .055) / 1.055)**2.4, srgb / 12.92)
    xyz = (linear @ np.array([[.4124564, .2126729, .0193339],
                              [.3575761, .7151522, .1191920],
                              [.1804375, .072175, .9503041]])) / [.95047, 1, 1.08883]
    x, y, z = np.moveaxis(np.where(xyz > .008856, np.cbrt(xyz), 7.787*xyz + 16/116), -1, 0)
    return np.stack((116*y - 16, 500*(x-y), 200*(y-z)), axis=-1)


def distances(data, centers):
    return ((data[:, None] - centers)**2 * METRIC).sum(axis=-1)


def prepare(path, size, padding, crop):
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert("RGBA")
    if crop is not None:
        l, t, r, b = crop
        if not (0 <= l < r <= im.width and 0 <= t < b <= im.height):
            raise ValueError(f"{path.name}: crop is outside the image")
        im = im.crop(crop)
    if im.getchannel("A").getextrema() == (255, 255):
        # Unmasked illustrations use a square crop rather than stretching.
        im = ImageOps.fit(im, (size, size), Image.Resampling.LANCZOS)
    else:
        bbox = im.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
        if not bbox:
            raise ValueError(f"{path.name}: no visible pixels (alpha >= 128)")
        im = im.crop(bbox)
        span = size - 2*padding
        # Even a feature whose rounded minor dimension would be zero gets a pixel.
        target = tuple(max(1, round(span*d/max(im.size))) for d in im.size)
        im = im.resize(target, Image.Resampling.LANCZOS)
        frame = Image.new("RGBA", (size, size))
        frame.paste(im, ((size-im.width)//2, (size-im.height)//2))
        im = frame
    rgba = np.asarray(im)
    mask = rgba[..., 3] >= 128
    if not mask.any():
        raise ValueError(f"{path.name}: the subject disappeared at this grid size")
    return rgba[..., :3], mask


def fit_palette(samples, maximum):
    # Weighted, deterministic perceptual clustering. Snap centers back to source
    # RGB values; no dithering or diffusion that creates stray single beads.
    unique, count = np.unique(samples, axis=0, return_counts=True)
    colors = lab(unique)
    centers = [colors[np.argmax(count)]]
    nearest = distances(colors, centers)[:, 0]
    for _ in range(1, min(maximum, len(unique))):
        i = np.argmax(nearest*np.sqrt(count))
        if nearest[i] < 1e-12:
            break
        centers.append(colors[i])
        nearest = np.minimum(nearest, distances(colors, [colors[i]])[:, 0])
    centers = np.asarray(centers)
    for _ in range(18):
        ids = nearest_ids(colors, centers)
        updated = centers.copy()
        for k in range(len(centers)):
            m = ids == k
            if np.any(m):
                updated[k] = np.average(colors[m], axis=0, weights=count[m])
        if np.max(np.abs(updated - centers)) < .02:
            break
        centers = updated
    snaps = [np.argmin(distances(colors, [c])[:, 0]) for c in centers]
    selected = sorted(set(snaps), key=lambda i: (colors[i, 0], *unique[i]))
    return [(f"{j:02}", tuple(map(int, unique[i]))) for j, i in enumerate(selected, 1)]


def nearest_ids(data, centers):
    # Limit peak memory on large batches and custom palettes.
    return np.concatenate([distances(data[i:i+4096], centers).argmin(axis=1)
                           for i in range(0, len(data), 4096)])


def read_palette(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"code", "hex"} <= set(reader.fieldnames):
            raise ValueError("palette CSV requires code,hex columns")
        result = []
        codes = set()
        for row in reader:
            code = (row["code"] or "").strip()
            color = (row["hex"] or "").strip().removeprefix("#")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,3}", code) or code in codes:
                raise ValueError(f"invalid or duplicate palette code: {code!r}")
            if not re.fullmatch(r"[0-9a-fA-F]{6}", color):
                raise ValueError(f"invalid hex color for {code}")
            result.append((code, tuple(bytes.fromhex(color))))
            codes.add(code)
            if len(result) > 256:
                raise ValueError("palette CSV must have at most 256 colors")
    if not result:
        raise ValueError("palette CSV is empty")
    return result


def hex_color(rgb):
    return "#" + bytes(rgb).hex().upper()


def render(folder, grid, palette, board, index):
    size = grid.shape[0]
    counts = np.bincount(grid[grid >= 0], minlength=len(palette))
    used = np.flatnonzero(counts)
    assert int(counts.sum()) == int(np.count_nonzero(grid >= 0))
    # Matching placement and color code everywhere, including CSV exports.
    with (folder / "grid.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(["" if k < 0 else palette[k][0] for k in row] for row in grid)
    with (folder / "palette.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "hex", "count"])
        w.writerows((palette[k][0], hex_color(palette[k][1]), int(counts[k])) for k in used)

    step, edge = 30, 48
    columns = min(7, max(2, (size*step + 2*edge - 40)//200))
    top = 120 + 29*((len(used)+columns-1)//columns)
    chart = Image.new("RGB", (size*step + 2*edge, top + size*step + edge), PAPER)
    draw = ImageDraw.Draw(chart)
    big, normal, small = (ImageFont.load_default(size=n) for n in (32, 17, 12))
    draw.text((edge, 16), f"{index:02}  Fuse bead chart", font=big, fill=INK)
    draw.text((edge, 56), f"{size} x {size} cells  |  {len(used)} colors  |  {counts.sum():,} beads",
              font=normal, fill=INK)
    draw.text((edge, 82), f"One bead per cell. Blank = empty. Heavy lines every {board} cells.",
              font=small, fill=INK)
    tiles, beads = {}, {}
    for pos, k in enumerate(used):
        code, rgb = palette[k]
        fg = INK if lab(rgb)[0] > 57 else "#fffaf1"
        tile = Image.new("RGB", (step, step), rgb)
        ImageDraw.Draw(tile).text((step//2, step//2), code, font=small, fill=fg, anchor="mm")
        tiles[k] = tile
        bead = Image.new("RGBA", (12, 12))
        g = ImageDraw.Draw(bead)

        def shade(factor, add=0):
            return tuple(min(255, int(v*factor+add)) for v in rgb) + (255,)

        g.rounded_rectangle((0, 0, 11, 11), radius=2, fill=shade(.86))
        g.ellipse((0, 0, 10, 10), fill=rgb+(255,))
        g.arc((1, 1, 9, 9), 205, 305, fill=shade(.87, 32))
        g.ellipse((4, 4, 6, 6), fill=shade(.60))
        beads[k] = bead
        x, y = edge + 200*(pos%columns), 104 + 29*(pos//columns)
        draw.rounded_rectangle((x, y, x+38, y+22), 3, fill=rgb)
        draw.text((x+19, y+11), code, font=small, fill=fg, anchor="mm")
        draw.text((x+44, y+11), f"{hex_color(rgb)}  {counts[k]:,}", font=normal, fill=INK, anchor="lm")

    preview = Image.new("RGBA", (size*12, size*12))
    for y, x in np.argwhere(grid >= 0):
        k = grid[y, x]
        chart.paste(tiles[k], (edge+x*step, top+y*step))
        preview.paste(beads[k], (x*12, y*12))
    for i in range(size+1):
        x, y = edge+i*step, top+i*step
        heavy = i % board == 0 or i == size
        color = "#63576b" if heavy else "#a1968d" if i%5 == 0 else "#c6bfb6"
        draw.line((x, top, x, top+size*step), fill=color, width=3 if heavy else 1)
        draw.line((edge, y, edge+size*step, y), fill=color, width=3 if heavy else 1)
        if i < size:
            for wx in (edge-15, edge+size*step+15):
                draw.text((wx, y+step//2), str(i+1), font=small, fill=INK, anchor="mm")
            for wy in (top-10, top+size*step+15):
                draw.text((x+step//2, wy), str(i+1), font=small, fill=INK, anchor="mm")
    chart.save(folder / "chart.png", optimize=True)
    preview.save(folder / "preview.png", optimize=True)
    return {"colors": len(used), "beads": int(counts.sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="clean illustrations or Q-style cutouts")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--size", type=int, default=87)
    parser.add_argument("--board", type=int, default=29)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--colors", type=int, default=26, help="maximum automatically fitted colors")
    parser.add_argument("--palette", type=Path, help="explicit code,hex CSV")
    parser.add_argument("--crop", help="left,top,right,bottom (single input only)")
    args = parser.parse_args(argv)
    max_images = 20
    if not 16 <= args.size <= 145 or args.board < 1 or not 0 <= args.padding < args.size//2:
        parser.error("size must be 16..145, board positive, and padding less than half the size")
    if not 1 <= args.colors <= 64 or len(args.images) > max_images:
        parser.error(f"use 1..64 automatic colors and at most {max_images} images per batch")
    if args.palette and args.colors != 26:
        parser.error("use --colors only with an automatic palette")
    # Keep existing outputs intact on re-runs.
    try:
        crop = tuple(map(int, args.crop.split(","))) if args.crop is not None else None
        if crop is not None and (len(crop) != 4 or len(args.images) != 1):
            raise ValueError("--crop requires four integers and a single input")
        frames = [prepare(im, args.size, args.padding, crop) for im in args.images]
        palette = read_palette(args.palette) if args.palette else fit_palette(
            np.concatenate([rgb[m] for rgb, m in frames]), args.colors)
        centers = lab([rgb for _, rgb in palette])
        args.out.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    stats = {"size": args.size, "board": args.board,
             "palette_type": "supplied" if args.palette else "batch-local",
             "palette": dict((code, hex_color(rgb)) for code, rgb in palette),
             "images": []}
    for i, (source, (rgb, mask)) in enumerate(zip(args.images, frames), 1):
        folder = args.out / f"{i:02}"
        folder.mkdir()
        grid = np.full(mask.shape, -1, dtype=int)
        grid[mask] = nearest_ids(lab(rgb[mask]), centers)
        stats["images"].append({"folder": folder.name, "source": source.name,
                                **render(folder, grid, palette, args.board, i)})
    (args.out / "summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False)+"\n",
                                           encoding="utf-8")


if __name__ == "__main__":
    main()

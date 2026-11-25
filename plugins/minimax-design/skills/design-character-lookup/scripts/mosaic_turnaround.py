"""Pixelate two manually identified heads on a character sheet. Requires Pillow."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    for name in ("front", "side"):
        parser.add_argument(f"--{name}", type=int, nargs=4, required=True,
                            metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("choose a new .png output file")
    with Image.open(args.source) as source:
        sheet = ImageOps.exif_transpose(source)
        if sheet.mode not in ("RGB", "RGBA"):
            has_alpha = "A" in sheet.getbands() or "transparency" in sheet.info
            sheet = sheet.convert("RGBA" if has_alpha else "RGB")
    for left, top, right, bottom in (args.front, args.side):
        if not (0 <= left < right <= sheet.width and 0 <= top < bottom <= sheet.height):
            parser.error("head box is empty or outside the image")
        region = sheet.crop((left, top, right, bottom))
        block = max(1, round(region.width / 6))
        small = region.resize((max(1, round(region.width / block)),
                               max(1, round(region.height / block))), Image.Resampling.BOX)
        sheet.paste(small.resize(region.size, Image.Resampling.NEAREST), (left, top))
    with args.output.open("xb") as output:
        sheet.save(output, format="PNG")


if __name__ == "__main__":
    main()

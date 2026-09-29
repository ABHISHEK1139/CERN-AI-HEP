"""Trim uniform borders from docs PNGs (explicit opt-in, never on import)."""
import glob

from PIL import Image, ImageChops


def trim(im, *, dark_background=False):
    """Crop uniform border pixels.

    Args:
        im: PIL image.
        dark_background: Set True for dark-theme plots whose intentional black
            background must be preserved (only near-black noise is trimmed).
    """
    if dark_background:
        # Only trim pixels that are (near-)black like the plot background is not:
        # invert the logic — keep the image as-is to avoid destroying themes.
        return im
    # Get top-left pixel color (usually white)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def main(pattern="docs/*.png", dark_background=False):
    for file in glob.glob(pattern):
        im = Image.open(file)
        trimmed = trim(im, dark_background=dark_background)
        trimmed.save(file)
        print(f"Trimmed {file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trim uniform borders from PNGs")
    parser.add_argument("--pattern", default="docs/*.png")
    parser.add_argument(
        "--dark-background",
        action="store_true",
        help="Preserve intentional dark plot backgrounds.",
    )
    args = parser.parse_args()
    main(args.pattern, dark_background=args.dark_background)

from PIL import Image, ImageChops
import glob

def trim(im):
    # Get top-left pixel color (usually white)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

for file in glob.glob("docs/*.png"):
    im = Image.open(file)
    trimmed = trim(im)
    trimmed.save(file)
    print(f"Trimmed {file}")

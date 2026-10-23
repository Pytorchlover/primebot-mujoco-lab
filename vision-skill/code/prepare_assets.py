"""Prepare repeatable image perturbations, not additional independent scenes."""

from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

root = Path(__file__).resolve().parents[1]
a = root / "assets"
for name in ["washer", "laundry"]:
    im = ImageOps.exif_transpose(Image.open(a / f"{name}.jpg")).convert("RGB")
    im.thumbnail((960, 960))
    im.save(a / f"{name}_normal.jpg", quality=95)
    ImageEnhance.Brightness(im).enhance(0.25).save(a / f"{name}_dark.jpg", quality=95)
    w, h = im.size
    oc = im.copy()
    ImageDraw.Draw(oc).rectangle(
        (0, int(h * 0.38), w, int(h * 0.70)), fill=(80, 80, 80)
    )
    oc.save(a / f"{name}_occluded.jpg", quality=95)
# Blank test is a deliberate fault input; the robot render is a distinct negative input.
Image.new("RGB", (640, 480), (120, 120, 120)).save(a / "blank.jpg")

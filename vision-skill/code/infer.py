"""python code/infer.py path/to/photo.jpg --output results/my-photo.png"""

import argparse, json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from detector import OwlDetector, ROOT

p = argparse.ArgumentParser()
p.add_argument("image", type=Path)
p.add_argument("--output", type=Path, default=ROOT / "results/detection.png")
p.add_argument("--threshold", type=float, default=0.5)
a = p.parse_args()
d = OwlDetector(ROOT / "model", a.threshold)
hits = d.detect(a.image.read_bytes())
im = ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
draw = ImageDraw.Draw(im)
if hits:
    draw.rectangle(d.last["box"], outline="#00cc99", width=4)
    draw.text(
        (10, 10),
        f"washing_machine {d.last['score']:.3f}",
        fill="#00cc99",
        stroke_width=1,
    )
a.output.parent.mkdir(parents=True, exist_ok=True)
im.save(a.output)
print(json.dumps({"accepted": bool(hits), **d.last}, indent=2))

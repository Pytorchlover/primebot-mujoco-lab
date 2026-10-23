"""Real OWL-ViT inference -> carry-clothes Detection (pixel height)."""

import io
import sys
import time
from pathlib import Path
import torch
from PIL import Image, ImageOps
from transformers import OwlViTProcessor, OwlViTForObjectDetection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "src"))
from vision import Detector, Detection

MODEL = "google/owlvit-base-patch32"


class OwlDetector(Detector):
    def __init__(self, model_path, threshold=0.5):
        self.processor = OwlViTProcessor.from_pretrained(model_path)
        self.model = (
            OwlViTForObjectDetection.from_pretrained(model_path).eval().to("cuda")
        )
        self.threshold = threshold
        self.prompt = "a photo of a washing machine"
        self.last = {}

    @torch.inference_mode()
    def detect(self, image: bytes):
        torch.cuda.synchronize()
        start = time.perf_counter()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB")
        w, h = img.size
        inputs = self.processor(text=[[self.prompt]], images=img, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([[h, w]], device="cuda"),
            threshold=0.0,
        )[0]
        # One requested object class. Keep the highest scoring valid box.
        candidates = []
        for score, box in zip(
            result["scores"].cpu().tolist(), result["boxes"].cpu().tolist()
        ):
            x1, y1, x2, y2 = box
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 > x1 and y2 > y1:
                candidates.append((score, [x1, y1, x2, y2]))
        candidates.sort(reverse=True)
        score, box = candidates[0] if candidates else (0.0, [0.0, 0.0, 0.0, 0.0])
        torch.cuda.synchronize()
        self.last = {
            "score": score,
            "box": box,
            "width": w,
            "height": h,
            "latency_ms": 1000 * (time.perf_counter() - start),
        }
        if score < self.threshold:
            return []
        x1, y1, x2, y2 = box
        return [
            Detection(
                "washing_machine",
                score,
                cx=(x1 + x2) / w - 1,
                cy=1 - (y1 + y2) / h,
                height=y2 - y1,
            )
        ]

import json, platform
from pathlib import Path
import numpy as np
import torch, transformers
from detector import OwlDetector

root = Path(__file__).resolve().parents[1]
(root / "results").mkdir(exist_ok=True)
det = OwlDetector(root / "model")
paths = (
    sorted((root / "assets").glob("*_normal.jpg"))
    + sorted((root / "assets").glob("*_dark.jpg"))
    + sorted((root / "assets").glob("*_occluded.jpg"))
    + [root / "assets/no_washer.png", root / "assets/blank.jpg"]
)
for _ in range(5):
    det.detect(paths[0].read_bytes())
torch.cuda.reset_peak_memory_stats()
rows = []
for p in paths:
    det.detect(p.read_bytes())
    rows.append({"file": p.name, **det.last})
    print(rows[-1], flush=True)
times = []
for _ in range(40):
    det.detect((root / "assets/washer_normal.jpg").read_bytes())
    times.append(det.last["latency_ms"])
report = {
    "model": "google/owlvit-base-patch32",
    "prompt": det.prompt,
    "threshold": det.threshold,
    "precision": "float32",
    "batch_size": 1,
    "warmup": 5,
    "timing_repeats": 40,
    "rows": rows,
    "timing_ms": times,
    "latency_median_ms": float(np.median(times)),
    "latency_p95_ms": float(np.percentile(times, 95)),
    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    },
}
(root / "results/probe.json").write_text(json.dumps(report, indent=2))

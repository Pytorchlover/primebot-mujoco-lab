class ConfirmDetector:
    """Require 3 successive accepted observations; never change model scores."""

    def __init__(self, detector, required=3):
        self.detector = detector
        self.required = required
        self.streak = 0
        self.trace = []
        self.prev = None

    def detect(self, image):
        dets = self.detector.detect(image)
        # Identity gate: consecutive boxes must overlap, not just share a class.
        box = self.detector.last["box"]

        def iou(a, b):
            x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = x * y
            return inter / max(
                1e-9,
                (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter,
            )

        if dets:
            self.streak = (
                self.streak + 1
                if self.prev is not None and iou(box, self.prev) >= 0.5
                else 1
            )
            self.prev = box
        else:
            self.streak = 0
            self.prev = None
        passed = self.streak >= self.required
        self.trace.append(
            {**self.detector.last, "streak": self.streak, "passed": passed}
        )
        return dets if passed else []

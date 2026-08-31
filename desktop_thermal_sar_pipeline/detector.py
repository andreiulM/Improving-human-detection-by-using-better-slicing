from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = "person"
    area_px: float = 0.0

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.area_px = (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def is_small_target(self, max_size: int = 30) -> bool:
        return max(self.width, self.height) <= max_size

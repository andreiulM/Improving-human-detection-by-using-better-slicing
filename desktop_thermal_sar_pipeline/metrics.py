from __future__ import annotations
import time
import logging
import csv
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional
import torch

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FrameMetrics:
    frame_id: int
    latency_ms: float
    fps: float
    mode: str
    num_detections: int

    def __str__(self) -> str:
        return (
            f"[Frame {self.frame_id:06d}] "
            f"Mode={self.mode:<12s}  "
            f"Latency={self.latency_ms:6.1f}ms  "
            f"FPS={self.fps:5.1f}  "
            f"Dets={self.num_detections}"
        )

class CUDATimer:
    def __init__(self, device: str = "cuda:0") -> None:
        self.use_cuda = torch.cuda.is_available() and "cuda" in device
        self.elapsed_ms: float = 0.0
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._start_time: float = 0.0

    def __enter__(self) -> "CUDATimer":
        if self.use_cuda:
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0

class MetricsTracker:
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._buffer: Deque[FrameMetrics] = deque(maxlen=window_size)
        self._all_metrics: List[FrameMetrics] = []
        self._total_frames: int = 0

    def record(self, metrics: FrameMetrics) -> None:
        self._buffer.append(metrics)
        self._all_metrics.append(metrics)
        self._total_frames += 1

    @property
    def rolling_fps(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(m.fps for m in self._buffer) / len(self._buffer)

    @property
    def rolling_latency_ms(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(m.latency_ms for m in self._buffer) / len(self._buffer)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def latest(self) -> Optional[FrameMetrics]:
        return self._buffer[-1] if self._buffer else None

    def export_csv(self, filename: str = "pipeline_metrics.csv") -> str:
        if not self._all_metrics:
            return ""
        path = Path(filename)
        keys = self._all_metrics[0].__dict__.keys()
        with open(path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            for m in self._all_metrics:
                dict_writer.writerow(m.__dict__)
        return str(path.absolute())

    def report(self) -> str:
        if not self._all_metrics:
            return "No frames processed yet."
        modes = set(m.mode for m in self._all_metrics)
        mode_stats = {}
        for mode in modes:
            latencies = [m.latency_ms for m in self._all_metrics if m.mode == mode]
            detections = [m.num_detections for m in self._all_metrics if m.mode == mode]
            mode_stats[mode] = {
                "count": len(latencies),
                "lat_avg": np.mean(latencies),
                "lat_p90": np.percentile(latencies, 90),
                "dets_avg": np.mean(detections),
            }
        lines = [
            "═" * 60,
            "  DETAILED PERFORMANCE ANALYSIS REPORT",
            "═" * 60,
            f"  Total Frames Processed : {self._total_frames}",
            f"  Global Avg Latency     : {np.mean([m.latency_ms for m in self._all_metrics]):.1f} ms",
            f"  Global Avg FPS         : {np.mean([m.fps for m in self._all_metrics]):.1f}",
            "",
            "  PER-MODE BREAKDOWN:",
            "  " + "─" * 56,
            f"  {'Mode':<15} | {'Count':>6} | {'Avg Lat':>8} | {'P90 Lat':>8} | {'Avg Det':>7}",
            "  " + "─" * 56,
        ]
        for mode, stats in sorted(mode_stats.items()):
            lines.append(
                f"  {mode:<15} | {stats['count']:>6} | {stats['lat_avg']:>6.1f}ms | "
                f"{stats['lat_p90']:>6.1f}ms | {stats['dets_avg']:>7.1f}"
            )
        lines.append("  " + "─" * 56)
        lines.append("═" * 60)
        return "\n".join(lines)

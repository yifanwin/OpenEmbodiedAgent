from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import numpy as np


def normalize_camera_capture_info(*, source: str, camera_type: str, serial: str | None = None, depth_scale: float | None = None, frame_id: str | None = None, timestamp: str | None = None, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source": str(source),
        "camera_type": str(camera_type),
        "serial": serial,
        "depth_scale": float(depth_scale) if isinstance(depth_scale, (int, float)) else None,
        "frame_id": frame_id,
        "timestamp": timestamp,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


@runtime_checkable
class RgbdCameraAdapter(Protocol):
    """Hardware abstraction interface for RGB-D camera backends."""

    def capture_rgbd(self, *, warmup_frames: int = 6, timeout_s: float = 8.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: ...


class FunctionRgbdCameraAdapter(RgbdCameraAdapter):
    def __init__(self, *, source: str, capture_fn):
        self.source = str(source)
        self.capture_fn = capture_fn

    def capture_rgbd(self, *, warmup_frames: int = 6, timeout_s: float = 8.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return self.capture_fn(
            camera_source=self.source,
            warmup_frames=warmup_frames,
            timeout_s=timeout_s,
        )

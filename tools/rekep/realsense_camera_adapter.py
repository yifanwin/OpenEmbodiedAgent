from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from camera_adapter import RgbdCameraAdapter, normalize_camera_capture_info
from dobot_bridge import capture_realsense_rgbd


class RealsenseRgbdCameraAdapter(RgbdCameraAdapter):
    def __init__(self, camera_source: str):
        self.camera_source = str(camera_source)

    def capture_rgbd(self, *, warmup_frames: int = 6, timeout_s: float = 8.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        rgb, depth, capture_info = capture_realsense_rgbd(
            camera_source=self.camera_source,
            warmup_frames=warmup_frames,
            timeout_s=timeout_s,
        )
        capture_info = normalize_camera_capture_info(
            source=str(capture_info.get("source", self.camera_source)),
            camera_type="realsense_rgbd",
            serial=capture_info.get("serial"),
            depth_scale=capture_info.get("depth_scale"),
            timestamp=capture_info.get("timestamp"),
            extra=capture_info,
        )
        return rgb, depth, capture_info

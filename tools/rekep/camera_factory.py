from __future__ import annotations

from typing import Any, Dict

from camera_adapter import FunctionRgbdCameraAdapter, RgbdCameraAdapter
from hardware_profile import HardwareProfile, coerce_hardware_profile
from dobot_bridge import (
    capture_realsense_rgbd,
    capture_realsense_zmq_rgbd,
    parse_realsense_source,
    parse_realsense_zmq_source,
)


def create_camera_adapter(
    camera_source: str | None = None,
    *,
    hardware_profile: HardwareProfile | Dict[str, Any] | None = None,
) -> RgbdCameraAdapter | None:
    if hardware_profile is not None:
        profile = coerce_hardware_profile(hardware_profile)
        source = str(camera_source or profile.camera_source)
    else:
        source = str(camera_source or "")
    if not source:
        return None
    if parse_realsense_source(source).get("enabled"):
        return FunctionRgbdCameraAdapter(source=source, capture_fn=capture_realsense_rgbd)
    if parse_realsense_zmq_source(source).get("enabled"):
        return FunctionRgbdCameraAdapter(source=source, capture_fn=capture_realsense_zmq_rgbd)
    return None

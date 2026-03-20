from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from camera_adapter import RgbdCameraAdapter, normalize_camera_capture_info
from hardware_profile import HardwareProfile, coerce_hardware_profile


@dataclass
class RealCapture:
    frame_path: str
    depth_path: str
    overlay_path: str = ""
    capture_info: dict | None = None
    keypoint_obs: dict | None = None


class RealReKepEnv:
    def __init__(
        self,
        *,
        state_dir,
        camera_calibration,
        hardware_profile: HardwareProfile | dict | None = None,
        camera_source=None,
        camera_warmup_frames=6,
        camera_timeout_s=8.0,
        camera_adapter: RgbdCameraAdapter | None = None,
    ):
        self.state_dir = Path(state_dir)
        self.frames_dir = self.state_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        profile = coerce_hardware_profile(hardware_profile)
        if camera_source is not None:
            profile.camera_source = str(camera_source)
        self.hardware_profile = profile
        self.camera_source = str(profile.camera_source)
        self.camera_calibration = camera_calibration or {}
        self.camera_warmup_frames = int(camera_warmup_frames)
        self.camera_timeout_s = float(camera_timeout_s)
        self.camera_adapter = camera_adapter

    def capture_rgbd(self, prefix, capture_fn=None):
        frame_path = self.frames_dir / f"{prefix}.png"
        depth_path = self.frames_dir / f"{prefix}.depth.npy"
        if self.camera_adapter is not None:
            rgb, depth, capture_info = self.camera_adapter.capture_rgbd(
                warmup_frames=max(2, self.camera_warmup_frames),
                timeout_s=self.camera_timeout_s,
            )
        else:
            if capture_fn is None:
                raise RuntimeError("capture_fn is required when camera_adapter is not provided")
            rgb, depth, capture_info = capture_fn(
                camera_source=self.camera_source,
                warmup_frames=max(2, self.camera_warmup_frames),
                timeout_s=self.camera_timeout_s,
            )
        capture_info = capture_info or {}
        capture_info = normalize_camera_capture_info(
            source=str(capture_info.get("source", self.camera_source)),
            camera_type=str(capture_info.get("camera_type", "rgbd")),
            serial=capture_info.get("serial"),
            depth_scale=capture_info.get("depth_scale"),
            frame_id=str(frame_path.name),
            timestamp=capture_info.get("timestamp"),
            extra=capture_info,
        )
        cv2.imwrite(str(frame_path), rgb)
        np.save(depth_path, depth)
        return rgb, depth, RealCapture(
            frame_path=str(frame_path),
            depth_path=str(depth_path),
            capture_info=capture_info,
        )

    def add_overlay(self, capture: RealCapture, overlay_path: str, keypoint_obs: dict):
        capture.overlay_path = overlay_path
        capture.keypoint_obs = keypoint_obs
        return capture

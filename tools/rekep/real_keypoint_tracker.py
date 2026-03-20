from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np


@dataclass
class KeypointTrackState:
    keypoints_3d: Dict[str, List[float]] = field(default_factory=dict)
    keypoints_2d: Dict[str, Dict[str, float]] = field(default_factory=dict)
    visible: bool = True
    reason: str = ""
    frame_path: str = ""
    schema: List[Dict[str, Any]] = field(default_factory=list)


class RealKeypointTracker:
    def __init__(self, smoothing_alpha: float = 0.5, max_jump_m: float = 0.20):
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_jump_m = float(max_jump_m)
        self.state = KeypointTrackState()

    def reset(self):
        self.state = KeypointTrackState()

    def update(self, observation: Dict[str, Any], frame_path: str = "") -> Dict[str, Any]:
        keypoints_3d = observation.get("keypoints_3d") if isinstance(observation.get("keypoints_3d"), dict) else {}
        keypoints_2d = observation.get("keypoints_2d") if isinstance(observation.get("keypoints_2d"), dict) else {}
        schema = observation.get("schema") if isinstance(observation.get("schema"), list) else self.state.schema

        smoothed = {}
        warnings = []
        for key, point in keypoints_3d.items():
            current = np.asarray(point, dtype=float)
            previous = self.state.keypoints_3d.get(key)
            if previous is None:
                smoothed[key] = current.tolist()
                continue
            previous_np = np.asarray(previous, dtype=float)
            jump = float(np.linalg.norm(current - previous_np))
            if jump > self.max_jump_m:
                warnings.append(f"keypoint {key} jumped by {jump:.3f}m")
            blended = self.smoothing_alpha * current + (1.0 - self.smoothing_alpha) * previous_np
            smoothed[key] = blended.tolist()

        self.state = KeypointTrackState(
            keypoints_3d=smoothed,
            keypoints_2d=keypoints_2d,
            visible=bool(observation.get("visible", True)),
            reason=str(observation.get("reason", "")),
            frame_path=frame_path,
            schema=schema,
        )
        payload = {
            **observation,
            "keypoints_3d_raw": keypoints_3d,
            "keypoints_3d": smoothed,
            "tracker_warnings": warnings,
            "tracker_state": {
                "frame_path": frame_path,
                "visible": self.state.visible,
                "reason": self.state.reason,
            },
        }
        return payload

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RealGraspState:
    gripper_closed: bool = False
    gripper_position: float | None = None
    grasped_keypoints: List[int] = field(default_factory=list)
    object_bindings: Dict[str, str] = field(default_factory=dict)
    source: str = "unknown"


class RealGraspStateEstimator:
    def __init__(self, *, closed_threshold: float = 0.5):
        self.closed_threshold = float(closed_threshold)
        self.last_state = RealGraspState()

    def update_from_adapter(self, adapter, keypoint_obs: Dict[str, Any] | None = None, stage_info: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = {}
        if hasattr(adapter, "get_runtime_state"):
            try:
                state = adapter.get_runtime_state() or {}
            except Exception:
                state = {}
        gripper_position = state.get("gripper_position")
        gripper_closed = state.get("gripper_closed")
        if gripper_closed is None and gripper_position is not None:
            try:
                gripper_closed = float(gripper_position) >= self.closed_threshold
            except Exception:
                gripper_closed = False
        gripper_closed = bool(gripper_closed)

        grasped_keypoints: List[int] = []
        object_bindings: Dict[str, str] = {}
        schema = keypoint_obs.get("schema") if isinstance((keypoint_obs or {}).get("schema"), list) else []
        grasp_keypoint = (stage_info or {}).get("grasp_keypoint", -1)
        if gripper_closed and grasp_keypoint not in (-1, None):
            grasped_keypoints.append(int(grasp_keypoint))
            for item in schema:
                if not isinstance(item, dict):
                    continue
                if int(item.get("id", -1)) == int(grasp_keypoint):
                    object_bindings[str(grasp_keypoint)] = str(item.get("object", "object"))
                    break

        payload = RealGraspState(
            gripper_closed=gripper_closed,
            gripper_position=gripper_position if isinstance(gripper_position, (int, float)) else None,
            grasped_keypoints=grasped_keypoints,
            object_bindings=object_bindings,
            source=str(state.get("source", "unknown")),
        )
        self.last_state = payload
        return {
            "gripper_closed": payload.gripper_closed,
            "gripper_position": payload.gripper_position,
            "grasped_keypoints": payload.grasped_keypoints,
            "object_bindings": payload.object_bindings,
            "source": payload.source,
        }

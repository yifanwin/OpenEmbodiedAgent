from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class ConstraintMonitorResult:
    ok: bool
    score: float
    status: str
    reasons: List[str] = field(default_factory=list)
    suggested_action: str = "continue"


class RealConstraintMonitor:
    def __init__(self, *, position_tolerance_m: float = 0.08, grasp_required_penalty: float = 1.0):
        self.position_tolerance_m = float(position_tolerance_m)
        self.grasp_required_penalty = float(grasp_required_penalty)

    def evaluate(self, *, stage_info: Dict[str, Any], keypoint_obs: Dict[str, Any], constraint_eval: Dict[str, Any] | None = None) -> Dict[str, Any]:
        keypoints = keypoint_obs.get("keypoints_3d") if isinstance(keypoint_obs.get("keypoints_3d"), dict) else {}
        reasons: List[str] = []
        score = 0.0

        if not keypoints:
            return ConstraintMonitorResult(
                ok=False,
                score=99.0,
                status="no_keypoints",
                reasons=["no 3d keypoints available"],
                suggested_action="reobserve",
            ).__dict__

        grasp_keypoint = stage_info.get("grasp_keypoint", -1)
        release_keypoint = stage_info.get("release_keypoint", -1)
        tracker_warnings = keypoint_obs.get("tracker_warnings") if isinstance(keypoint_obs.get("tracker_warnings"), list) else []
        if tracker_warnings:
            reasons.extend(tracker_warnings)
            score += 0.2 * len(tracker_warnings)

        if grasp_keypoint not in (-1, None):
            if str(grasp_keypoint) not in keypoints:
                reasons.append(f"required grasp keypoint {grasp_keypoint} missing")
                score += self.grasp_required_penalty

        if release_keypoint not in (-1, None):
            if str(release_keypoint) not in keypoints:
                reasons.append(f"required release keypoint {release_keypoint} missing")
                score += self.grasp_required_penalty

        visible = bool(keypoint_obs.get("visible", True))
        if not visible:
            reasons.append("scene/object visibility uncertain")
            score += 0.5

        constraint_eval = constraint_eval or {}
        if constraint_eval:
            if not bool(constraint_eval.get("ok", True)):
                score += 1.0
                reasons.extend(constraint_eval.get("reasons", []))
            if not bool(constraint_eval.get("subgoal_ok", True)):
                score += 0.5
            if not bool(constraint_eval.get("path_ok", True)):
                score += 0.5

        status = "on_track"
        suggested_action = "continue"
        if score >= 1.0:
            status = "deviation"
            suggested_action = "replan"
        elif score > 0.0:
            status = "watch"
            suggested_action = "continue"

        return ConstraintMonitorResult(
            ok=score < 1.0,
            score=score,
            status=status,
            reasons=reasons,
            suggested_action=suggested_action,
        ).__dict__

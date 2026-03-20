from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils import load_functions_from_txt


class RealConstraintEvaluator:
    def __init__(self, *, constraint_tolerance: float = 0.10):
        self.constraint_tolerance = float(constraint_tolerance)

    def _build_grasping_cost_fn(self, grasped_keypoints: List[int] | None = None):
        grasped = set(int(x) for x in (grasped_keypoints or []))

        def get_grasping_cost_by_keypoint_idx(keypoint_idx):
            return 0 if int(keypoint_idx) in grasped else 1

        return get_grasping_cost_by_keypoint_idx

    def _load_constraint_functions(self, path: str, grasped_keypoints: List[int] | None = None):
        p = Path(path)
        if not p.exists():
            return []
        return load_functions_from_txt(str(p), self._build_grasping_cost_fn(grasped_keypoints))

    def evaluate_stage(self, *, stage_info: Dict[str, Any], keypoint_obs: Dict[str, Any], grasped_keypoints: List[int] | None = None, grasp_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        keypoints_map = keypoint_obs.get("keypoints_3d") if isinstance(keypoint_obs.get("keypoints_3d"), dict) else {}
        if not keypoints_map:
            return {
                "ok": False,
                "subgoal_ok": False,
                "path_ok": False,
                "subgoal_scores": [],
                "path_scores": [],
                "reasons": ["missing keypoints for constraint evaluation"],
            }

        ordered = sorted((int(k), v) for k, v in keypoints_map.items())
        keypoints = np.asarray([v for _, v in ordered], dtype=float)
        end_effector = keypoints[0] if len(keypoints) > 0 else np.zeros(3, dtype=float)

        if grasp_state and isinstance(grasp_state.get("grasped_keypoints"), list):
            grasped_keypoints = grasp_state.get("grasped_keypoints")

        subgoal_fns = self._load_constraint_functions(stage_info.get("subgoal_constraints_path", ""), grasped_keypoints)
        path_fns = self._load_constraint_functions(stage_info.get("path_constraints_path", ""), grasped_keypoints)

        subgoal_scores = []
        path_scores = []
        reasons = []

        for idx, fn in enumerate(subgoal_fns, start=1):
            try:
                score = float(fn(end_effector, keypoints))
            except Exception as exc:
                score = 999.0
                reasons.append(f"subgoal constraint {idx} eval error: {exc}")
            subgoal_scores.append(score)
            if score > self.constraint_tolerance:
                reasons.append(f"subgoal constraint {idx} violated: {score:.4f}")

        for idx, fn in enumerate(path_fns, start=1):
            try:
                score = float(fn(end_effector, keypoints))
            except Exception as exc:
                score = 999.0
                reasons.append(f"path constraint {idx} eval error: {exc}")
            path_scores.append(score)
            if score > self.constraint_tolerance:
                reasons.append(f"path constraint {idx} violated: {score:.4f}")

        subgoal_ok = all(score <= self.constraint_tolerance for score in subgoal_scores) if subgoal_scores else True
        path_ok = all(score <= self.constraint_tolerance for score in path_scores) if path_scores else True
        return {
            "ok": subgoal_ok and path_ok,
            "subgoal_ok": subgoal_ok,
            "path_ok": path_ok,
            "subgoal_scores": subgoal_scores,
            "path_scores": path_scores,
            "reasons": reasons,
            "tolerance": self.constraint_tolerance,
            "grasp_state": grasp_state or {},
        }

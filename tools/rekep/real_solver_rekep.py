from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d

import transform_utils as T
from keypoint_proposal import KeypointProposer
from path_solver import PathSolver
from subgoal_solver import SubgoalSolver
from utils import get_config, get_linear_interpolation_steps, load_functions_from_txt

# Right-arm safe zone adapted from dobot_xtrainer_remote/experiments/run_control.py.
RIGHT_ARM_BOUNDS_MIN_M = np.array([-0.25, -0.75, 0.04], dtype=np.float64)
RIGHT_ARM_BOUNDS_MAX_M = np.array([0.45, -0.16, 0.45], dtype=np.float64)
LEFT_ARM_BOUNDS_MIN_M = np.array([-0.45, -0.75, 0.04], dtype=np.float64)
LEFT_ARM_BOUNDS_MAX_M = np.array([0.30, -0.16, 0.45], dtype=np.float64)
DEFAULT_REAL_GRASP_DEPTH_M = 0.03
DEFAULT_RELEASE_OPEN_WAIT_S = 0.15
DEFAULT_POST_GRASP_LIFT_M = 0.08
DEFAULT_PRE_RELEASE_HOVER_M = 0.06
DEFAULT_PRE_RELEASE_DESCEND_M = 0.015
DEFAULT_POST_RELEASE_RETREAT_M = 0.08
DEFAULT_TOOL_COLLISION_LOCAL_POINTS_M = np.array(
    [
        [0.000, 0.000, -0.030],
        [0.018, 0.018, -0.010],
        [-0.018, 0.018, -0.010],
        [0.018, -0.018, -0.010],
        [-0.018, -0.018, -0.010],
        [0.014, 0.012, 0.015],
        [-0.014, 0.012, 0.015],
        [0.014, -0.012, 0.015],
        [-0.014, -0.012, 0.015],
        [0.010, 0.000, 0.040],
        [-0.010, 0.000, 0.040],
        [0.000, 0.008, 0.050],
        [0.000, -0.008, 0.050],
    ],
    dtype=np.float64,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _normalized_axis(vector: Any, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.asarray(fallback, dtype=np.float64).reshape(-1)
    arr = arr[:3]
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        arr = np.asarray(fallback, dtype=np.float64).reshape(-1)[:3]
        norm = np.linalg.norm(arr)
    return arr / max(norm, 1e-9)


class DummyIKSolver:
    """Cheap reachability stub for the real Dobot path.

    The official ReKep solver expects an IK object with Lula-like fields. We keep the
    same interface but only enforce workspace bounds, since live Dobot execution uses
    Cartesian MovL directly.
    """

    def __init__(self, *, bounds_min: np.ndarray, bounds_max: np.ndarray, joint_dim: int = 6):
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.joint_dim = int(joint_dim)

    def solve(
        self,
        target_pose_homo,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        target_pose_homo = np.asarray(target_pose_homo, dtype=np.float64)
        pos = target_pose_homo[:3, 3]
        success = bool(np.all(pos >= self.bounds_min) and np.all(pos <= self.bounds_max))
        cspace_dim = max(self.joint_dim, len(initial_joint_pos) if initial_joint_pos is not None else self.joint_dim)
        cspace = np.zeros(int(cspace_dim), dtype=np.float64)
        if initial_joint_pos is not None:
            seed = np.asarray(initial_joint_pos, dtype=np.float64).flatten()
            cspace[: min(len(seed), len(cspace))] = seed[: min(len(seed), len(cspace))]
        return SimpleNamespace(
            success=success,
            position_error=0.0 if success else float(np.linalg.norm(np.clip(pos, self.bounds_min, self.bounds_max) - pos)),
            num_descents=0 if success else max(1, int(max_iterations)),
            cspace_position=cspace,
        )


class RealSolverContext:
    def __init__(self, *, config: Dict[str, Any], adapter, emit_progress):
        self.config = config
        self.adapter = adapter
        self.emit_progress = emit_progress
        self.bounds_min = np.asarray(self.config["subgoal_solver"]["bounds_min"], dtype=np.float64)
        self.bounds_max = np.asarray(self.config["subgoal_solver"]["bounds_max"], dtype=np.float64)
        self.ik_solver = DummyIKSolver(bounds_min=self.bounds_min, bounds_max=self.bounds_max)
        self.subgoal_solver = SubgoalSolver(self.config["subgoal_solver"], self.ik_solver, np.zeros(6, dtype=np.float64))
        self.path_solver = PathSolver(self.config["path_solver"], self.ik_solver, np.zeros(6, dtype=np.float64))
        self.sdf_voxels = np.zeros((24, 24, 24), dtype=np.float32)
        self.current_ee_pose = None
        self.scene_keypoints = None
        self.rigid_group_ids = None
        self.attached_group = None
        self.attached_local_points: Dict[int, np.ndarray] = {}
        self.grasped_keypoints: set[int] = set()
        self.active_arm = "right"

    def set_initial_scene(self, keypoints_3d: Dict[str, List[float]], rigid_group_ids: Dict[str, int], *, arm: str = "right"):
        ordered_ids = sorted(int(k) for k in keypoints_3d.keys())
        self.scene_keypoints = np.asarray([keypoints_3d[str(idx)] for idx in ordered_ids], dtype=np.float64)
        self.rigid_group_ids = np.asarray([int(rigid_group_ids.get(str(idx), -1)) for idx in ordered_ids], dtype=np.int32)
        self.active_arm = arm
        self.current_ee_pose = self._get_current_ee_pose(arm)

    def _get_current_ee_pose(self, arm: str) -> np.ndarray:
        if hasattr(self.adapter, "get_tool_pose_mm_deg"):
            raw = self.adapter.get_tool_pose_mm_deg(arm)
            if isinstance(raw, (list, tuple)) and len(raw) >= 6:
                pos_m = np.asarray(raw[:3], dtype=np.float64) / 1000.0
                quat = T.euler2quat(np.deg2rad(np.asarray(raw[3:6], dtype=np.float64)))
                return np.concatenate([pos_m, quat], axis=0)
        # Conservative fallback used only if remote pose query is unavailable.
        return np.array([0.18, -0.42, 0.18, *T.euler2quat(np.deg2rad(np.array([180.0, 0.0, 0.0])))], dtype=np.float64)

    def refresh_attached_world_keypoints(self):
        if self.scene_keypoints is None or not self.attached_local_points or self.current_ee_pose is None:
            return
        ee_homo = T.pose2mat([self.current_ee_pose[:3], self.current_ee_pose[3:]])
        for idx, local_point in self.attached_local_points.items():
            world = ee_homo @ np.array([float(local_point[0]), float(local_point[1]), float(local_point[2]), 1.0], dtype=np.float64)
            self.scene_keypoints[idx] = world[:3]

    def movable_mask(self) -> np.ndarray:
        mask = np.zeros((len(self.scene_keypoints) + 1,), dtype=bool)
        mask[0] = True
        if self.attached_group is None:
            return mask
        mask[1:] = self.rigid_group_ids == int(self.attached_group)
        return mask

    def full_keypoints(self) -> np.ndarray:
        self.refresh_attached_world_keypoints()
        return np.concatenate([self.current_ee_pose[:3][None], self.scene_keypoints], axis=0)

    def current_collision_points(self) -> np.ndarray:
        if self.current_ee_pose is None:
            return np.zeros((0, 3), dtype=np.float64)
        ee_homo = T.pose2mat([self.current_ee_pose[:3], self.current_ee_pose[3:]])
        tool_points = (
            DEFAULT_TOOL_COLLISION_LOCAL_POINTS_M @ ee_homo[:3, :3].T
        ) + ee_homo[:3, 3]
        attached_points = []
        if self.attached_local_points:
            self.refresh_attached_world_keypoints()
            for idx in sorted(self.attached_local_points.keys()):
                attached_points.append(self.scene_keypoints[int(idx)])
        if attached_points:
            return np.vstack([tool_points, np.asarray(attached_points, dtype=np.float64)])
        return tool_points

    def mark_grasped_group(self, grasp_keypoint: int):
        if grasp_keypoint < 0:
            return
        self.refresh_attached_world_keypoints()
        group = int(self.rigid_group_ids[grasp_keypoint])
        self.attached_group = group
        ee_homo_inv = np.linalg.inv(T.pose2mat([self.current_ee_pose[:3], self.current_ee_pose[3:]]))
        self.attached_local_points = {}
        for idx, group_id in enumerate(self.rigid_group_ids):
            if int(group_id) != group:
                continue
            point = np.append(self.scene_keypoints[idx], 1.0)
            local_point = ee_homo_inv @ point
            self.attached_local_points[idx] = local_point[:3]
        self.grasped_keypoints.add(int(grasp_keypoint))

    def clear_grasp(self):
        self.refresh_attached_world_keypoints()
        self.attached_group = None
        self.attached_local_points = {}
        self.grasped_keypoints.clear()



def build_real_solver_config(*, arm: str = "right", grasp_depth_m: float = DEFAULT_REAL_GRASP_DEPTH_M) -> Dict[str, Any]:
    config = copy.deepcopy(get_config())
    arm_name = str(arm or "right").strip().lower()
    if arm_name == "left":
        bounds_min = LEFT_ARM_BOUNDS_MIN_M.copy()
        bounds_max = LEFT_ARM_BOUNDS_MAX_M.copy()
    else:
        bounds_min = RIGHT_ARM_BOUNDS_MIN_M.copy()
        bounds_max = RIGHT_ARM_BOUNDS_MAX_M.copy()

    for section in ("main", "path_solver", "subgoal_solver", "keypoint_proposer", "visualizer"):
        if section not in config:
            continue
        config[section]["bounds_min"] = bounds_min.copy()
        config[section]["bounds_max"] = bounds_max.copy()

    config["main"]["grasp_depth"] = float(grasp_depth_m)
    config["main"]["grasp_retry_backoff"] = 0.012
    config["main"]["grasp_retry_settle_time"] = 0.15
    config["main"]["post_grasp_lift"] = DEFAULT_POST_GRASP_LIFT_M
    config["main"]["pre_release_hover"] = DEFAULT_PRE_RELEASE_HOVER_M
    config["main"]["pre_release_descend"] = DEFAULT_PRE_RELEASE_DESCEND_M
    config["main"]["post_release_retreat"] = DEFAULT_POST_RELEASE_RETREAT_M
    # Real Dobot tool frame convention used here:
    # local +Z = tool forward / insertion direction, local +X = tool right.
    config["main"]["tool_forward_local_axis"] = [0.0, 0.0, 1.0]
    config["main"]["tool_right_local_axis"] = [1.0, 0.0, 0.0]
    config["main"]["grasp_retry_offsets"] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.008, 0.0],
        [0.0, -0.008, 0.0],
    ]
    config["subgoal_solver"]["grasp_axis_local"] = [0.0, 0.0, 1.0]
    config["subgoal_solver"]["grasp_preferred_world_dir"] = [0.0, 0.0, -1.0]
    # The real Dobot path does not have a high-rate impedance controller, so keep the
    # optimization lighter than the simulator defaults.
    config["path_solver"]["sampling_maxfun"] = min(int(config["path_solver"].get("sampling_maxfun", 5000)), 1200)
    config["subgoal_solver"]["sampling_maxfun"] = min(int(config["subgoal_solver"].get("sampling_maxfun", 5000)), 1200)
    config["path_solver"]["opt_pos_step_size"] = 0.10
    config["path_solver"]["opt_rot_step_size"] = 0.60
    config["path_solver"]["opt_interpolate_pos_step_size"] = 0.03
    config["path_solver"]["opt_interpolate_rot_step_size"] = 0.18
    config["keypoint_proposer"]["max_mask_ratio"] = 0.45
    return config



def _depth_to_base_points(depth_image: np.ndarray, camera_calibration: Dict[str, Any]) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    color = (camera_calibration or {}).get("color_intrinsic") or {}
    transform = np.asarray((camera_calibration or {}).get("T_base_camera") or np.eye(4), dtype=np.float64)
    fx = float(color["fx"])
    fy = float(color["fy"])
    cx = float(color["cx"])
    cy = float(color["cy"])
    h, w = depth.shape[:2]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float64)
    x = (uu - fx * 0 + uu - uu)  # keep dtype warm; overwritten below
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    ones = np.ones_like(z)
    cam = np.stack([x, y, z, ones], axis=-1)
    base = cam.reshape(-1, 4) @ transform.T
    return base[:, :3].reshape(h, w, 3)



def _build_cluster_mask(
    points_base: np.ndarray,
    depth_image: np.ndarray,
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    plane_distance_m: float = 0.008,
    object_height_thresh_m: float = 0.010,
    dbscan_eps_m: float = 0.030,
    dbscan_min_points: int = 120,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid_mask = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    if not np.any(valid_mask):
        raise RuntimeError("no valid depth points for candidate proposal")

    bounds_pad_min = np.asarray(bounds_min, dtype=np.float64).copy()
    bounds_pad_max = np.asarray(bounds_max, dtype=np.float64).copy()
    bounds_pad_min[:2] -= 0.05
    bounds_pad_max[:2] += 0.05
    bounds_pad_max[2] += 0.05
    in_workspace = np.all(points_base >= bounds_pad_min[None, None, :], axis=-1) & np.all(points_base <= bounds_pad_max[None, None, :], axis=-1)
    candidate_mask = valid_mask & in_workspace
    candidate_points = points_base[candidate_mask]
    if candidate_points.shape[0] < 300:
        raise RuntimeError(f"insufficient workspace points for candidate proposal: {candidate_points.shape[0]}")

    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane_offset = -float(np.median(candidate_points[:, 2]))
    plane_inlier_count = 0
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(candidate_points.astype(np.float64))
        plane_model, inliers = pcd.segment_plane(distance_threshold=plane_distance_m, ransac_n=3, num_iterations=150)
        plane_model = np.asarray(plane_model, dtype=np.float64)
        plane_inlier_count = int(len(inliers))
        if plane_model.shape[0] == 4 and abs(plane_model[2]) >= 0.75 and plane_inlier_count >= max(200, int(0.2 * candidate_points.shape[0])):
            plane_normal = plane_model[:3]
            plane_offset = float(plane_model[3])
            if plane_normal[2] < 0:
                plane_normal = -plane_normal
                plane_offset = -plane_offset
    except Exception:
        pass

    signed_dist = np.tensordot(points_base, plane_normal, axes=([-1], [0])) + plane_offset
    object_mask = candidate_mask & (signed_dist > object_height_thresh_m)
    if int(np.count_nonzero(object_mask)) < 200:
        object_mask = candidate_mask

    object_points = points_base[object_mask]
    labels = np.full(depth.shape[:2], -1, dtype=np.int32)
    num_clusters = 0
    if object_points.shape[0] >= max(80, dbscan_min_points):
        try:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points.astype(np.float64))
            cluster_labels = np.asarray(object_pcd.cluster_dbscan(eps=dbscan_eps_m, min_points=dbscan_min_points, print_progress=False), dtype=np.int32)
            positive = sorted(int(v) for v in np.unique(cluster_labels) if int(v) >= 0)
            if positive:
                remap = {label: idx + 1 for idx, label in enumerate(positive)}
                coords = np.argwhere(object_mask)
                for (v, u), label in zip(coords, cluster_labels):
                    if int(label) >= 0:
                        labels[int(v), int(u)] = remap[int(label)]
                num_clusters = len(remap)
        except Exception:
            num_clusters = 0

    if num_clusters == 0:
        labels[object_mask] = 1
        num_clusters = 1

    debug = {
        "valid_point_count": int(np.count_nonzero(valid_mask)),
        "workspace_point_count": int(np.count_nonzero(candidate_mask)),
        "object_point_count": int(np.count_nonzero(object_mask)),
        "plane_normal": plane_normal.tolist(),
        "plane_offset": float(plane_offset),
        "plane_inlier_count": int(plane_inlier_count),
        "cluster_count": int(num_clusters),
    }
    return labels, debug



def _make_mask_debug_image(rgb_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    overlay = rgb_bgr.copy()
    unique_labels = [int(v) for v in np.unique(labels) if int(v) > 0]
    if not unique_labels:
        return overlay
    colors = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 180, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 0, 0),
    ]
    for idx, label in enumerate(unique_labels):
        mask = labels == int(label)
        color = np.array(colors[idx % len(colors)], dtype=np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cv2.putText(
            overlay,
            f"obj{label}",
            (int(np.median(xs)), int(np.median(ys))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return overlay



def propose_candidate_keypoints(
    *,
    rgb_bgr: np.ndarray,
    depth_image: np.ndarray,
    camera_calibration: Dict[str, Any],
    output_prefix: str,
    output_dir: str | Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds_min = np.asarray(config["keypoint_proposer"]["bounds_min"], dtype=np.float64)
    bounds_max = np.asarray(config["keypoint_proposer"]["bounds_max"], dtype=np.float64)
    points_base = _depth_to_base_points(depth_image, camera_calibration)
    masks, mask_debug = _build_cluster_mask(points_base, depth_image, bounds_min=bounds_min, bounds_max=bounds_max)

    proposer = KeypointProposer(config["keypoint_proposer"])
    rgb_rgb = cv2.cvtColor(np.asarray(rgb_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    candidate_points, projected_rgb, metadata = proposer.get_keypoints(rgb_rgb, points_base, masks, return_metadata=True)
    candidate_pixels = np.asarray(metadata.get("candidate_pixels", []), dtype=np.int32)
    rigid_groups = np.asarray(metadata.get("candidate_rigid_group_ids", []), dtype=np.int32)
    if candidate_points.shape[0] == 0:
        raise RuntimeError("DINOv2 proposer returned zero candidate keypoints")

    overlay_bgr = cv2.cvtColor(np.asarray(projected_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    overlay_path = output_dir / f"{output_prefix}.keypoints.png"
    mask_path = output_dir / f"{output_prefix}.masks.png"
    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(mask_path), _make_mask_debug_image(rgb_bgr, masks))

    keypoints_2d = {}
    keypoints_3d = {}
    rigid_group_payload = {}
    for idx, point in enumerate(candidate_points):
        key = str(idx)
        pixel = candidate_pixels[idx] if idx < len(candidate_pixels) else np.array([0, 0], dtype=np.int32)
        keypoints_2d[key] = {
            "u": float(pixel[1]),
            "v": float(pixel[0]),
            "label": f"candidate_{idx}",
            "rigid_group_id": int(rigid_groups[idx]) if idx < len(rigid_groups) else -1,
        }
        keypoints_3d[key] = [float(v) for v in point.tolist()]
        rigid_group_payload[key] = int(rigid_groups[idx]) if idx < len(rigid_groups) else -1

    debug_path = output_dir / f"{output_prefix}.proposal_debug.json"
    debug_payload = {
        "proposal_method": "dinov2_rgbd_candidate_clustering",
        "mask_debug": mask_debug,
        "candidate_count": int(candidate_points.shape[0]),
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "rigid_group_ids": rigid_group_payload,
        "mask_overlay_path": str(mask_path),
        "candidate_overlay_path": str(overlay_path),
    }
    debug_path.write_text(json.dumps(_jsonable(debug_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "visible": True,
        "reason": f"proposed {candidate_points.shape[0]} DINOv2 candidate keypoints from RGB-D foreground clusters",
        "proposal_method": "dinov2_rgbd_candidate_clustering",
        "raw_output": "",
        "vlm": {},
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "rigid_group_ids": rigid_group_payload,
        "schema": [
            {"id": idx, "label": f"candidate_{idx}", "object": f"rigid_group_{rigid_group_payload[str(idx)]}", "purpose": "candidate_keypoint"}
            for idx in range(candidate_points.shape[0])
        ],
        "overlay_path": str(overlay_path),
        "mask_overlay_path": str(mask_path),
        "proposal_debug_path": str(debug_path),
        "proposal_debug": debug_payload,
    }



def _pose_quat_to_movel_action(pose_quat: np.ndarray, *, arm: str) -> Dict[str, Any]:
    pos_mm = np.asarray(pose_quat[:3], dtype=np.float64) * 1000.0
    euler_deg = np.rad2deg(T.quat2euler(np.asarray(pose_quat[3:], dtype=np.float64)))
    return {
        "type": "movel",
        "arm": str(arm),
        "units": "mm_deg",
        "pose": [float(pos_mm[0]), float(pos_mm[1]), float(pos_mm[2]), float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])],
    }



def _grasp_cost_closure(active_grasped: set[int]):
    def _cost(keypoint_idx):
        return 0 if int(keypoint_idx) in active_grasped else 1
    return _cost



def _load_stage_constraints(program_dir: str | Path, stage_info: Dict[str, Any], active_grasped: set[int]) -> Tuple[List[Any], List[Any]]:
    subgoal_path = stage_info.get("subgoal_constraints_path")
    path_path = stage_info.get("path_constraints_path")
    grasp_cost_fn = _grasp_cost_closure(active_grasped)
    subgoal_constraints = load_functions_from_txt(str(subgoal_path), grasp_cost_fn) if subgoal_path and Path(subgoal_path).exists() else []
    path_constraints = load_functions_from_txt(str(path_path), grasp_cost_fn) if path_path and Path(path_path).exists() else []
    return subgoal_constraints, path_constraints


def _sanitize_stage_constraints(stage_info: Dict[str, Any], subgoal_constraints: List[Any], path_constraints: List[Any]) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "release_stage_subgoal_constraints_original": int(len(subgoal_constraints)),
        "release_stage_subgoal_constraints_kept": int(len(subgoal_constraints)),
        "release_stage_constraint_sanitized": False,
    }
    is_release_stage = int(stage_info.get("release_keypoint", -1)) >= 0
    if is_release_stage and len(subgoal_constraints) > 1:
        subgoal_constraints = subgoal_constraints[:1]
        debug["release_stage_subgoal_constraints_kept"] = 1
        debug["release_stage_constraint_sanitized"] = True
    return subgoal_constraints, path_constraints, debug



def _control_points_to_actions(current_ee_pose: np.ndarray, control_points: np.ndarray, *, arm: str) -> List[Dict[str, Any]]:
    if control_points is None or len(control_points) == 0:
        return []
    actions = []
    last_pose = np.asarray(current_ee_pose, dtype=np.float64)
    for pose in control_points:
        pose = np.asarray(pose, dtype=np.float64)
        pos_delta = np.linalg.norm(pose[:3] - last_pose[:3])
        rot_delta = np.linalg.norm(np.rad2deg(T.quat2euler(pose[3:]) - T.quat2euler(last_pose[3:])))
        if pos_delta < 0.005 and rot_delta < 3.0:
            continue
        actions.append(_pose_quat_to_movel_action(pose, arm=arm))
        last_pose = pose
    return actions



def _execute_action_list(adapter, actions: List[Dict[str, Any]], *, execute_motion: bool, action_interval_s: float, emit_progress, stage: int) -> Tuple[List[Dict[str, Any]], str]:
    records: List[Dict[str, Any]] = []
    execution_error = ""
    for idx, action in enumerate(actions, start=1):
        try:
            result = adapter.execute_action(action, execute_motion=bool(execute_motion))
            records.append({"index": idx, "ok": True, "result": result, "action": action})
            emit_progress(f"[rekep-solver][stage={stage}] action[{idx}] {action.get('type')} ok")
            if bool(execute_motion) and float(action_interval_s) > 0.0 and idx < len(actions):
                emit_progress(f"[rekep-solver][stage={stage}] action[{idx}] cooldown {float(action_interval_s):.1f}s")
                time.sleep(float(action_interval_s))
        except Exception as exc:
            records.append({"index": idx, "ok": False, "error": str(exc), "action": action})
            execution_error = str(exc)
            emit_progress(f"[rekep-solver][stage={stage}] action[{idx}] {action.get('type')} failed: {exc}")
            break
    return records, execution_error



def _pose_with_local_offset(pose_quat: np.ndarray, local_offset: np.ndarray) -> np.ndarray:
    pose_quat = np.asarray(pose_quat, dtype=np.float64)
    local_offset = np.asarray(local_offset, dtype=np.float64)
    pose_out = pose_quat.copy()
    pose_out[:3] = pose_out[:3] + T.quat2mat(pose_quat[3:]) @ local_offset
    return pose_out


def _pose_with_world_offset(pose_quat: np.ndarray, world_offset: np.ndarray) -> np.ndarray:
    pose_quat = np.asarray(pose_quat, dtype=np.float64)
    world_offset = np.asarray(world_offset, dtype=np.float64)
    pose_out = pose_quat.copy()
    pose_out[:3] = pose_out[:3] + world_offset
    return pose_out


def _tool_forward_axis(config: Dict[str, Any]) -> np.ndarray:
    main_cfg = config.get("main", {}) if isinstance(config, dict) else {}
    return _normalized_axis(main_cfg.get("tool_forward_local_axis", [1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0], dtype=np.float64))


def _clamp_pose_to_workspace(pose_quat: np.ndarray, ctx: RealSolverContext, *, margin: float = 0.005) -> np.ndarray:
    pose = np.asarray(pose_quat, dtype=np.float64).copy()
    lower = np.asarray(ctx.bounds_min, dtype=np.float64) + float(margin)
    upper = np.asarray(ctx.bounds_max, dtype=np.float64) - float(margin)
    pose[:3] = np.clip(pose[:3], lower, upper)
    return pose


def _update_ctx_pose_from_action(ctx: RealSolverContext, action: Dict[str, Any]) -> None:
    if str(action.get("type", "")).strip().lower() != "movel":
        return
    pose = np.asarray(action.get("pose", []), dtype=np.float64).reshape(-1)
    if pose.size < 6:
        return
    pos_m = pose[:3] / 1000.0
    quat = T.euler2quat(np.deg2rad(pose[3:6]))
    ctx.current_ee_pose = np.concatenate([pos_m, quat], axis=0)
    ctx.refresh_attached_world_keypoints()


def _make_transport_actions(ctx: RealSolverContext, *, target_pose: np.ndarray, arm: str) -> List[Dict[str, Any]]:
    current_pose = np.asarray(ctx.current_ee_pose, dtype=np.float64).copy()
    target_pose = np.asarray(target_pose, dtype=np.float64).copy()
    main_cfg = ctx.config.get("main", {})
    hover = float(main_cfg.get("pre_release_hover", DEFAULT_PRE_RELEASE_HOVER_M))
    descend = float(main_cfg.get("pre_release_descend", DEFAULT_PRE_RELEASE_DESCEND_M))
    safe_top_z = float(ctx.bounds_max[2]) - 0.005
    transport_z = min(safe_top_z, max(current_pose[2], target_pose[2] + hover))
    carry_pose = current_pose.copy()
    carry_pose[2] = transport_z
    hover_pose = target_pose.copy()
    hover_pose[2] = min(safe_top_z, target_pose[2] + hover)
    place_pose = target_pose.copy()
    place_pose[2] = min(hover_pose[2], target_pose[2] + max(descend, 0.0))
    carry_pose = _clamp_pose_to_workspace(carry_pose, ctx)
    hover_pose = _clamp_pose_to_workspace(hover_pose, ctx)
    place_pose = _clamp_pose_to_workspace(place_pose, ctx)
    poses = np.asarray([carry_pose, hover_pose, place_pose], dtype=np.float64)
    return _control_points_to_actions(current_pose, poses, arm=arm)


def _execute_post_grasp_lift(
    ctx: RealSolverContext,
    *,
    execute_motion: bool,
    action_interval_s: float,
    emit_progress,
    stage: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    lift = float(ctx.config["main"].get("post_grasp_lift", DEFAULT_POST_GRASP_LIFT_M))
    if lift <= 1e-6:
        return [], [], ""
    target_pose = _pose_with_world_offset(np.asarray(ctx.current_ee_pose, dtype=np.float64), np.array([0.0, 0.0, lift], dtype=np.float64))
    target_pose = _clamp_pose_to_workspace(target_pose, ctx)
    actions = _control_points_to_actions(np.asarray(ctx.current_ee_pose, dtype=np.float64), np.asarray([target_pose], dtype=np.float64), arm=ctx.active_arm)
    if not actions:
        return [], [], ""
    records, execution_error = _execute_action_list(
        ctx.adapter,
        actions,
        execute_motion=execute_motion,
        action_interval_s=action_interval_s,
        emit_progress=emit_progress,
        stage=stage,
    )
    if not execution_error:
        _update_ctx_pose_from_action(ctx, actions[-1])
    return actions, records, execution_error



def _execute_grasp_routine(ctx: RealSolverContext, *, grasp_keypoint: int, execute_motion: bool, action_interval_s: float, emit_progress, stage: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    pregrasp_pose = np.asarray(ctx.current_ee_pose, dtype=np.float64).copy()
    retry_offsets = [np.asarray(v, dtype=np.float64) for v in ctx.config["main"].get("grasp_retry_offsets", [[0.0, 0.0, 0.0]])]
    retreat_backoff = float(ctx.config["main"].get("grasp_retry_backoff", 0.012))
    settle_time = float(ctx.config["main"].get("grasp_retry_settle_time", 0.15))
    grasp_depth = float(ctx.config["main"].get("grasp_depth", DEFAULT_REAL_GRASP_DEPTH_M))
    forward_axis = _tool_forward_axis(ctx.config)

    actions: List[Dict[str, Any]] = []
    execution_records: List[Dict[str, Any]] = []
    execution_error = ""
    for attempt_idx, offset in enumerate(retry_offsets, start=1):
        if attempt_idx > 1:
            retry_pose = _pose_with_local_offset(pregrasp_pose, offset - retreat_backoff * forward_axis)
            actions.append({"type": "open_gripper", "arm": ctx.active_arm})
            actions.append(_pose_quat_to_movel_action(retry_pose, arm=ctx.active_arm))
        grasp_pose = _pose_with_local_offset(pregrasp_pose, offset + grasp_depth * forward_axis)
        actions.append(_pose_quat_to_movel_action(grasp_pose, arm=ctx.active_arm))
        actions.append({"type": "close_gripper", "arm": ctx.active_arm})
        if settle_time > 0:
            actions.append({"type": "wait", "seconds": settle_time})
        # The live Dobot stack does not expose a reliable object-contact signal yet.
        # Execute the first grasp attempt and stop there; offsets remain available for manual tuning.
        break

    execution_records, execution_error = _execute_action_list(
        ctx.adapter,
        actions,
        execute_motion=execute_motion,
        action_interval_s=action_interval_s,
        emit_progress=emit_progress,
        stage=stage,
    )
    if not execution_error and actions:
        for action in reversed(actions):
            if str(action.get("type", "")).strip().lower() == "movel":
                _update_ctx_pose_from_action(ctx, action)
                break
        ctx.mark_grasped_group(int(grasp_keypoint))
    return actions, execution_records, execution_error



def _execute_release_routine(ctx: RealSolverContext, *, execute_motion: bool, action_interval_s: float, emit_progress, stage: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    actions = [
        {"type": "open_gripper", "arm": ctx.active_arm},
        {"type": "wait", "seconds": DEFAULT_RELEASE_OPEN_WAIT_S},
    ]
    records, execution_error = _execute_action_list(
        ctx.adapter,
        actions,
        execute_motion=execute_motion,
        action_interval_s=action_interval_s,
        emit_progress=emit_progress,
        stage=stage,
    )
    if execution_error:
        return actions, records, execution_error

    ctx.clear_grasp()
    retreat = float(ctx.config["main"].get("post_release_retreat", DEFAULT_POST_RELEASE_RETREAT_M))
    if retreat > 1e-6:
        retreat_pose = _pose_with_world_offset(np.asarray(ctx.current_ee_pose, dtype=np.float64), np.array([0.0, 0.0, retreat], dtype=np.float64))
        retreat_pose = _clamp_pose_to_workspace(retreat_pose, ctx)
        retreat_actions = _control_points_to_actions(np.asarray(ctx.current_ee_pose, dtype=np.float64), np.asarray([retreat_pose], dtype=np.float64), arm=ctx.active_arm)
        if retreat_actions:
            retreat_records, execution_error = _execute_action_list(
                ctx.adapter,
                retreat_actions,
                execute_motion=execute_motion,
                action_interval_s=action_interval_s,
                emit_progress=emit_progress,
                stage=stage,
            )
            actions.extend(retreat_actions)
            records.extend(retreat_records)
            if not execution_error:
                _update_ctx_pose_from_action(ctx, retreat_actions[-1])
    return actions, records, execution_error



def execute_solver_program(
    *,
    program: Dict[str, Any],
    planning_keypoint_obs: Dict[str, Any],
    adapter,
    execute_motion: bool,
    action_interval_s: float,
    state_dir: str | Path,
    frame_prefix: str,
    emit_progress,
    arm: str = "right",
    grasp_depth_m: float = DEFAULT_REAL_GRASP_DEPTH_M,
) -> Dict[str, Any]:
    config = build_real_solver_config(arm=arm, grasp_depth_m=grasp_depth_m)
    ctx = RealSolverContext(config=config, adapter=adapter, emit_progress=emit_progress)
    forward_axis = _tool_forward_axis(config)
    ctx.set_initial_scene(
        planning_keypoint_obs.get("keypoints_3d", {}),
        planning_keypoint_obs.get("rigid_group_ids", {}),
        arm=arm,
    )

    state_dir = Path(state_dir)
    frames_dir = state_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stage_results: List[Dict[str, Any]] = []
    execution_error = None

    for stage_info in program.get("stages", []):
        stage = int(stage_info.get("stage", 0) or 0)
        subgoal_constraints, path_constraints = _load_stage_constraints(program.get("program_dir", ""), stage_info, ctx.grasped_keypoints)
        subgoal_constraints, path_constraints, constraint_debug = _sanitize_stage_constraints(stage_info, subgoal_constraints, path_constraints)
        is_grasp_stage = int(stage_info.get("grasp_keypoint", -1)) >= 0
        is_release_stage = int(stage_info.get("release_keypoint", -1)) >= 0

        full_keypoints = ctx.full_keypoints()
        movable_mask = ctx.movable_mask()
        collision_points = ctx.current_collision_points()
        subgoal_pose, subgoal_debug = ctx.subgoal_solver.solve(
            ctx.current_ee_pose,
            full_keypoints,
            movable_mask,
            subgoal_constraints,
            path_constraints,
            ctx.sdf_voxels,
            collision_points,
            is_grasp_stage,
            None,
            from_scratch=True,
        )
        if is_grasp_stage:
            subgoal_homo = T.pose2mat([subgoal_pose[:3], subgoal_pose[3:]])
            subgoal_pose = np.asarray(subgoal_pose, dtype=np.float64).copy()
            subgoal_pose[:3] += subgoal_homo[:3, :3] @ (-float(grasp_depth_m) / 2.0 * forward_axis)

        path_control_points, path_debug = ctx.path_solver.solve(
            ctx.current_ee_pose,
            subgoal_pose,
            full_keypoints,
            movable_mask,
            path_constraints,
            ctx.sdf_voxels,
            collision_points,
            None,
            from_scratch=True,
        )
        if is_release_stage:
            move_actions = _make_transport_actions(ctx, target_pose=np.asarray(subgoal_pose, dtype=np.float64), arm=ctx.active_arm)
        else:
            move_actions = _control_points_to_actions(ctx.current_ee_pose, path_control_points, arm=ctx.active_arm)
        plan_payload = {
            "actions": move_actions,
            "notes": "solver-based SE(3) path from subgoal/path solver",
            "stage_goal_summary": f"solver stage {stage}",
            "solver_debug": {
                "subgoal": _jsonable(subgoal_debug),
                "path": _jsonable(path_debug),
                "constraint_debug": constraint_debug,
            },
        }
        plan_path = frames_dir / f"{frame_prefix}_stage{stage}_attempt1.stage_plan.txt"

        execution_records, stage_error = _execute_action_list(
            adapter,
            move_actions,
            execute_motion=execute_motion,
            action_interval_s=action_interval_s,
            emit_progress=emit_progress,
            stage=stage,
        )
        if not stage_error and move_actions:
            _update_ctx_pose_from_action(ctx, move_actions[-1])

        routine_actions: List[Dict[str, Any]] = []
        routine_records: List[Dict[str, Any]] = []
        if not stage_error and is_grasp_stage:
            routine_actions, routine_records, stage_error = _execute_grasp_routine(
                ctx,
                grasp_keypoint=int(stage_info.get("grasp_keypoint", -1)),
                execute_motion=execute_motion,
                action_interval_s=action_interval_s,
                emit_progress=emit_progress,
                stage=stage,
            )
            if not stage_error:
                lift_actions, lift_records, stage_error = _execute_post_grasp_lift(
                    ctx,
                    execute_motion=execute_motion,
                    action_interval_s=action_interval_s,
                    emit_progress=emit_progress,
                    stage=stage,
                )
                routine_actions.extend(lift_actions)
                routine_records.extend(lift_records)
        if not stage_error and is_release_stage:
            release_actions, release_records, stage_error = _execute_release_routine(
                ctx,
                execute_motion=execute_motion,
                action_interval_s=action_interval_s,
                emit_progress=emit_progress,
                stage=stage,
            )
            routine_actions.extend(release_actions)
            routine_records.extend(release_records)

        full_actions = move_actions + routine_actions
        full_records = execution_records + routine_records
        plan_payload["actions"] = full_actions
        if is_grasp_stage:
            plan_payload["notes"] += "; appended explicit grasp-and-lift routine"
        elif is_release_stage:
            plan_payload["notes"] += "; appended explicit hover-descend-release-retreat routine"
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if stage_error and execution_error is None:
            execution_error = f"stage {stage} failed: {stage_error}"

        stage_results.append(
            {
                "stage": stage,
                "frame_path": "",
                "depth_path": "",
                "overlay_path": planning_keypoint_obs.get("overlay_path", ""),
                "keypoint_obs": planning_keypoint_obs,
                "object_schema": planning_keypoint_obs.get("schema", []),
                "capture_info": {},
                "instruction": "",
                "stage_constraints": {
                    "subgoal_constraints_path": stage_info.get("subgoal_constraints_path", ""),
                    "path_constraints_path": stage_info.get("path_constraints_path", ""),
                    "grasp_keypoint": int(stage_info.get("grasp_keypoint", -1)),
                    "release_keypoint": int(stage_info.get("release_keypoint", -1)),
                    "solver_debug": {
                        "subgoal": _jsonable(subgoal_debug),
                        "path": _jsonable(path_debug),
                        "constraint_debug": constraint_debug,
                    },
                    "movable_mask": movable_mask.tolist(),
                    "collision_point_count": int(collision_points.shape[0]),
                    "attached_group": None if ctx.attached_group is None else int(ctx.attached_group),
                },
                "grasp_state": {
                    "grasped_keypoints": sorted(int(v) for v in ctx.grasped_keypoints),
                    "attached_group": None if ctx.attached_group is None else int(ctx.attached_group),
                },
                "constraint_eval": {},
                "monitor_result": {},
                "recovery_result": {},
                "plan_actions": full_actions,
                "plan_raw_output_path": str(plan_path),
                "execution_records": full_records,
                "execution_error": stage_error,
            }
        )
        if stage_error:
            break

    return {
        "stage_results": stage_results,
        "execution_error": execution_error,
        "config": _jsonable(config),
        "current_ee_pose": _jsonable(ctx.current_ee_pose),
    }

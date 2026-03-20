import json
import os
from pathlib import Path

import cv2
import numpy as np

from vlm_client import ask_image_question


def read_string(value, fallback=""):
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return fallback


def sanitize_json_from_text(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_object_from_text(text):
    cleaned = sanitize_json_from_text(text)
    if not cleaned:
        return {}
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def load_pen_rekep_program(repo_dir):
    repo_dir = Path(repo_dir)
    program_dir = repo_dir / "vlm_query" / "pen"
    metadata_path = program_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    stages = []
    num_stages = int(metadata.get("num_stages", 0) or 0)
    for stage_idx in range(1, num_stages + 1):
        subgoal_path = program_dir / f"stage{stage_idx}_subgoal_constraints.txt"
        path_path = program_dir / f"stage{stage_idx}_path_constraints.txt"
        stages.append(
            {
                "stage": stage_idx,
                "subgoal_constraints_path": str(subgoal_path),
                "path_constraints_path": str(path_path),
                "subgoal_constraints": subgoal_path.read_text(encoding="utf-8") if subgoal_path.exists() else "",
                "path_constraints": path_path.read_text(encoding="utf-8") if path_path.exists() else "",
                "grasp_keypoint": (metadata.get("grasp_keypoints") or [])[stage_idx - 1] if stage_idx - 1 < len(metadata.get("grasp_keypoints") or []) else -1,
                "release_keypoint": (metadata.get("release_keypoints") or [])[stage_idx - 1] if stage_idx - 1 < len(metadata.get("release_keypoints") or []) else -1,
            }
        )
    return {
        "program_dir": str(program_dir),
        "metadata": metadata,
        "stages": stages,
    }


def _point_from_depth(u, v, depth_image, calibration_summary):
    color = (calibration_summary or {}).get("color_intrinsic") or {}
    transform = (calibration_summary or {}).get("T_base_camera") or None
    if not color or transform is None:
        raise RuntimeError("camera calibration is incomplete; missing intrinsics or T_base_camera")
    fx = float(color["fx"])
    fy = float(color["fy"])
    cx = float(color["cx"])
    cy = float(color["cy"])
    h, w = depth_image.shape[:2]
    radius = 4
    u0 = max(0, int(round(u)) - radius)
    u1 = min(w, int(round(u)) + radius + 1)
    v0 = max(0, int(round(v)) - radius)
    v1 = min(h, int(round(v)) + radius + 1)
    patch = depth_image[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
    if valid.size == 0:
        raise RuntimeError(f"no valid depth around pixel ({u}, {v})")
    z = float(np.median(valid))
    x = (float(u) - cx) / fx * z
    y = (float(v) - cy) / fy * z
    point_cam = np.array([x, y, z, 1.0], dtype=float)
    point_base = np.asarray(transform, dtype=float) @ point_cam
    return point_base[:3].tolist(), z


PEN_KEYPOINT_SCHEMA = {
    "keypoints": {
        "0": "pen lower/body endpoint",
        "1": "pen upper/body endpoint suitable for grasp geometry",
        "2": "holder rim point near upper-left",
        "3": "holder rim point near upper-right",
        "4": "holder rim point near lower-right",
        "5": "holder rim point near lower-left",
        "6": "holder opening center or another stable rim point",
    }
}


def localize_pen_keypoints(image_path, depth_image, camera_calibration, model="gpt-5.4", temperature=0.0, max_tokens=1200):
    prompt = (
        "You are localizing ReKep keypoints for a Dobot pen insertion task from one RGB image. "
        "Return strict JSON only.\n"
        "Task: locate 7 image keypoints for the white pen and black pen holder.\n"
        "Schema:\n"
        "{\n"
        '  "visible": true,\n'
        '  "reason": "short note",\n'
        '  "keypoints": {\n'
        '    "0": {"u": 0, "v": 0, "label": "pen lower/body endpoint"},\n'
        '    "1": {"u": 0, "v": 0, "label": "pen upper/body endpoint suitable for grasp geometry"},\n'
        '    "2": {"u": 0, "v": 0, "label": "holder rim point near upper-left"},\n'
        '    "3": {"u": 0, "v": 0, "label": "holder rim point near upper-right"},\n'
        '    "4": {"u": 0, "v": 0, "label": "holder rim point near lower-right"},\n'
        '    "5": {"u": 0, "v": 0, "label": "holder rim point near lower-left"},\n'
        '    "6": {"u": 0, "v": 0, "label": "holder opening center or another stable rim point"}\n'
        "  }\n"
        "}\n"
        "Rules: image origin is top-left; u is x pixel, v is y pixel. "
        "Choose precise visible pixels on object surfaces. If partially occluded, still estimate conservatively. No markdown fences."
    )
    image_bytes = Path(image_path).read_bytes()
    answer, vlm_cfg = ask_image_question(
        image_bytes=image_bytes,
        question=prompt,
        default_model=model,
        system_prompt="You are a precise visual keypoint localizer. Return strict JSON only.",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = parse_json_object_from_text(answer)
    keypoints_2d = parsed.get("keypoints") if isinstance(parsed.get("keypoints"), dict) else {}
    keypoints_3d = {}
    depths = {}
    for key in sorted(PEN_KEYPOINT_SCHEMA["keypoints"].keys(), key=int):
        item = keypoints_2d.get(key) if isinstance(keypoints_2d.get(key), dict) else {}
        if "u" not in item or "v" not in item:
            raise RuntimeError(f"missing pixel for keypoint {key} in VLM localization output")
        point_base, depth_m = _point_from_depth(float(item["u"]), float(item["v"]), depth_image, camera_calibration)
        keypoints_3d[key] = point_base
        depths[key] = depth_m
    return {
        "visible": bool(parsed.get("visible", True)),
        "reason": read_string(parsed.get("reason")),
        "raw_output": answer,
        "vlm": vlm_cfg,
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "depths_m": depths,
    }


def build_pen_stage_execution_prompt(*, instruction, stage_info, keypoint_obs, camera_calibration, current_stage, total_stages):
    keypoints_json = json.dumps(keypoint_obs.get("keypoints_3d", {}), ensure_ascii=False)
    pixels_json = json.dumps(keypoint_obs.get("keypoints_2d", {}), ensure_ascii=False)
    calibration_json = json.dumps(camera_calibration or {}, ensure_ascii=False)
    subgoal_constraints = stage_info.get("subgoal_constraints", "")
    path_constraints = stage_info.get("path_constraints", "")
    grasp_keypoint = stage_info.get("grasp_keypoint", -1)
    release_keypoint = stage_info.get("release_keypoint", -1)
    return (
        "You are executing a ReKep-style real robot stage for a Dobot arm. "
        "Plan conservative robot actions that satisfy the current stage constraints using the observed 3D keypoints.\n"
        "Return strict JSON only with schema:\n"
        "{\n"
        '  "actions": [\n'
        '    {"type":"movej","joints":[j1,j2,j3,j4,j5,j6]},\n'
        '    {"type":"movel","pose":[x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg]},\n'
        '    {"type":"open_gripper"},\n'
        '    {"type":"close_gripper"},\n'
        '    {"type":"wait","seconds":0.5}\n'
        "  ],\n"
        '  "notes": "short note",\n'
        '  "stage_goal_summary": "one sentence"\n'
        "}\n"
        "Rules:\n"
        "- Use the ReKep stage constraints as the primary task specification.\n"
        "- Favor short conservative sequences (<= 8 actions).\n"
        "- Use movel poses in Dobot units: xyz in mm in robot base frame, rxyz in degrees.\n"
        "- Use open_gripper / close_gripper only when needed for this stage.\n"
        "- If stage is a grasp stage, close gripper near the end.\n"
        "- If stage releases the object, open gripper near the end.\n"
        "- Avoid unsafe long motions; approach above target before descending when relevant.\n"
        f"Task instruction: {instruction}\n"
        f"Current stage: {current_stage}/{total_stages}\n"
        f"grasp_keypoint: {grasp_keypoint}\n"
        f"release_keypoint: {release_keypoint}\n"
        f"Observed keypoints in robot base frame (meters): {keypoints_json}\n"
        f"Observed keypoints in image pixels: {pixels_json}\n"
        f"Camera calibration summary: {calibration_json}\n"
        "Current stage subgoal constraints (Python):\n"
        f"{subgoal_constraints}\n"
        "Current stage path constraints (Python):\n"
        f"{path_constraints}\n"
    )


def draw_keypoints_overlay(image_path, keypoint_obs, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return ""
    keypoints_2d = keypoint_obs.get("keypoints_2d", {}) if isinstance(keypoint_obs, dict) else {}
    for key, item in keypoints_2d.items():
        if not isinstance(item, dict):
            continue
        u = int(round(float(item.get("u", 0))))
        v = int(round(float(item.get("v", 0))))
        cv2.circle(image, (u, v), 6, (0, 255, 0), -1)
        cv2.putText(image, str(key), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(output_path), image)
    return str(output_path)

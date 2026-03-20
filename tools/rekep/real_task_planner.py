import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from vlm_client import ask_image_question

_SAM_PREDICTOR_CACHE = {}
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SAM_CANDIDATES = (
    ("vit_h", _REPO_ROOT / ".downloads/models/sam/sam_vit_h_4b8939.pth"),
    ("vit_l", _REPO_ROOT / ".downloads/models/sam/sam_vit_l_0b3195.pth"),
    ("vit_b", _REPO_ROOT / ".downloads/models/sam/sam_vit_b_01ec64.pth"),
)


_KEYPOINT_VLM_ENV_MAP = {
    "REKEP_KEYPOINT_VLM_API_KEY": "REKEP_VLM_API_KEY",
    "REKEP_KEYPOINT_VLM_API_KEY_ENV": "REKEP_VLM_API_KEY_ENV",
    "REKEP_KEYPOINT_VLM_BASE_URL": "REKEP_VLM_BASE_URL",
    "REKEP_KEYPOINT_VLM_MODEL": "REKEP_VLM_MODEL",
    "REKEP_KEYPOINT_VLM_MAX_RETRIES": "REKEP_VLM_MAX_RETRIES",
    "REKEP_KEYPOINT_VLM_RETRY_BACKOFF_S": "REKEP_VLM_RETRY_BACKOFF_S",
}


def _read_nonempty_env(name):
    value = os.environ.get(name)
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text if text else ""


def _ask_image_question_for_keypoint_stage(**kwargs):
    overrides = {}
    for source_env, target_env in _KEYPOINT_VLM_ENV_MAP.items():
        value = _read_nonempty_env(source_env)
        if value:
            overrides[target_env] = value
    if not overrides:
        return ask_image_question(**kwargs)

    backup = {target_env: os.environ.get(target_env) for target_env in _KEYPOINT_VLM_ENV_MAP.values()}
    try:
        for target_env, value in overrides.items():
            os.environ[target_env] = value
        return ask_image_question(**kwargs)
    finally:
        for target_env, old_value in backup.items():
            if old_value is None:
                os.environ.pop(target_env, None)
            else:
                os.environ[target_env] = old_value


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


def _ascii_label(value, fallback):
    raw = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", raw)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    if not cleaned:
        return str(fallback)
    return cleaned[:64]


def _normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _contains_any(text, tokens):
    return any(token in text for token in tokens)


def _keypoint_compaction_score(item):
    label = _normalize_text(item.get("label", ""))
    obj = _normalize_text(item.get("object", ""))
    purpose = _normalize_text(item.get("purpose", ""))
    text = f"{label} {obj} {purpose}".strip()
    score = 0
    if _contains_any(text, ("center", "centroid", "middle", "mid", "core", "body", "torso")):
        score += 60
    if _contains_any(text, ("grasp", "pick", "pickup", "grip", "hold")):
        score += 35
    if _contains_any(text, ("target", "place", "drop", "inside", "insert", "opening")):
        score += 20
    if _contains_any(text, ("rim", "edge", "corner", "boundary", "tip", "stem", "end", "left", "right", "front", "back")):
        score -= 12
    if _contains_any(text, ("table", "mat", "desk", "floor", "background", "cloth", "support")):
        score -= 18
    return score


def _compact_task_keypoint_schema(keypoints, instruction):
    if not isinstance(keypoints, list):
        return []

    robot_tokens = ("gripper", "tcp", "end effector", "end_effector", "robot arm", "robot")
    orientation_tokens = (
        "insert",
        "upright",
        "align",
        "rotate",
        "reorient",
        "tilt",
        "pour",
        "angle",
        "orientation",
        "插",
        "对齐",
        "旋",
        "竖",
        "倾斜",
    )
    instruction_text = str(instruction or "").lower()
    max_points = 4 if _contains_any(instruction_text, orientation_tokens) else 2

    grouped = {}
    order = []
    for item in keypoints:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        obj = str(item.get("object", ""))
        purpose = str(item.get("purpose", ""))
        text = _normalize_text(f"{label} {obj} {purpose}")
        if _contains_any(text, robot_tokens):
            continue
        group_key = _normalize_text(obj) or f"id_{item.get('id', len(order))}"
        grouped.setdefault(group_key, []).append(item)
        if group_key not in order:
            order.append(group_key)

    selected = []
    for group_key in order:
        candidates = grouped.get(group_key, [])
        if not candidates:
            continue
        best = max(candidates, key=lambda it: (_keypoint_compaction_score(it), -int(it.get("id", 10**9))))
        selected.append(best)

    selected.sort(key=lambda it: (_keypoint_compaction_score(it), -int(it.get("id", 10**9))), reverse=True)
    if max_points > 0:
        selected = selected[:max_points]
    selected.sort(key=lambda it: int(it.get("id", 10**9)))

    reindexed = []
    for idx, item in enumerate(selected):
        reindexed.append(
            {
                "id": int(idx),
                "label": _ascii_label(item.get("label"), f"keypoint_{idx}"),
                "object": _ascii_label(item.get("object"), "object"),
                "purpose": _ascii_label(item.get("purpose"), "task_relevant"),
            }
        )
    return reindexed


def infer_task_keypoint_schema(image_path, instruction, model="gpt-5.4", temperature=0.0, max_tokens=1400):
    prompt = (
        "You are designing a compact ReKep keypoint schema for a robot manipulation task from one RGB image. "
        "Return strict JSON only.\n"
        "Schema:\n"
        "{\n"
        '  "task_summary": "short summary",\n'
        '  "objects": ["obj1", "obj2"],\n'
        '  "keypoints": [\n'
        '    {"id": 0, "label": "short label", "object": "object name", "purpose": "why this point matters"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Keep keypoints minimal and manipulation-relevant only.\n"
        "- For simple pick-and-place, prefer exactly 2 keypoints: grasp center on manipulated object and place center on target object/container.\n"
        "- For each object, prefer ONE center-like keypoint (center/mid/body), avoid extra rim/edge/tip unless orientation control is explicitly required.\n"
        "- Do NOT include unrelated points: robot gripper/tcp, table/mat/background/support references unless absolutely required by instruction.\n"
        "- Prefer semantically stable physical points visible in the image.\n"
        "- IDs must be consecutive starting at 0.\n"
        "- Use English-only ASCII for label/object/purpose fields.\n"
        f"Task instruction: {instruction}"
    )
    image_bytes = Path(image_path).read_bytes()
    answer, vlm_cfg = _ask_image_question_for_keypoint_stage(
        image_bytes=image_bytes,
        question=prompt,
        default_model=model,
        system_prompt="You design robot keypoint schemas. Return strict JSON only.",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = parse_json_object_from_text(answer)
    keypoints = parsed.get("keypoints") if isinstance(parsed.get("keypoints"), list) else []
    normalized = []
    for idx, item in enumerate(keypoints):
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "id": int(item.get("id", idx)),
                "label": _ascii_label(item.get("label"), f"keypoint_{idx}"),
                "object": _ascii_label(item.get("object"), "object"),
                "purpose": _ascii_label(item.get("purpose"), "task_relevant"),
            }
        )
    normalized = _compact_task_keypoint_schema(normalized, instruction)
    return {
        "task_summary": parsed.get("task_summary", ""),
        "objects": parsed.get("objects", []) if isinstance(parsed.get("objects"), list) else [],
        "keypoints": normalized,
        "raw_output": answer,
        "vlm": vlm_cfg,
    }


def _point_from_depth(u, v, depth_image, camera_calibration):
    color = (camera_calibration or {}).get("color_intrinsic") or {}
    transform = (camera_calibration or {}).get("T_base_camera") or None
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


def _read_int_env(name, default, lower, upper):
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except Exception:
        value = int(default)
    return max(int(lower), min(int(upper), value))


def _read_float_env(name, default, lower, upper):
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except Exception:
        value = float(default)
    return max(float(lower), min(float(upper), value))


def _read_bool_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _resolve_default_sam_asset():
    for model_type, checkpoint_path in _DEFAULT_SAM_CANDIDATES:
        if checkpoint_path.exists():
            return model_type, checkpoint_path
    return "", None


def _infer_sam_model_type(checkpoint_path, fallback="vit_b"):
    name = Path(checkpoint_path).name.lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"
    return str(fallback or "vit_b")


def _parse_roi_xyxy(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    parts = re.split(r"[\s,;]+", text)
    if len(parts) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(p))) for p in parts]
    except Exception:
        return None
    return [x0, y0, x1, y1]


def _sanitize_roi_xyxy(roi, width, height):
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in roi]
    except Exception:
        return None
    x0 = max(0, min(int(width), x0))
    x1 = max(0, min(int(width), x1))
    y0 = max(0, min(int(height), y0))
    y1 = max(0, min(int(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) < 64 or (y1 - y0) < 64:
        return None
    return [x0, y0, x1, y1]


def _has_valid_depth_near(u, v, depth_image, radius=4):
    h, w = depth_image.shape[:2]
    u0 = max(0, int(round(u)) - int(radius))
    u1 = min(w, int(round(u)) + int(radius) + 1)
    v0 = max(0, int(round(v)) - int(radius))
    v1 = min(h, int(round(v)) + int(radius) + 1)
    patch = depth_image[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
    return bool(valid.size > 0)


def _compute_depth_focus_roi(
    depth_image,
    *,
    min_depth_m=0.08,
    max_depth_m=2.2,
    min_area_ratio=0.08,
    margin_ratio=0.10,
):
    h, w = depth_image.shape[:2]
    valid = np.isfinite(depth_image) & (depth_image > float(min_depth_m)) & (depth_image < float(max_depth_m))
    valid_u8 = (valid.astype(np.uint8) * 255)
    if int(np.count_nonzero(valid_u8)) < int(0.02 * h * w):
        return None

    kernel = np.ones((7, 7), dtype=np.uint8)
    valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((valid_u8 > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    area = int(stats[best, cv2.CC_STAT_AREA])
    area_ratio = float(area) / float(max(1, h * w))
    if area_ratio < float(min_area_ratio):
        return None

    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    bw = int(stats[best, cv2.CC_STAT_WIDTH])
    bh = int(stats[best, cv2.CC_STAT_HEIGHT])
    mx = int(round(float(bw) * float(margin_ratio)))
    my = int(round(float(bh) * float(margin_ratio)))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w, x + bw + mx)
    y1 = min(h, y + bh + my)
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    return (int(x0), int(y0), int(x1), int(y1), float(area_ratio))


def _clamp_uv(u, v, width, height):
    uu = max(0, min(int(width) - 1, int(round(float(u)))))
    vv = max(0, min(int(height) - 1, int(round(float(v)))))
    return uu, vv


def _nearest_valid_depth_uv(u, v, depth_image, max_radius=8):
    h, w = depth_image.shape[:2]
    uu, vv = _clamp_uv(u, v, w, h)
    if _has_valid_depth_near(uu, vv, depth_image, radius=0):
        return uu, vv
    for radius in range(1, int(max_radius) + 1):
        u0 = max(0, uu - radius)
        u1 = min(w, uu + radius + 1)
        v0 = max(0, vv - radius)
        v1 = min(h, vv + radius + 1)
        patch = depth_image[v0:v1, u0:u1]
        valid = np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)
        if not np.any(valid):
            continue
        ys, xs = np.where(valid)
        xs = xs + u0
        ys = ys + v0
        d2 = (xs - uu) ** 2 + (ys - vv) ** 2
        best = int(np.argmin(d2))
        return int(xs[best]), int(ys[best])
    return None


def _extract_main_component(mask_bool, seed_xy):
    if not isinstance(mask_bool, np.ndarray) or mask_bool.ndim != 2:
        return None
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    if int(np.count_nonzero(mask_u8)) == 0:
        return None
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_bool.astype(bool)
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    sy = max(0, min(labels.shape[0] - 1, sy))
    sx = max(0, min(labels.shape[1] - 1, sx))
    seed_label = int(labels[sy, sx])
    if seed_label > 0 and int(stats[seed_label, cv2.CC_STAT_AREA]) >= 16:
        return labels == seed_label
    best = -1
    best_score = -1.0
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 16:
            continue
        cx, cy = centroids[label]
        d2 = float((cx - sx) ** 2 + (cy - sy) ** 2)
        score = area / (1.0 + 0.02 * d2)
        if score > best_score:
            best_score = score
            best = label
    if best <= 0:
        return None
    return labels == best


def _semantic_anchor_kind(item):
    label = _normalize_text(item.get("label", ""))
    purpose = _normalize_text(item.get("purpose", ""))
    text = f"{label} {purpose}"
    if _contains_any(text, ("center", "centroid", "middle", "mid", "body", "torso", "core", "中心", "中部")):
        return "center"
    if _contains_any(text, ("tip", "stem", "end", "top", "bottom", "upper", "lower", "尖", "端", "顶部", "底部", "根")):
        return "endpoint"
    if _contains_any(text, ("rim", "edge", "boundary", "contour", "边缘", "轮廓")):
        return "boundary"
    return "nearest"


def _mask_centroid_uv(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _mask_endpoint_uv(mask_bool, coarse_uv):
    ys, xs = np.where(mask_bool)
    if len(xs) < 8:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = np.mean(pts, axis=0, keepdims=True)
    centered = pts - center
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    if vt.shape[0] == 0:
        return None
    axis = vt[0]
    proj = centered @ axis
    p_lo = pts[int(np.argmin(proj))]
    p_hi = pts[int(np.argmax(proj))]
    cuv = np.asarray([float(coarse_uv[0]), float(coarse_uv[1])], dtype=np.float32)
    d_lo = float(np.linalg.norm(p_lo - cuv))
    d_hi = float(np.linalg.norm(p_hi - cuv))
    chosen = p_lo if d_lo <= d_hi else p_hi
    return float(chosen[0]), float(chosen[1])


def _mask_boundary_uv(mask_bool, coarse_uv):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pts = np.concatenate([cnt.reshape(-1, 2) for cnt in contours if cnt.size > 0], axis=0)
    if pts.size == 0:
        return None
    cu = float(coarse_uv[0])
    cv = float(coarse_uv[1])
    d2 = (pts[:, 0].astype(np.float32) - cu) ** 2 + (pts[:, 1].astype(np.float32) - cv) ** 2
    best = int(np.argmin(d2))
    return float(pts[best, 0]), float(pts[best, 1])


def _get_sam_predictor():
    default_model_type, default_checkpoint_path = _resolve_default_sam_asset()
    use_sam = _read_bool_env("REKEP_KEYPOINT_FINE_USE_SAM", default_checkpoint_path is not None)
    if not use_sam:
        return None
    checkpoint = str(os.environ.get("REKEP_KEYPOINT_FINE_SAM_CHECKPOINT", "")).strip()
    checkpoint_path = Path(checkpoint) if checkpoint else default_checkpoint_path
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        return None
    model_type = str(os.environ.get("REKEP_KEYPOINT_FINE_SAM_MODEL_TYPE", "")).strip()
    if not model_type:
        model_type = default_model_type or _infer_sam_model_type(checkpoint_path, "vit_b")
    device = str(os.environ.get("REKEP_KEYPOINT_FINE_SAM_DEVICE", "")).strip()
    if not device:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    cache_key = (str(checkpoint_path.resolve()), model_type, device)
    if cache_key in _SAM_PREDICTOR_CACHE:
        return _SAM_PREDICTOR_CACHE[cache_key]
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception:
        return None
    try:
        sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    except Exception:
        return None
    try:
        sam_model.to(device=device)
    except Exception:
        return None
    predictor = SamPredictor(sam_model)
    _SAM_PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def _predict_sam_mask(predictor, seed_uv, image_shape):
    try:
        point_coords = np.asarray([[float(seed_uv[0]), float(seed_uv[1])]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int32)
        masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    except Exception:
        return None
    if masks is None or len(masks) == 0:
        return None
    best = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
    mask = np.asarray(masks[best], dtype=bool)
    if mask.shape[:2] != tuple(image_shape[:2]):
        return None
    area_ratio = float(np.count_nonzero(mask)) / float(max(1, image_shape[0] * image_shape[1]))
    if area_ratio <= 0.0002 or area_ratio >= 0.85:
        return None
    return mask


def _build_local_depth_grabcut_mask(image_bgr, depth_image, seed_uv, window_px, depth_tol_m):
    h, w = image_bgr.shape[:2]
    half = max(24, int(window_px) // 2)
    su, sv = _clamp_uv(seed_uv[0], seed_uv[1], w, h)
    x0 = max(0, su - half)
    x1 = min(w, su + half + 1)
    y0 = max(0, sv - half)
    y1 = min(h, sv + half + 1)
    local_bgr = image_bgr[y0:y1, x0:x1]
    local_depth = depth_image[y0:y1, x0:x1]
    if local_bgr.size == 0 or local_depth.size == 0:
        return None, "none", 0
    lu = su - x0
    lv = sv - y0

    valid = np.isfinite(local_depth) & (local_depth > 0.05) & (local_depth < 3.0)
    if not np.any(valid):
        return None, "none", 0
    sv0 = max(0, lv - 2)
    sv1 = min(local_depth.shape[0], lv + 3)
    su0 = max(0, lu - 2)
    su1 = min(local_depth.shape[1], lu + 3)
    seed_patch = local_depth[sv0:sv1, su0:su1]
    seed_valid = seed_patch[np.isfinite(seed_patch) & (seed_patch > 0.05) & (seed_patch < 3.0)]
    if seed_valid.size > 0:
        z_ref = float(np.median(seed_valid))
    else:
        z_ref = float(np.median(local_depth[valid]))
    depth_mask = valid & (np.abs(local_depth - z_ref) <= float(depth_tol_m))
    if int(np.count_nonzero(depth_mask)) < 24:
        depth_mask = valid & (np.abs(local_depth - z_ref) <= float(depth_tol_m) * 1.8)
    if int(np.count_nonzero(depth_mask)) < 12:
        depth_mask = valid.copy()

    depth_u8 = (depth_mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), dtype=np.uint8)
    depth_u8 = cv2.morphologyEx(depth_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    depth_u8 = cv2.morphologyEx(depth_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    depth_mask = depth_u8 > 0

    gc_mask = np.full(local_depth.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[depth_mask] = cv2.GC_PR_FGD
    cv2.circle(gc_mask, (int(lu), int(lv)), max(5, half // 5), cv2.GC_FGD, -1)
    gc_mask[:2, :] = cv2.GC_BGD
    gc_mask[-2:, :] = cv2.GC_BGD
    gc_mask[:, :2] = cv2.GC_BGD
    gc_mask[:, -2:] = cv2.GC_BGD
    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(local_bgr, gc_mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
        fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        method = "depth_grabcut"
    except Exception:
        fg = depth_mask
        method = "depth_mask"

    component = _extract_main_component(fg.astype(bool), (lu, lv))
    if component is None or int(np.count_nonzero(component)) < 12:
        component = _extract_main_component(depth_mask.astype(bool), (lu, lv))
        method = "depth_mask"
    if component is None or int(np.count_nonzero(component)) < 12:
        return None, "none", 0

    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = component
    return mask, method, int(np.count_nonzero(mask))


def _save_keypoint_refine_overlay(image_bgr, coarse_2d, refined_2d, refine_summary, output_path):
    canvas = image_bgr.copy()
    for key, coarse_item in (coarse_2d or {}).items():
        if not isinstance(coarse_item, dict):
            continue
        refined_item = (refined_2d or {}).get(key)
        if not isinstance(refined_item, dict):
            continue
        cu, cv = int(round(float(coarse_item.get("u", 0)))), int(round(float(coarse_item.get("v", 0))))
        ru, rv = int(round(float(refined_item.get("u", cu)))), int(round(float(refined_item.get("v", cv))))
        cv2.circle(canvas, (cu, cv), 5, (0, 215, 255), -1)
        cv2.circle(canvas, (ru, rv), 5, (0, 255, 0), -1)
        cv2.line(canvas, (cu, cv), (ru, rv), (255, 255, 0), 1)
        method = str(refined_item.get("refine_method", ""))
        cv2.putText(
            canvas,
            f"{key}:{method}",
            (ru + 6, max(14, rv - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
        )
    out_path = Path(output_path)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def _refine_keypoints_with_local_models(image_path, image_bgr, depth_image, keypoint_schema, keypoints_2d):
    if image_bgr is None or not isinstance(keypoints_2d, dict):
        return keypoints_2d, [], ""

    window_px = _read_int_env("REKEP_KEYPOINT_FINE_WINDOW_PX", default=72, lower=32, upper=240)
    depth_tol_m = _read_float_env("REKEP_KEYPOINT_FINE_DEPTH_TOL_M", default=0.08, lower=0.02, upper=0.35)
    max_shift_px = _read_float_env("REKEP_KEYPOINT_FINE_MAX_SHIFT_PX", default=65.0, lower=8.0, upper=240.0)
    save_debug = _read_bool_env("REKEP_KEYPOINT_FINE_SAVE_DEBUG", True)
    h, w = image_bgr.shape[:2]

    refined = {}
    summary = []

    sam_predictor = _get_sam_predictor()
    sam_ready = False
    if sam_predictor is not None:
        try:
            sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            sam_ready = True
        except Exception:
            sam_ready = False

    schema_map = {str(item.get("id")): item for item in keypoint_schema if isinstance(item, dict)}
    for key, coarse_item in keypoints_2d.items():
        if not isinstance(coarse_item, dict):
            continue
        cu = float(coarse_item.get("u", 0.0))
        cv = float(coarse_item.get("v", 0.0))
        key_meta = schema_map.get(str(key), {})
        seed = _nearest_valid_depth_uv(cu, cv, depth_image, max_radius=max(6, int(window_px // 3)))
        if seed is None:
            su, sv = _clamp_uv(cu, cv, w, h)
        else:
            su, sv = int(seed[0]), int(seed[1])

        mask = None
        mask_method = "none"
        mask_area = 0
        if sam_ready:
            sam_mask = _predict_sam_mask(sam_predictor, (su, sv), image_bgr.shape)
            if sam_mask is not None:
                comp = _extract_main_component(sam_mask, (su, sv))
                if comp is not None and int(np.count_nonzero(comp)) >= 12:
                    mask = comp
                    mask_method = "sam_point"
                    mask_area = int(np.count_nonzero(mask))

        if mask is None:
            mask, mask_method, mask_area = _build_local_depth_grabcut_mask(
                image_bgr,
                depth_image,
                (su, sv),
                window_px=window_px,
                depth_tol_m=depth_tol_m,
            )

        kind = _semantic_anchor_kind(key_meta)
        if mask is not None:
            if kind == "center":
                candidate = _mask_centroid_uv(mask)
            elif kind == "endpoint":
                candidate = _mask_endpoint_uv(mask, (cu, cv))
            elif kind == "boundary":
                candidate = _mask_boundary_uv(mask, (cu, cv))
            else:
                candidate = None
            if candidate is None:
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    d2 = (xs.astype(np.float32) - float(cu)) ** 2 + (ys.astype(np.float32) - float(cv)) ** 2
                    best = int(np.argmin(d2))
                    candidate = (float(xs[best]), float(ys[best]))
        else:
            candidate = (float(su), float(sv))

        candidate_seed = _nearest_valid_depth_uv(candidate[0], candidate[1], depth_image, max_radius=max(4, int(window_px // 4)))
        if candidate_seed is not None:
            ru, rv = float(candidate_seed[0]), float(candidate_seed[1])
            depth_snap = True
        else:
            ru, rv = float(candidate[0]), float(candidate[1])
            depth_snap = False

        shift = float(np.hypot(ru - cu, rv - cv))
        if shift > float(max_shift_px) and _has_valid_depth_near(cu, cv, depth_image, radius=3):
            ru, rv = float(cu), float(cv)
            shift = 0.0
            final_method = "coarse_kept_large_shift"
        else:
            final_method = f"{mask_method}:{kind}"

        refined[str(key)] = {
            **coarse_item,
            "u": float(max(0.0, min(float(w - 1), ru))),
            "v": float(max(0.0, min(float(h - 1), rv))),
            "refine_method": final_method,
            "refine_shift_px": shift,
            "refine_mask_area_px": int(mask_area),
            "refine_depth_snap": bool(depth_snap),
        }
        summary.append(
            {
                "id": str(key),
                "label": str(coarse_item.get("label", key)),
                "coarse_uv": [float(cu), float(cv)],
                "refined_uv": [float(refined[str(key)]["u"]), float(refined[str(key)]["v"])],
                "shift_px": float(shift),
                "mask_method": str(mask_method),
                "anchor_kind": str(kind),
                "mask_area_px": int(mask_area),
                "depth_snap": bool(depth_snap),
                "used_sam": bool(mask_method == "sam_point"),
            }
        )

    debug_path = ""
    if save_debug:
        try:
            debug_path = _save_keypoint_refine_overlay(
                image_bgr,
                keypoints_2d,
                refined,
                summary,
                Path(image_path).with_name(f"{Path(image_path).stem}.keypoints_refined.png"),
            )
        except Exception:
            debug_path = ""
    return refined, summary, debug_path


def _build_schema_localization_prompt(keypoint_lines, prior_issues=None, crop_context=""):
    strict_block = ""
    if prior_issues:
        joined = "\n".join(f"- {str(issue)}" for issue in prior_issues)
        strict_block = (
            "Previous localization had quality issues. Fix all of these in this retry:\n"
            f"{joined}\n"
        )
    crop_block = f"{crop_context}\n" if crop_context else ""
    return (
        "You are localizing robot manipulation keypoints from one RGB image. Return strict JSON only.\n"
        "Schema:\n"
        "{\n"
        '  "visible": true,\n'
        '  "reason": "short note",\n'
        '  "keypoints": {\n'
        '    "0": {"u": 0, "v": 0, "label": "...", "confidence": 0.0}\n'
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Output one pixel for every requested keypoint id.\n"
        "- Place the pixel ON the target object surface, not nearby background/tablecloth/shadow.\n"
        "- If keypoints describe stem/body/tip (or top/middle/lower), keep them on the same object and spatially distinct.\n"
        "- Confidence is a float in [0,1] for each keypoint.\n"
        "- No markdown.\n"
        "Use English-only ASCII for label text.\n"
        f"{crop_block}"
        f"{strict_block}"
        "Requested keypoints:\n"
        + "\n".join(keypoint_lines)
    )


def _normalize_schema_keypoints_from_parsed(parsed, keypoint_schema):
    keypoints_2d = parsed.get("keypoints") if isinstance(parsed.get("keypoints"), dict) else {}
    normalized = {}
    for item in keypoint_schema:
        key = str(item["id"])
        kp = keypoints_2d.get(key) if isinstance(keypoints_2d.get(key), dict) else {}
        if "u" not in kp or "v" not in kp:
            raise RuntimeError(f"missing pixel for keypoint {key}")
        try:
            u = float(kp.get("u"))
            v = float(kp.get("v"))
        except Exception as exc:
            raise RuntimeError(f"invalid pixel for keypoint {key}: {kp}") from exc
        confidence = kp.get("confidence", kp.get("score", 0.75))
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.75
        confidence = max(0.0, min(1.0, confidence))
        normalized[key] = {
            "u": u,
            "v": v,
            "label": _ascii_label(item.get("label"), f"keypoint_{key}"),
            "confidence": confidence,
        }
    return normalized


def _localization_quality_warnings(keypoint_schema, keypoints_2d, spread_px_map):
    warnings = []
    object_to_ids = {}
    for item in keypoint_schema:
        if not isinstance(item, dict):
            continue
        obj = _ascii_label(item.get("object", "object"), "object").lower()
        object_to_ids.setdefault(obj, []).append(str(item["id"]))

    for obj, ids in object_to_ids.items():
        if len(ids) < 2:
            continue
        points = []
        for kid in ids:
            kp = keypoints_2d.get(kid)
            if isinstance(kp, dict):
                points.append(np.asarray([float(kp["u"]), float(kp["v"])], dtype=float))
        if len(points) < 2:
            continue
        dists = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dists.append(float(np.linalg.norm(points[i] - points[j])))
        if dists and min(dists) < 6.0:
            warnings.append(f"object {obj} has nearly-overlapping keypoints (min_dist_px={min(dists):.1f})")

    for key, spread in spread_px_map.items():
        if float(spread) > 18.0:
            warnings.append(f"keypoint {key} has high multi-pass spread ({float(spread):.1f}px)")
    return warnings


def _save_vlm_raw_overlay(image_bgr, parsed, keypoint_schema, output_path):
    if image_bgr is None:
        return ""
    canvas = image_bgr.copy()
    schema_labels = {}
    for item in keypoint_schema:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id"))
        schema_labels[sid] = _ascii_label(item.get("label"), f"keypoint_{sid}")
    keypoints = parsed.get("keypoints") if isinstance(parsed.get("keypoints"), dict) else {}
    drawn = 0
    for key, kp in keypoints.items():
        if not isinstance(kp, dict):
            continue
        try:
            u = int(round(float(kp.get("u"))))
            v = int(round(float(kp.get("v"))))
        except Exception:
            continue
        sid = str(key)
        label = _ascii_label(kp.get("label"), schema_labels.get(sid, f"keypoint_{sid}"))
        conf = kp.get("confidence", kp.get("score", ""))
        conf_text = ""
        try:
            conf_text = f" ({float(conf):.2f})"
        except Exception:
            conf_text = ""
        cv2.circle(canvas, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"{sid}:{label}{conf_text}",
            (u + 8, max(14, v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
        drawn += 1
    if drawn == 0:
        cv2.putText(
            canvas,
            "No valid VLM keypoints parsed",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
        )
    out_path = Path(output_path)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def localize_schema_keypoints(image_path, depth_image, camera_calibration, keypoint_schema, model="gpt-5.4", temperature=0.0, max_tokens=1600):
    keypoint_lines = []
    for item in keypoint_schema:
        keypoint_lines.append(f'{item["id"]}: {item["label"]} | object={item.get("object","")} | purpose={item.get("purpose","")}')
    num_passes = _read_int_env("REKEP_KEYPOINT_LOCALIZE_PASSES", default=1, lower=1, upper=5)
    confidence_retry_threshold = _read_float_env("REKEP_KEYPOINT_MIN_CONFIDENCE", default=0.65, lower=0.0, upper=1.0)
    auto_crop = _read_bool_env("REKEP_KEYPOINT_AUTO_CROP", True)
    roi_min_area_ratio = _read_float_env("REKEP_KEYPOINT_AUTO_CROP_MIN_AREA_RATIO", default=0.08, lower=0.02, upper=0.60)
    roi_max_area_ratio = _read_float_env("REKEP_KEYPOINT_AUTO_CROP_MAX_AREA_RATIO", default=0.95, lower=0.30, upper=1.00)
    roi_margin_ratio = _read_float_env("REKEP_KEYPOINT_AUTO_CROP_MARGIN", default=0.10, lower=0.00, upper=0.40)
    save_debug_overlays = _read_bool_env("REKEP_KEYPOINT_SAVE_CROP_DEBUG", True)
    fine_refine_enabled = _read_bool_env("REKEP_KEYPOINT_FINE_REFINE", True)
    fixed_roi_raw = os.environ.get("REKEP_KEYPOINT_FIXED_ROI_XYXY", "250,150,600,450")

    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()
    image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    focus_roi = None
    focus_roi_area_ratio = None
    focus_roi_debug_image = ""
    query_image_bytes = image_bytes
    query_image_bgr = image_bgr.copy() if image_bgr is not None else None
    crop_context = ""
    focus_roi_source = ""

    fixed_roi = None
    if image_bgr is not None:
        parsed_fixed = _parse_roi_xyxy(fixed_roi_raw)
        fixed_roi = _sanitize_roi_xyxy(parsed_fixed, image_bgr.shape[1], image_bgr.shape[0]) if parsed_fixed is not None else None

    if fixed_roi is not None and image_bgr is not None:
        x0, y0, x1, y1 = fixed_roi
        cropped_bgr = image_bgr[y0:y1, x0:x1]
        if cropped_bgr.size > 0:
            ok, encoded = cv2.imencode(".png", cropped_bgr)
            if ok:
                query_image_bytes = encoded.tobytes()
                query_image_bgr = cropped_bgr.copy()
                focus_roi = [int(x0), int(y0), int(x1), int(y1)]
                focus_roi_area_ratio = float((x1 - x0) * (y1 - y0)) / float(max(1, image_bgr.shape[0] * image_bgr.shape[1]))
                focus_roi_source = "fixed_roi_env"
                crop_context = (
                    "Input image is cropped from a full frame. "
                    "Return keypoint (u,v) in THIS cropped image coordinates (origin at crop top-left)."
                )
                if save_debug_overlays:
                    focus_roi_debug_path = image_path.with_name(f"{image_path.stem}.focus_roi.png")
                    cv2.imwrite(str(focus_roi_debug_path), query_image_bgr)
                    focus_roi_debug_image = str(focus_roi_debug_path)

    if focus_roi is None and auto_crop:
        roi_info = _compute_depth_focus_roi(
            depth_image,
            min_area_ratio=roi_min_area_ratio,
            margin_ratio=roi_margin_ratio,
        )
        if roi_info is not None:
            x0, y0, x1, y1, area_ratio = roi_info
            focus_roi_area_ratio = float(area_ratio)
            if float(area_ratio) <= roi_max_area_ratio:
                if image_bgr is not None and image_bgr.shape[0] >= y1 and image_bgr.shape[1] >= x1:
                    cropped_bgr = image_bgr[y0:y1, x0:x1]
                    if cropped_bgr.size > 0:
                        ok, encoded = cv2.imencode(".png", cropped_bgr)
                        if ok:
                            query_image_bytes = encoded.tobytes()
                            query_image_bgr = cropped_bgr.copy()
                            focus_roi = [int(x0), int(y0), int(x1), int(y1)]
                            focus_roi_source = "depth_valid_mask"
                            crop_context = (
                                "Input image is cropped from a full frame. "
                                "Return keypoint (u,v) in THIS cropped image coordinates (origin at crop top-left)."
                            )
                            if save_debug_overlays:
                                focus_roi_debug_path = image_path.with_name(f"{image_path.stem}.focus_roi.png")
                                cv2.imwrite(str(focus_roi_debug_path), query_image_bgr)
                                focus_roi_debug_image = str(focus_roi_debug_path)

    attempts = []
    attempt_debug_overlays = []
    prior_issues = []
    vlm_cfg = {}
    for attempt_idx in range(1, num_passes + 1):
        prompt = _build_schema_localization_prompt(
            keypoint_lines,
            prior_issues=prior_issues if attempt_idx > 1 else None,
            crop_context=crop_context,
        )
        answer, vlm_cfg = _ask_image_question_for_keypoint_stage(
            image_bytes=query_image_bytes,
            question=prompt,
            default_model=model,
            system_prompt="You localize robot keypoints precisely. Return strict JSON only.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = parse_json_object_from_text(answer)
        if save_debug_overlays and query_image_bgr is not None:
            if focus_roi is not None:
                debug_overlay_path = image_path.with_name(f"{image_path.stem}.focus_roi_attempt{attempt_idx}.vlm_raw.png")
            else:
                debug_overlay_path = image_path.with_name(f"{image_path.stem}.full_attempt{attempt_idx}.vlm_raw.png")
            saved_overlay = _save_vlm_raw_overlay(
                query_image_bgr,
                parsed,
                keypoint_schema,
                debug_overlay_path,
            )
            if saved_overlay:
                attempt_debug_overlays.append(saved_overlay)
        visible = bool(parsed.get("visible", True))
        reason = str(parsed.get("reason", ""))
        try:
            normalized = _normalize_schema_keypoints_from_parsed(parsed, keypoint_schema)
        except Exception as exc:
            prior_issues = [str(exc)]
            continue
        if focus_roi is not None:
            x0, y0, _, _ = focus_roi
            for key in list(normalized.keys()):
                kp = normalized.get(key)
                if not isinstance(kp, dict):
                    continue
                kp["u"] = float(kp["u"]) + float(x0)
                kp["v"] = float(kp["v"]) + float(y0)
        invalid_depth = []
        for item in keypoint_schema:
            key = str(item["id"])
            kp = normalized.get(key)
            if not isinstance(kp, dict):
                invalid_depth.append(key)
                continue
            if not _has_valid_depth_near(kp["u"], kp["v"], depth_image):
                invalid_depth.append(key)
        if invalid_depth:
            prior_issues = [f"invalid depth near keypoints: {invalid_depth}"]
            continue
        attempts.append(
            {
                "visible": visible,
                "reason": reason,
                "answer": answer,
                "keypoints_2d": normalized,
            }
        )
        min_conf = min(float(v.get("confidence", 0.0)) for v in normalized.values()) if normalized else 0.0
        if min_conf >= confidence_retry_threshold:
            break
        prior_issues = [f"low confidence detected (min={min_conf:.2f}); refine exact on-object pixels"]

    if not attempts:
        raise RuntimeError("failed to localize schema keypoints with valid depth after retries")

    # Aggregate multi-pass observations by median for better spatial stability.
    aggregated_2d = {}
    spread_px_map = {}
    for item in keypoint_schema:
        key = str(item["id"])
        us = []
        vs = []
        confidences = []
        for attempt in attempts:
            kp = attempt["keypoints_2d"].get(key)
            if not isinstance(kp, dict):
                continue
            us.append(float(kp["u"]))
            vs.append(float(kp["v"]))
            confidences.append(float(kp.get("confidence", 0.75)))
        if not us:
            raise RuntimeError(f"no candidate for keypoint {key}")
        u_med = float(np.median(np.asarray(us, dtype=float)))
        v_med = float(np.median(np.asarray(vs, dtype=float)))
        spread = float(np.max(np.sqrt((np.asarray(us) - u_med) ** 2 + (np.asarray(vs) - v_med) ** 2)))
        spread_px_map[key] = spread
        aggregated_2d[key] = {
            "u": u_med,
            "v": v_med,
            "label": _ascii_label(item.get("label"), f"keypoint_{key}"),
            "confidence": float(np.median(np.asarray(confidences, dtype=float))),
        }

    coarse_2d = {str(k): dict(v) for k, v in aggregated_2d.items() if isinstance(v, dict)}
    fine_refine_summary = []
    fine_refine_debug_image = ""
    fine_refine_applied = False
    if fine_refine_enabled and image_bgr is not None and aggregated_2d:
        refined_2d, fine_refine_summary, fine_refine_debug_image = _refine_keypoints_with_local_models(
            image_path=image_path,
            image_bgr=image_bgr,
            depth_image=depth_image,
            keypoint_schema=keypoint_schema,
            keypoints_2d=aggregated_2d,
        )
        if isinstance(refined_2d, dict) and refined_2d:
            aggregated_2d = refined_2d
            fine_refine_applied = True

    keypoints_3d = {}
    depths = {}
    for item in keypoint_schema:
        key = str(item["id"])
        kp = aggregated_2d.get(key) if isinstance(aggregated_2d.get(key), dict) else {}
        point_base, depth_m = _point_from_depth(float(kp["u"]), float(kp["v"]), depth_image, camera_calibration)
        keypoints_3d[key] = point_base
        depths[key] = depth_m

    warnings = _localization_quality_warnings(keypoint_schema, aggregated_2d, spread_px_map)
    moved = [item for item in fine_refine_summary if float(item.get("shift_px", 0.0)) > 0.5]
    if fine_refine_applied:
        warnings.append(f"fine_refine moved {len(moved)}/{len(fine_refine_summary)} keypoints")
    visible_values = [bool(attempt.get("visible", True)) for attempt in attempts]
    reason_values = [str(attempt.get("reason", "")).strip() for attempt in attempts if str(attempt.get("reason", "")).strip()]
    return {
        "visible": bool(all(visible_values)),
        "reason": reason_values[-1] if reason_values else "",
        "raw_output": attempts[-1].get("answer", ""),
        "vlm": vlm_cfg,
        "keypoints_2d": aggregated_2d,
        "keypoints_3d": keypoints_3d,
        "depths_m": depths,
        "schema": keypoint_schema,
        "localization_method": "multi_pass_median_local_refine" if fine_refine_applied else "multi_pass_median",
        "localization_attempt_count": len(attempts),
        "localization_spread_px": spread_px_map,
        "localization_warnings": warnings,
        "focus_roi_xyxy": focus_roi or [],
        "focus_roi_area_ratio": focus_roi_area_ratio,
        "focus_roi_source": focus_roi_source,
        "focus_roi_debug_image": focus_roi_debug_image,
        "vlm_raw_overlay_paths": attempt_debug_overlays,
        "coarse_keypoints_2d": coarse_2d,
        "fine_refine_enabled": fine_refine_enabled,
        "fine_refine_applied": fine_refine_applied,
        "fine_refine_summary": fine_refine_summary,
        "fine_refine_debug_image": fine_refine_debug_image,
    }


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
        label = _ascii_label(item.get("label"), f"keypoint_{key}")
        cv2.circle(image, (u, v), 6, (0, 255, 0), -1)
        cv2.putText(image, f"{key}:{label}", (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(str(output_path), image)
    return str(output_path)


def build_generic_stage_execution_prompt(*, instruction, stage_info, keypoint_obs, camera_calibration, current_stage, total_stages):
    schema_json = json.dumps(keypoint_obs.get("schema", []), ensure_ascii=False)
    keypoints_json = json.dumps(keypoint_obs.get("keypoints_3d", {}), ensure_ascii=False)
    pixels_json = json.dumps(keypoint_obs.get("keypoints_2d", {}), ensure_ascii=False)
    calibration_json = json.dumps(camera_calibration or {}, ensure_ascii=False)
    return (
        "You are executing a real robot manipulation stage under a ReKep-style constraint program.\n"
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
        "- Prefer safe, short, conservative plans (<= 8 actions).\n"
        "- Use observed 3D keypoints as the primary geometric reference.\n"
        "- Use movel pose units in Dobot base frame: xyz mm, rxyz deg.\n"
        "- Respect grasp and release semantics from the stage program.\n"
        f"Task instruction: {instruction}\n"
        f"Current stage: {current_stage}/{total_stages}\n"
        f"grasp_keypoint: {stage_info.get('grasp_keypoint', -1)}\n"
        f"release_keypoint: {stage_info.get('release_keypoint', -1)}\n"
        f"Keypoint schema: {schema_json}\n"
        f"Observed keypoints in base frame (m): {keypoints_json}\n"
        f"Observed keypoints in image pixels: {pixels_json}\n"
        f"Camera calibration summary: {calibration_json}\n"
        f"Current stage subgoal constraints:\n{stage_info.get('subgoal_constraints','')}\n"
        f"Current stage path constraints:\n{stage_info.get('path_constraints','')}\n"
    )

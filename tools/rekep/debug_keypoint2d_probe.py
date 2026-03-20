#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from dobot_bridge import (
    capture_realsense_rgbd,
    capture_realsense_zmq_rgbd,
    parse_realsense_source,
    parse_realsense_zmq_source,
    resolve_real_camera_calibration,
    summarize_camera_calibration,
)
import real_task_planner as task_planner
from real_task_planner import (
    _compute_depth_focus_roi,
    _point_from_depth,
    draw_keypoints_overlay,
    infer_task_keypoint_schema,
    localize_schema_keypoints,
)
from vlm_client import encode_image_bytes, get_vlm_request_config


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SAM_CANDIDATES = (
    ("vit_h", _REPO_ROOT / ".downloads/models/sam/sam_vit_h_4b8939.pth"),
    ("vit_l", _REPO_ROOT / ".downloads/models/sam/sam_vit_l_0b3195.pth"),
    ("vit_b", _REPO_ROOT / ".downloads/models/sam/sam_vit_b_01ec64.pth"),
)


def _resolve_default_sam_asset():
    for model_type, checkpoint_path in _DEFAULT_SAM_CANDIDATES:
        if checkpoint_path.exists():
            return model_type, checkpoint_path
    return "vit_b", None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_schema(schema_path: Path):
    payload = _read_json(schema_path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("keypoints"), list):
        return payload["keypoints"]
    raise RuntimeError(f"unsupported schema file format: {schema_path}")


def _save_rgb_depth(rgb: np.ndarray, depth: np.ndarray, out_dir: Path):
    image_path = out_dir / "input.png"
    depth_path = out_dir / "input.depth.npy"
    if not cv2.imwrite(str(image_path), rgb):
        raise RuntimeError(f"failed to write image: {image_path}")
    np.save(depth_path, depth)
    return image_path, depth_path


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


def _resolve_focus_roi(image_bgr, depth_image, args):
    h, w = image_bgr.shape[:2]
    if not args.disable_fixed_roi:
        fixed = _sanitize_roi_xyxy(_parse_roi_xyxy(args.fixed_roi), w, h)
        if fixed is not None:
            x0, y0, x1, y1 = fixed
            area_ratio = float((x1 - x0) * (y1 - y0)) / float(max(1, h * w))
            return fixed, area_ratio, "fixed_roi_arg"
    if bool(args.auto_crop):
        roi_info = _compute_depth_focus_roi(
            depth_image,
            min_area_ratio=float(args.auto_crop_min_area_ratio),
            margin_ratio=float(args.auto_crop_margin_ratio),
        )
        if roi_info is not None:
            x0, y0, x1, y1, area_ratio = roi_info
            if float(area_ratio) <= float(args.auto_crop_max_area_ratio):
                return [int(x0), int(y0), int(x1), int(y1)], float(area_ratio), "depth_valid_mask"
    return None, None, ""


def _manual_click_keypoints(display_bgr, keypoint_schema, *, offset_xy=(0, 0), window_name="debug_kp2d_manual"):
    ordered = sorted([item for item in keypoint_schema if isinstance(item, dict)], key=lambda x: int(x.get("id", 0)))
    if not ordered:
        raise RuntimeError("manual click requires non-empty keypoint schema")

    state = {
        "idx": 0,
        "points": [],
        "done": False,
        "quit": False,
    }
    canvas = display_bgr.copy()
    ox, oy = int(offset_xy[0]), int(offset_xy[1])

    def _refresh_canvas():
        nonlocal canvas
        canvas = display_bgr.copy()
        for point in state["points"]:
            x = int(point["x"])
            y = int(point["y"])
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f'{point["id"]}:{point["label"]}',
                (x + 8, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        if state["idx"] < len(ordered):
            item = ordered[state["idx"]]
            hint = f'Click {state["idx"] + 1}/{len(ordered)} -> id={item["id"]} label={item.get("label","")}'
            cv2.putText(canvas, hint, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "Keys: u=undo, r=reset, q=quit",
                (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(canvas, "All points selected. Press Enter to confirm.", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def _on_mouse(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["idx"] >= len(ordered):
            return
        item = ordered[state["idx"]]
        state["points"].append(
            {
                "id": int(item["id"]),
                "label": str(item.get("label", f'keypoint_{item["id"]}')),
                "x": int(x),
                "y": int(y),
                "u_full": float(x + ox),
                "v_full": float(y + oy),
            }
        )
        state["idx"] += 1
        _refresh_canvas()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)
    _refresh_canvas()
    while True:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # Enter
            if state["idx"] >= len(ordered):
                state["done"] = True
                break
        elif key == ord("u"):
            if state["points"]:
                state["points"].pop()
                state["idx"] = max(0, state["idx"] - 1)
                _refresh_canvas()
        elif key == ord("r"):
            state["points"] = []
            state["idx"] = 0
            _refresh_canvas()
        elif key in (ord("q"), 27):
            state["quit"] = True
            break
    cv2.destroyWindow(window_name)

    if state["quit"]:
        raise RuntimeError("manual click cancelled by user")
    if not state["done"] or len(state["points"]) != len(ordered):
        raise RuntimeError("manual click did not finish all keypoints")
    return state["points"], canvas


def _manual_localize_schema_keypoints(*, image_path, depth_image, camera_calibration, keypoint_schema, args, out_dir):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    focus_roi, focus_roi_area_ratio, focus_roi_source = _resolve_focus_roi(image_bgr, depth_image, args)
    if focus_roi is not None:
        x0, y0, x1, y1 = focus_roi
        click_bgr = image_bgr[y0:y1, x0:x1].copy()
        offset_xy = (x0, y0)
    else:
        click_bgr = image_bgr.copy()
        offset_xy = (0, 0)

    focus_roi_debug_image = ""
    if focus_roi is not None:
        roi_path = out_dir / "manual.focus_roi.png"
        cv2.imwrite(str(roi_path), click_bgr)
        focus_roi_debug_image = str(roi_path)

    points, click_overlay = _manual_click_keypoints(
        click_bgr,
        keypoint_schema,
        offset_xy=offset_xy,
        window_name=args.manual_window_name,
    )
    raw_overlay_path = out_dir / "manual.click_overlay.png"
    cv2.imwrite(str(raw_overlay_path), click_overlay)

    keypoints_2d = {}
    keypoints_3d = {}
    depths_m = {}
    spread_map = {}
    for point in points:
        key = str(point["id"])
        keypoints_2d[key] = {
            "u": float(point["u_full"]),
            "v": float(point["v_full"]),
            "label": str(point["label"]),
            "confidence": 1.0,
        }
        base_point, depth_m = _point_from_depth(
            float(point["u_full"]),
            float(point["v_full"]),
            depth_image,
            camera_calibration,
        )
        keypoints_3d[key] = base_point
        depths_m[key] = float(depth_m)
        spread_map[key] = 0.0

    return {
        "visible": True,
        "reason": "manual click annotation",
        "raw_output": "manual_click",
        "vlm": {"model": "manual_click", "base_url": "", "api_key_env": ""},
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "depths_m": depths_m,
        "schema": keypoint_schema,
        "localization_method": "manual_click",
        "localization_attempt_count": 1,
        "localization_spread_px": spread_map,
        "localization_warnings": [],
        "focus_roi_xyxy": focus_roi or [],
        "focus_roi_area_ratio": focus_roi_area_ratio,
        "focus_roi_source": focus_roi_source,
        "focus_roi_debug_image": focus_roi_debug_image,
        "vlm_raw_overlay_paths": [str(raw_overlay_path)],
    }


def _capture_rgbd(camera_source: str, warmup_frames: int, timeout_s: float):
    if parse_realsense_zmq_source(camera_source)["enabled"]:
        return capture_realsense_zmq_rgbd(
            camera_source=camera_source,
            warmup_frames=warmup_frames,
            timeout_s=timeout_s,
        )
    if parse_realsense_source(camera_source)["enabled"]:
        return capture_realsense_rgbd(
            camera_source=camera_source,
            warmup_frames=warmup_frames,
            timeout_s=timeout_s,
        )
    raise RuntimeError(
        f"only RealSense sources are supported in this probe, got: {camera_source}"
    )


def _configure_localization_env(args):
    os.environ["REKEP_KEYPOINT_SAVE_CROP_DEBUG"] = "1"
    os.environ["REKEP_KEYPOINT_LOCALIZE_PASSES"] = str(args.localize_passes)
    os.environ["REKEP_KEYPOINT_MIN_CONFIDENCE"] = str(args.min_confidence)
    os.environ["REKEP_KEYPOINT_AUTO_CROP"] = "1" if args.auto_crop else "0"
    os.environ["REKEP_KEYPOINT_AUTO_CROP_MIN_AREA_RATIO"] = str(args.auto_crop_min_area_ratio)
    os.environ["REKEP_KEYPOINT_AUTO_CROP_MAX_AREA_RATIO"] = str(args.auto_crop_max_area_ratio)
    os.environ["REKEP_KEYPOINT_AUTO_CROP_MARGIN"] = str(args.auto_crop_margin_ratio)
    os.environ["REKEP_KEYPOINT_FINE_REFINE"] = "1" if args.fine_refine else "0"
    os.environ["REKEP_KEYPOINT_FINE_WINDOW_PX"] = str(args.fine_window_px)
    os.environ["REKEP_KEYPOINT_FINE_DEPTH_TOL_M"] = str(args.fine_depth_tol_m)
    os.environ["REKEP_KEYPOINT_FINE_MAX_SHIFT_PX"] = str(args.fine_max_shift_px)
    os.environ["REKEP_KEYPOINT_FINE_USE_SAM"] = "1" if args.fine_use_sam else "0"
    os.environ["REKEP_KEYPOINT_FINE_SAVE_DEBUG"] = "1" if args.fine_save_debug else "0"
    if args.fine_sam_model_type:
        os.environ["REKEP_KEYPOINT_FINE_SAM_MODEL_TYPE"] = str(args.fine_sam_model_type)
    if args.fine_sam_device:
        os.environ["REKEP_KEYPOINT_FINE_SAM_DEVICE"] = str(args.fine_sam_device)
    if args.fine_sam_checkpoint:
        os.environ["REKEP_KEYPOINT_FINE_SAM_CHECKPOINT"] = str(args.fine_sam_checkpoint)
    if args.disable_fixed_roi:
        os.environ["REKEP_KEYPOINT_FIXED_ROI_XYXY"] = ""
    elif args.fixed_roi:
        os.environ["REKEP_KEYPOINT_FIXED_ROI_XYXY"] = args.fixed_roi


def _extract_text_from_response_payload(payload):
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_chunks = []
    output_items = payload.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                text_value = content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    text_chunks.append(text_value.strip())
    if text_chunks:
        return "\n".join(text_chunks).strip()

    # Fallback: chat-completions-like payload
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        text = item["text"].strip()
                        if text:
                            parts.append(text)
                if parts:
                    return "\n".join(parts).strip()
    return ""


def _extract_text_from_responses_event(event):
    if not isinstance(event, dict):
        return ""
    event_type = str(event.get("type", ""))
    if event_type.endswith("output_text.delta"):
        delta = event.get("delta")
        return delta if isinstance(delta, str) else ""
    if event_type.endswith("output_text.done"):
        text = event.get("text")
        return text if isinstance(text, str) else ""
    if event_type == "response.completed":
        response_payload = event.get("response")
        return _extract_text_from_response_payload(response_payload)
    # Some providers send plain payload fragments in data lines.
    return _extract_text_from_response_payload(event)


def _request_qwen_responses(
    *,
    image_bytes,
    question,
    default_model=None,
    system_prompt=None,
    temperature=0.0,
    max_tokens=512,
):
    config = get_vlm_request_config(default_model=default_model)
    base_url = config.get("base_url") or "https://api.openai.com/v1"
    url = f"{base_url.rstrip('/')}/responses"

    input_messages = []
    if system_prompt:
        input_messages.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": str(system_prompt)}],
            }
        )
    input_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": str(question)},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encode_image_bytes(image_bytes)}",
                },
            ],
        }
    )

    payload = {
        "model": config["model"],
        "input": input_messages,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_output_tokens"] = int(max_tokens)

    max_retries = 3
    backoff_s = 1.5
    retry_status = {429, 500, 502, 503, 504}
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=300,
                stream=True,
            )
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(backoff_s * (2**attempt))
                continue
            raise RuntimeError(
                f"responses API request failed (model={config['model']}, base_url={base_url}): {exc}"
            ) from exc

        if int(response.status_code) in retry_status and attempt < max_retries:
            response.close()
            time.sleep(backoff_s * (2**attempt))
            continue

        if int(response.status_code) >= 400:
            body = response.text.strip().replace("\n", " ")
            if len(body) > 400:
                body = f"{body[:400]}...(truncated)"
            raise requests.HTTPError(
                f"{response.status_code} error from responses API "
                f"(model={config['model']}, base_url={base_url}): {body}",
                response=response,
            )

        chunks = []
        fallback_events = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = str(raw_line).strip()
            if not line.startswith("data:"):
                continue
            data_text = line[5:].strip()
            if not data_text or data_text == "[DONE]":
                continue
            try:
                event = json.loads(data_text)
            except Exception:
                continue
            fallback_events.append(event)
            text_part = _extract_text_from_responses_event(event)
            if text_part:
                chunks.append(text_part)

        answer = "".join(chunks).strip()
        if not answer and fallback_events:
            # Some providers only return completed payload once.
            for event in reversed(fallback_events):
                answer = _extract_text_from_responses_event(event).strip()
                if answer:
                    break

        if not answer:
            try:
                payload_json = response.json()
                answer = _extract_text_from_response_payload(payload_json)
            except Exception:
                answer = ""

        if not answer:
            raise RuntimeError(
                f"responses API returned no text output "
                f"(model={config['model']}, base_url={base_url})"
            )

        return answer, config

    raise RuntimeError(
        f"responses API failed after retries (model={config['model']}, base_url={base_url})"
    )


def _maybe_patch_qwen_responses(model_name: str):
    model = str(model_name or "").strip().lower()
    if not model.startswith("qwen3-omni-flash"):
        return
    task_planner.ask_image_question = _request_qwen_responses


def _build_parser():
    repo_dir = Path(__file__).resolve().parent
    default_sam_model_type, default_sam_checkpoint = _resolve_default_sam_asset()
    parser = argparse.ArgumentParser(
        description="Debug the front part of real ReKep: capture -> crop -> VLM keypoint2d localization."
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="Task instruction used for schema inference and localization context.",
    )
    parser.add_argument(
        "--camera-source",
        default="realsense_zmq://127.0.0.1:7001/realsense",
        help="RealSense source, supports realsense / realsense:<serial> / realsense_zmq://host:port/topic",
    )
    parser.add_argument("--camera-timeout-s", type=float, default=8.0)
    parser.add_argument("--camera-warmup-frames", type=int, default=6)
    parser.add_argument("--camera-profile", default="global3")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument(
        "--dobot-settings-ini",
        default=str(repo_dir / "real_calibration" / "dobot_settings.ini"),
    )
    parser.add_argument(
        "--camera-extrinsic-script",
        default=str(repo_dir / "real_calibration" / "eval_dobot_v1.py"),
    )
    parser.add_argument(
        "--realsense-calib-dir",
        default=str(repo_dir / "real_calibration" / "realsense_config"),
    )
    parser.add_argument(
        "--image-path",
        default="",
        help="Optional existing RGB image path. If set, capture step is skipped.",
    )
    parser.add_argument(
        "--depth-path",
        default="",
        help="Optional existing depth .npy path (meters). Required when --image-path is set.",
    )
    parser.add_argument(
        "--schema-path",
        default="",
        help="Optional schema json path; if omitted, schema is inferred from image + instruction.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--schema-max-tokens", type=int, default=1400)
    parser.add_argument("--localize-passes", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument(
        "--fine-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stage-2 local keypoint refinement after VLM coarse localization.",
    )
    parser.add_argument(
        "--fine-window-px",
        type=int,
        default=72,
        help="Local refinement window size in pixels.",
    )
    parser.add_argument(
        "--fine-depth-tol-m",
        type=float,
        default=0.08,
        help="Depth tolerance (meters) for local refinement mask construction.",
    )
    parser.add_argument(
        "--fine-max-shift-px",
        type=float,
        default=65.0,
        help="Maximum allowed refine shift from coarse keypoint before fallback.",
    )
    parser.add_argument(
        "--fine-use-sam",
        action=argparse.BooleanOptionalAction,
        default=default_sam_checkpoint is not None,
        help="Use local SAM point-prompt refinement when checkpoint is configured.",
    )
    parser.add_argument(
        "--fine-sam-checkpoint",
        default=str(default_sam_checkpoint) if default_sam_checkpoint is not None else "",
        help="Optional local SAM checkpoint path.",
    )
    parser.add_argument(
        "--fine-sam-model-type",
        default=default_sam_model_type,
        help="SAM model type (vit_b/vit_l/vit_h).",
    )
    parser.add_argument(
        "--fine-sam-device",
        default="",
        help="Optional SAM device override (cpu/cuda).",
    )
    parser.add_argument(
        "--fine-save-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save coarse->refined keypoint debug overlay.",
    )
    parser.add_argument(
        "--manual-kp2d",
        action="store_true",
        help="Use manual mouse clicks for keypoint2d instead of VLM localization.",
    )
    parser.add_argument(
        "--manual-window-name",
        default="debug_kp2d_manual",
        help="OpenCV window name for manual click mode.",
    )
    parser.add_argument("--auto-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-crop-min-area-ratio", type=float, default=0.08)
    parser.add_argument("--auto-crop-max-area-ratio", type=float, default=0.95)
    parser.add_argument("--auto-crop-margin-ratio", type=float, default=0.10)
    parser.add_argument(
        "--fixed-roi",
        default="250,150,600,450",
        help='Fixed ROI "x0,y0,x1,y1". Empty string to keep current env/default.',
    )
    parser.add_argument("--disable-fixed-roi", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: openclaw-runtime/state/rekep/real/debug_kp2d/<timestamp>",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    _maybe_patch_qwen_responses(args.model)
    repo_root = Path(__file__).resolve().parent.parent
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_out = (
        repo_root
        / "openclaw-runtime"
        / "state"
        / "rekep"
        / "real"
        / "debug_kp2d"
        / f"probe_{ts}"
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    _configure_localization_env(args)

    capture_info = {}
    if args.image_path:
        if not args.depth_path:
            raise RuntimeError("--depth-path is required when --image-path is provided")
        image_path = Path(args.image_path).expanduser().resolve()
        depth_path = Path(args.depth_path).expanduser().resolve()
        rgb = cv2.imread(str(image_path))
        if rgb is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        depth = np.load(depth_path)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        image_copy_path = out_dir / "input.png"
        depth_copy_path = out_dir / "input.depth.npy"
        cv2.imwrite(str(image_copy_path), rgb)
        np.save(depth_copy_path, depth)
        image_path = image_copy_path
        depth_path = depth_copy_path
        capture_info = {"source": "file", "image_path": str(args.image_path), "depth_path": str(args.depth_path)}
    else:
        rgb, depth, capture_info = _capture_rgbd(
            camera_source=args.camera_source,
            warmup_frames=max(1, int(args.camera_warmup_frames)),
            timeout_s=float(args.camera_timeout_s),
        )
        image_path, depth_path = _save_rgb_depth(rgb, depth.astype(np.float32), out_dir)

    camera_calibration = resolve_real_camera_calibration(args, camera_source=args.camera_source)
    calibration_summary = summarize_camera_calibration(camera_calibration)
    _write_json(out_dir / "camera_calibration.json", calibration_summary)

    if args.schema_path:
        schema = _load_schema(Path(args.schema_path).expanduser().resolve())
        schema_info = {"keypoints": schema, "source": str(args.schema_path)}
    else:
        schema_info = infer_task_keypoint_schema(
            image_path=image_path,
            instruction=args.instruction,
            model=args.model,
            temperature=min(0.2, float(args.temperature)),
            max_tokens=max(200, int(args.schema_max_tokens)),
        )
        schema = schema_info.get("keypoints") or []
    if not schema:
        raise RuntimeError("schema is empty; cannot localize keypoints")
    _write_json(out_dir / "schema.json", schema_info)

    depth_for_localize = np.load(depth_path).astype(np.float32)
    if args.manual_kp2d:
        keypoint_obs = _manual_localize_schema_keypoints(
            image_path=image_path,
            depth_image=depth_for_localize,
            camera_calibration=calibration_summary,
            keypoint_schema=schema,
            args=args,
            out_dir=out_dir,
        )
    else:
        keypoint_obs = localize_schema_keypoints(
            image_path=image_path,
            depth_image=depth_for_localize,
            camera_calibration=calibration_summary,
            keypoint_schema=schema,
            model=args.model,
            temperature=float(args.temperature),
            max_tokens=max(200, int(args.max_tokens)),
        )
    _write_json(out_dir / "keypoint_obs.json", keypoint_obs)

    overlay_path = out_dir / "keypoints_overlay.png"
    draw_keypoints_overlay(image_path, keypoint_obs, overlay_path)

    summary = {
        "ok": True,
        "instruction": args.instruction,
        "camera_source": args.camera_source,
        "capture_info": capture_info,
        "output_dir": str(out_dir),
        "image_path": str(image_path),
        "depth_path": str(depth_path),
        "overlay_path": str(overlay_path),
        "schema_path": str(out_dir / "schema.json"),
        "keypoint_obs_path": str(out_dir / "keypoint_obs.json"),
        "mode": "manual_kp2d" if args.manual_kp2d else "vlm_kp2d",
        "focus_roi_xyxy": keypoint_obs.get("focus_roi_xyxy"),
        "focus_roi_source": keypoint_obs.get("focus_roi_source"),
        "focus_roi_debug_image": keypoint_obs.get("focus_roi_debug_image"),
        "vlm_raw_overlay_paths": keypoint_obs.get("vlm_raw_overlay_paths"),
        "fine_refine_enabled": keypoint_obs.get("fine_refine_enabled"),
        "fine_refine_applied": keypoint_obs.get("fine_refine_applied"),
        "fine_refine_debug_image": keypoint_obs.get("fine_refine_debug_image"),
        "fine_refine_summary": keypoint_obs.get("fine_refine_summary"),
        "keypoints_2d": keypoint_obs.get("keypoints_2d"),
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

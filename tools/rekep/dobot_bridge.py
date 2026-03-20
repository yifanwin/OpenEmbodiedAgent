#!/usr/bin/env python3
import argparse
import ast
import configparser
import copy
import datetime as dt
import importlib.util
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np

from vlm_client import ask_image_question, resolve_vlm_config, vlm_ready
from real_pen_rekep import (
    build_pen_stage_execution_prompt,
    load_pen_rekep_program,
    localize_pen_keypoints,
)
from real_rekep_live import RealTimeConstraintGenerator, load_generated_program
from real_runtime import RealObservation, RealTaskRuntime
from real_rekep_env import RealReKepEnv
from real_stage_runner import RealStageRunner
from real_keypoint_tracker import RealKeypointTracker
from real_constraint_monitor import RealConstraintMonitor
from real_recovery_manager import RealRecoveryManager
from real_constraint_evaluator import RealConstraintEvaluator
from real_grasp_state import RealGraspStateEstimator
from real_task_planner import (
    build_generic_stage_execution_prompt,
    draw_keypoints_overlay,
    infer_task_keypoint_schema,
    localize_schema_keypoints,
)
from hardware_profile import build_hardware_profile, coerce_hardware_profile
from robot_adapter import RobotAdapter


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_REAL_STATE_ENV = "REKEP_REAL_STATE_DIR"
DEFAULT_REAL_STATE_DIR = Path("/tmp/rekep_real")
DEFAULT_CAMERA_SOURCE_ENV = "REKEP_CAMERA_SOURCE"
DEFAULT_ROBOT_FAMILY_ENV = "REKEP_ROBOT_FAMILY"
DEFAULT_ROBOT_DRIVER_ENV = "REKEP_ROBOT_DRIVER"
DEFAULT_ROBOT_HOST_ENV = "REKEP_ROBOT_HOST"
DEFAULT_ROBOT_PORT_ENV = "REKEP_ROBOT_PORT"
DEFAULT_ROBOT_MOVE_PORT_ENV = "REKEP_ROBOT_MOVE_PORT"
DEFAULT_DOBOT_DRIVER_ENV = "REKEP_DOBOT_DRIVER"
DEFAULT_DOBOT_HOST_ENV = "REKEP_DOBOT_HOST"
DEFAULT_DOBOT_PORT_ENV = "REKEP_DOBOT_PORT"
DEFAULT_DOBOT_MOVE_PORT_ENV = "REKEP_DOBOT_MOVE_PORT"
DEFAULT_XTRAINER_SDK_DIR_ENV = "REKEP_XTRAINER_SDK_DIR"
DEFAULT_CAMERA_PROFILE_ENV = "REKEP_CAMERA_PROFILE"
DEFAULT_CAMERA_SERIAL_ENV = "REKEP_CAMERA_SERIAL"
DEFAULT_DOBOT_SETTINGS_INI_ENV = "REKEP_DOBOT_SETTINGS_INI"
DEFAULT_CAMERA_EXTRINSIC_SCRIPT_ENV = "REKEP_CAMERA_EXTRINSIC_SCRIPT"
DEFAULT_REALSENSE_CALIB_DIR_ENV = "REKEP_REALSENSE_CALIB_DIR"
DEFAULT_XTRAINER_SDK_DIR = REPO_DIR / "third_party" / "dobot_xtrainer"
DEFAULT_CAMERA_PROFILE = "global3"
DEFAULT_DOBOT_SETTINGS_INI = REPO_DIR / "real_calibration" / "dobot_settings.ini"
DEFAULT_CAMERA_EXTRINSIC_SCRIPT = REPO_DIR / "real_calibration" / "eval_dobot_v1.py"
DEFAULT_REALSENSE_CALIB_DIR = REPO_DIR / "real_calibration" / "realsense_config"
DEFAULT_REMOTE_DOBOT_HOST = "127.0.0.1"
DEFAULT_REMOTE_DOBOT_PORT = 6001
DEFAULT_REMOTE_REALSENSE_ZMQ_PORT = 7001
DEFAULT_LOCAL_DOBOT_HOST = "192.168.5.1"
DEFAULT_LOCAL_DOBOT_PORT = 29999
DEFAULT_LOCAL_DOBOT_MOVE_PORT = 30003
SUPPORTED_DOBOT_DRIVERS = {"mock", "dashboard_tcp", "xtrainer_sdk", "xtrainer_zmq"}
SUPPORTED_CELLBOT_DRIVERS = {"mock", "cellbot_sdk", "cellbot_rpc"}
DEFAULT_LOCAL_CELLBOT_HOST = "127.0.0.1"
DEFAULT_LOCAL_CELLBOT_PORT = 9000
DEFAULT_REALSENSE_ZMQ_TOPIC = "realsense"
DEFAULT_LONGRUN_MAX_MINUTES = 30.0
DEFAULT_LONGRUN_MONITOR_INTERVAL_S = 2.0
DEFAULT_LONGRUN_RETRY_LIMIT = 2
DEFAULT_ACTION_INTERVAL_S = 8.0
DEFAULT_REKEP_EXECUTION_MODE = "vlm_stage"
SUPPORTED_REKEP_EXECUTION_MODES = {"solver", "vlm_stage"}
DEFAULT_REAL_GRASP_DEPTH_M = 0.03
DEFAULT_REKEP_VLM_STAGE_GRASP_DESCEND_ENV = "REKEP_VLM_STAGE_GRASP_DESCEND_M"
DEFAULT_REKEP_VLM_STAGE_GRASP_DESCEND_M = 0.0
LONGRUN_COMMAND_DIRNAME = "longrun_commands"


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_iso_ts(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def print_json(payload, pretty=False):
    indent = 2 if pretty else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


def emit_progress(message):
    print(message, file=sys.stderr, flush=True)


def parse_boolish(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def resolve_action_interval_s(args):
    raw = getattr(args, "action_interval_s", DEFAULT_ACTION_INTERVAL_S)
    try:
        value = float(raw)
    except Exception:
        value = DEFAULT_ACTION_INTERVAL_S
    return max(0.0, value)


def resolve_rekep_execution_mode(args):
    cli_value = getattr(args, "rekep_execution_mode", None)
    if isinstance(cli_value, str) and not cli_value.strip():
        cli_value = None
    raw = cli_value or os.environ.get("REKEP_EXECUTION_MODE") or DEFAULT_REKEP_EXECUTION_MODE
    mode = str(raw).strip().lower()
    if mode not in SUPPORTED_REKEP_EXECUTION_MODES:
        return DEFAULT_REKEP_EXECUTION_MODE
    return mode


def resolve_rekep_grasp_depth_m(args):
    raw = getattr(args, "rekep_grasp_depth_m", DEFAULT_REAL_GRASP_DEPTH_M)
    try:
        value = float(raw)
    except Exception:
        value = float(DEFAULT_REAL_GRASP_DEPTH_M)
    return max(0.0, value)


def resolve_rekep_vlm_stage_grasp_descend_m(args):
    raw = getattr(args, "rekep_vlm_stage_grasp_descend_m", None)
    if raw is None:
        raw = os.environ.get(DEFAULT_REKEP_VLM_STAGE_GRASP_DESCEND_ENV, DEFAULT_REKEP_VLM_STAGE_GRASP_DESCEND_M)
    try:
        value = float(raw)
    except Exception:
        value = float(DEFAULT_REKEP_VLM_STAGE_GRASP_DESCEND_M)
    return max(0.0, value)


def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def resolve_state_dir(explicit_dir=None):
    raw = explicit_dir or os.environ.get(DEFAULT_REAL_STATE_ENV) or os.environ.get("REKEP_JOB_STATE_DIR")
    if raw:
        root = Path(raw)
        # Keep real-robot runtime isolated from simulation runtime files.
        if root.name != "real":
            root = root / "real"
    else:
        root = DEFAULT_REAL_STATE_DIR
    ensure_dir(root)
    ensure_dir(root / "jobs")
    ensure_dir(root / "logs")
    ensure_dir(root / "frames")
    ensure_dir(root / LONGRUN_COMMAND_DIRNAME)
    return root


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clone_args(args, **updates):
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def read_string(value, fallback=""):
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return fallback


def as_dict(value):
    return value if isinstance(value, dict) else {}


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except Exception:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


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
        pass
    match = re.search(r"(\{[\s\S]*\})\s*$", cleaned)
    if not match:
        return {}
    try:
        value = json.loads(match.group(1))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def parse_camera_source(raw):
    value = str(raw).strip()
    if value.isdigit():
        return int(value)
    return value


def _parse_int_or_none(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def resolve_camera_source(args):
    explicit = read_string(getattr(args, "camera_source", None))
    if explicit:
        return explicit
    fallback = os.environ.get(DEFAULT_CAMERA_SOURCE_ENV, "").strip()
    if fallback:
        return fallback
    family = resolve_robot_family(args)
    driver = resolve_dobot_driver(args, family=family)
    host = resolve_dobot_host(args, family=family, driver=driver)
    if driver == "xtrainer_zmq" and host:
        return f"realsense_zmq://{host}:{DEFAULT_REMOTE_REALSENSE_ZMQ_PORT}/{DEFAULT_REALSENSE_ZMQ_TOPIC}"
    return "0"


def resolve_robot_family(args):
    explicit = read_string(getattr(args, "robot_family", None)).lower()
    if explicit:
        return explicit
    fallback = os.environ.get(DEFAULT_ROBOT_FAMILY_ENV, "").strip().lower()
    return fallback or "dobot"


def resolve_dobot_driver(args, *, family: str | None = None):
    explicit = read_string(getattr(args, "robot_driver", None)).lower()
    if explicit:
        return explicit
    explicit = read_string(getattr(args, "dobot_driver", None)).lower()
    if explicit:
        return explicit
    fallback = os.environ.get(DEFAULT_ROBOT_DRIVER_ENV, "").strip().lower()
    if fallback:
        return fallback
    fallback = os.environ.get(DEFAULT_DOBOT_DRIVER_ENV, "").strip().lower()
    if fallback:
        return fallback
    selected_family = read_string(family, resolve_robot_family(args)).lower()
    if selected_family == "cellbot":
        return "cellbot_sdk"
    return "xtrainer_zmq"


def resolve_dobot_host(args, *, family: str | None = None, driver: str | None = None):
    explicit = read_string(getattr(args, "robot_host", None))
    if explicit:
        return explicit
    explicit = read_string(getattr(args, "dobot_host", None))
    if explicit:
        return explicit
    from_env = os.environ.get(DEFAULT_ROBOT_HOST_ENV, "").strip()
    if from_env:
        return from_env
    from_env = os.environ.get(DEFAULT_DOBOT_HOST_ENV, "").strip()
    if from_env:
        return from_env
    selected_family = read_string(family, resolve_robot_family(args)).lower()
    selected_driver = read_string(driver, resolve_dobot_driver(args, family=selected_family)).lower()
    if selected_family == "dobot":
        if selected_driver == "xtrainer_zmq":
            return DEFAULT_REMOTE_DOBOT_HOST
        return DEFAULT_LOCAL_DOBOT_HOST
    if selected_family == "cellbot":
        return DEFAULT_LOCAL_CELLBOT_HOST
    return ""


def resolve_dobot_port(args, *, family: str | None = None, driver: str | None = None):
    explicit = _parse_int_or_none(getattr(args, "robot_port", None))
    if explicit is not None:
        return explicit
    explicit = _parse_int_or_none(getattr(args, "dobot_port", None))
    if explicit is not None:
        return explicit
    from_env = _parse_int_or_none(os.environ.get(DEFAULT_ROBOT_PORT_ENV))
    if from_env is not None:
        return from_env
    from_env = _parse_int_or_none(os.environ.get(DEFAULT_DOBOT_PORT_ENV))
    if from_env is not None:
        return from_env
    selected_family = read_string(family, resolve_robot_family(args)).lower()
    selected_driver = read_string(driver, resolve_dobot_driver(args, family=selected_family)).lower()
    if selected_family == "dobot":
        if selected_driver == "xtrainer_zmq":
            return DEFAULT_REMOTE_DOBOT_PORT
        return DEFAULT_LOCAL_DOBOT_PORT
    if selected_family == "cellbot":
        return DEFAULT_LOCAL_CELLBOT_PORT
    return None


def resolve_dobot_move_port(args, *, family: str | None = None, driver: str | None = None):
    explicit = _parse_int_or_none(getattr(args, "robot_move_port", None))
    if explicit is not None:
        return explicit
    explicit = _parse_int_or_none(getattr(args, "dobot_move_port", None))
    if explicit is not None:
        return explicit
    from_env = _parse_int_or_none(os.environ.get(DEFAULT_ROBOT_MOVE_PORT_ENV))
    if from_env is not None:
        return from_env
    from_env = _parse_int_or_none(os.environ.get(DEFAULT_DOBOT_MOVE_PORT_ENV))
    if from_env is not None:
        return from_env
    selected_family = read_string(family, resolve_robot_family(args)).lower()
    if selected_family != "dobot":
        return None
    return DEFAULT_LOCAL_DOBOT_MOVE_PORT


def resolve_xtrainer_sdk_dir(args):
    if args.xtrainer_sdk_dir:
        return Path(args.xtrainer_sdk_dir).expanduser().resolve()
    from_env = os.environ.get(DEFAULT_XTRAINER_SDK_DIR_ENV, "").strip()
    if from_env:
        return Path(from_env).expanduser().resolve()
    return DEFAULT_XTRAINER_SDK_DIR


def ensure_xtrainer_import_path(sdk_dir):
    sdk_dir = Path(sdk_dir).expanduser().resolve()
    if not sdk_dir.exists():
        raise RuntimeError(f"xtrainer sdk dir not found: {sdk_dir}")
    if not (sdk_dir / "dobot_control" / "robots" / "dobot_api.py").exists():
        raise RuntimeError(f"xtrainer sdk dir invalid (missing dobot_api.py): {sdk_dir}")
    sdk_dir_str = str(sdk_dir)
    if sdk_dir_str not in sys.path:
        sys.path.insert(0, sdk_dir_str)
    return sdk_dir


def resolve_runtime_hardware_profile(args):
    family = resolve_robot_family(args)
    driver = resolve_dobot_driver(args, family=family)
    host = resolve_dobot_host(args, family=family, driver=driver)
    port = resolve_dobot_port(args, family=family, driver=driver)
    move_port = resolve_dobot_move_port(args, family=family, driver=driver)
    camera_source = resolve_camera_source(args)
    camera_profile = resolve_camera_profile(args)
    return build_hardware_profile(
        robot_family=family,
        robot_driver=driver,
        robot_host=host,
        robot_port=port,
        robot_move_port=move_port,
        xtrainer_sdk_dir=str(resolve_xtrainer_sdk_dir(args)),
        camera_source=camera_source,
        camera_profile=camera_profile,
        extras={
            "robot_family": family,
            "robot_driver": driver,
            "robot_host": host,
            "robot_port": port,
            "robot_move_port": move_port,
            "dobot_host": host,
            "dobot_port": port,
            "dobot_move_port": move_port,
        },
    )


def resolve_profile_from_preflight_or_args(args, preflight):
    preflight = preflight if isinstance(preflight, dict) else {}
    payload = preflight.get("hardware_profile") if isinstance(preflight.get("hardware_profile"), dict) else {}
    if payload:
        return coerce_hardware_profile(payload)
    return resolve_runtime_hardware_profile(args)


def parse_realsense_source(camera_source):
    value = str(camera_source).strip()
    lowered = value.lower()
    if lowered in {"realsense", "rs", "d455"}:
        return {"enabled": True, "serial": None}
    if lowered.startswith("realsense:"):
        serial = value.split(":", 1)[1].strip()
        return {"enabled": True, "serial": serial or None}
    if lowered.startswith("rs:"):
        serial = value.split(":", 1)[1].strip()
        return {"enabled": True, "serial": serial or None}
    return {"enabled": False, "serial": None}


def parse_realsense_zmq_source(camera_source):
    value = str(camera_source).strip()
    lowered = value.lower()
    for prefix in ("realsense_zmq://", "rs_zmq://", "zmq+realsense://"):
        if lowered.startswith(prefix):
            body = value[len(prefix):]
            host_port, _, topic_part = body.partition("/")
            host_port = host_port.strip()
            topic = read_string(topic_part.strip(), DEFAULT_REALSENSE_ZMQ_TOPIC)
            if not host_port:
                return {"enabled": False, "host": "", "port": 0, "topic": DEFAULT_REALSENSE_ZMQ_TOPIC}
            if ":" in host_port:
                host, port_raw = host_port.rsplit(":", 1)
                host = host.strip()
                try:
                    port = int(port_raw)
                except Exception:
                    return {"enabled": False, "host": "", "port": 0, "topic": DEFAULT_REALSENSE_ZMQ_TOPIC}
            else:
                host = host_port
                port = 7001
            if not host:
                return {"enabled": False, "host": "", "port": 0, "topic": DEFAULT_REALSENSE_ZMQ_TOPIC}
            return {"enabled": True, "host": host, "port": port, "topic": topic}

    inline_match = re.match(
        r"^(?:realsense_zmq|rs_zmq):([^:]+):(\d+)(?::([A-Za-z0-9._-]+))?$",
        value,
        flags=re.IGNORECASE,
    )
    if inline_match:
        host = read_string(inline_match.group(1))
        port = int(inline_match.group(2))
        topic = read_string(inline_match.group(3), DEFAULT_REALSENSE_ZMQ_TOPIC)
        return {"enabled": True, "host": host, "port": port, "topic": topic}

    return {"enabled": False, "host": "", "port": 0, "topic": DEFAULT_REALSENSE_ZMQ_TOPIC}


def resolve_camera_profile(args):
    if getattr(args, "camera_profile", None):
        return str(args.camera_profile).strip()
    from_env = os.environ.get(DEFAULT_CAMERA_PROFILE_ENV, "").strip()
    if from_env:
        return from_env
    return DEFAULT_CAMERA_PROFILE


def resolve_camera_serial(args):
    if getattr(args, "camera_serial", None):
        return str(args.camera_serial).strip()
    from_env = os.environ.get(DEFAULT_CAMERA_SERIAL_ENV, "").strip()
    if from_env:
        return from_env
    return ""


def resolve_dobot_settings_ini(args):
    if getattr(args, "dobot_settings_ini", None):
        return Path(args.dobot_settings_ini).expanduser().resolve()
    from_env = os.environ.get(DEFAULT_DOBOT_SETTINGS_INI_ENV, "").strip()
    if from_env:
        return Path(from_env).expanduser().resolve()
    return DEFAULT_DOBOT_SETTINGS_INI


def resolve_camera_extrinsic_script(args):
    if getattr(args, "camera_extrinsic_script", None):
        return Path(args.camera_extrinsic_script).expanduser().resolve()
    from_env = os.environ.get(DEFAULT_CAMERA_EXTRINSIC_SCRIPT_ENV, "").strip()
    if from_env:
        return Path(from_env).expanduser().resolve()
    return DEFAULT_CAMERA_EXTRINSIC_SCRIPT


def resolve_realsense_calib_dir(args):
    if getattr(args, "realsense_calib_dir", None):
        return Path(args.realsense_calib_dir).expanduser().resolve()
    from_env = os.environ.get(DEFAULT_REALSENSE_CALIB_DIR_ENV, "").strip()
    if from_env:
        return Path(from_env).expanduser().resolve()
    return DEFAULT_REALSENSE_CALIB_DIR


def load_camera_serial_map(settings_ini_path):
    mapping = {}
    try:
        if not settings_ini_path.exists():
            return mapping
        parser = configparser.ConfigParser()
        parser.read(str(settings_ini_path), encoding="utf-8")
        if not parser.has_section("CAMERA"):
            return mapping
        section = parser["CAMERA"]
        for key in section:
            value = str(section.get(key, "")).strip()
            if value:
                mapping[key.strip()] = value
    except Exception:
        return {}
    return mapping


def _parse_numpy_array_literal(node):
    if isinstance(node, ast.Call):
        func = node.func
        is_np_array = False
        if isinstance(func, ast.Attribute):
            is_np_array = getattr(func, "attr", "") == "array"
        elif isinstance(func, ast.Name):
            is_np_array = func.id == "array"
        if is_np_array and node.args:
            try:
                value = ast.literal_eval(node.args[0])
                return np.array(value, dtype=float)
            except Exception:
                return None
    return None


def load_camera_extrinsic_profile(script_path, profile_name):
    if not script_path.exists():
        return None
    try:
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script_path))
    except Exception:
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "CAMERA_CONFIGS" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values):
            try:
                key = ast.literal_eval(key_node)
            except Exception:
                continue
            if key != profile_name or not isinstance(value_node, ast.Call):
                continue
            serial = ""
            rotation = None
            translation_mm = None
            for kw in value_node.keywords:
                if kw.arg == "serial":
                    try:
                        serial = str(ast.literal_eval(kw.value)).strip()
                    except Exception:
                        serial = ""
                elif kw.arg == "R":
                    rotation = _parse_numpy_array_literal(kw.value)
                elif kw.arg == "T":
                    translation_mm = _parse_numpy_array_literal(kw.value)
            if rotation is None or translation_mm is None:
                return None
            rotation = np.array(rotation, dtype=float).reshape(3, 3)
            translation_mm = np.array(translation_mm, dtype=float).reshape(3)
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation_mm / 1000.0
            return {
                "profile": str(profile_name),
                "serial": serial,
                "R": rotation.tolist(),
                "T_mm": translation_mm.tolist(),
                "T_m": (translation_mm / 1000.0).tolist(),
                "T_base_camera": transform.tolist(),
                "unit": {
                    "rotation": "matrix",
                    "translation": "mm_in_source/m_in_transform",
                },
                "source": str(script_path),
            }
    return None


def load_realsense_intrinsics(calib_dir, serial):
    if not serial:
        return None
    calib_file = calib_dir / f"realsense_calibration_{serial}_lastest.json"
    if not calib_file.exists():
        return None
    try:
        payload = json.loads(calib_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    color = payload.get("color_intrinsics", {}) if isinstance(payload, dict) else {}
    depth = payload.get("depth_intrinsics", {}) if isinstance(payload, dict) else {}
    if not isinstance(color, dict):
        color = {}
    if not isinstance(depth, dict):
        depth = {}
    return {
        "serial": serial,
        "source": str(calib_file),
        "color": {
            "width": color.get("width"),
            "height": color.get("height"),
            "fx": color.get("fx"),
            "fy": color.get("fy"),
            "cx": color.get("ppx"),
            "cy": color.get("ppy"),
            "distortion_model": color.get("distortion_model") or color.get("model"),
            "distortion_coeffs": color.get("coeffs"),
        },
        "depth": {
            "width": depth.get("width"),
            "height": depth.get("height"),
            "fx": depth.get("fx"),
            "fy": depth.get("fy"),
            "cx": depth.get("ppx"),
            "cy": depth.get("ppy"),
            "distortion_model": depth.get("distortion_model") or depth.get("model"),
            "distortion_coeffs": depth.get("coeffs"),
            "depth_scale": payload.get("depth_scale"),
        },
    }


def resolve_real_camera_calibration(args, camera_source):
    settings_ini_path = resolve_dobot_settings_ini(args)
    script_path = resolve_camera_extrinsic_script(args)
    calib_dir = resolve_realsense_calib_dir(args)
    profile_name = resolve_camera_profile(args)
    serial_map = load_camera_serial_map(settings_ini_path)
    source_serial = parse_realsense_source(camera_source).get("serial")
    requested_serial = resolve_camera_serial(args)

    serial = ""
    if requested_serial:
        serial = requested_serial
    elif source_serial:
        serial = source_serial
    elif profile_name in serial_map:
        serial = serial_map[profile_name]

    extrinsic = load_camera_extrinsic_profile(script_path, profile_name)
    if not serial and isinstance(extrinsic, dict):
        serial = str(extrinsic.get("serial", "")).strip()
    intrinsics = load_realsense_intrinsics(calib_dir, serial)

    warnings = []
    if not settings_ini_path.exists():
        warnings.append(f"dobot settings ini not found: {settings_ini_path}")
    if not script_path.exists():
        warnings.append(f"camera extrinsic script not found: {script_path}")
    if extrinsic is None:
        warnings.append(f"extrinsic profile not found in script: {profile_name}")
    if not calib_dir.exists():
        warnings.append(f"realsense calibration dir not found: {calib_dir}")
    elif not intrinsics:
        warnings.append(f"intrinsics file not found for serial {serial or 'unknown'} in {calib_dir}")

    return {
        "profile": profile_name,
        "serial": serial,
        "camera_source": str(camera_source),
        "settings_ini_path": str(settings_ini_path),
        "camera_extrinsic_script_path": str(script_path),
        "realsense_calib_dir": str(calib_dir),
        "serial_map": serial_map,
        "extrinsic": extrinsic,
        "intrinsic": intrinsics,
        "configured": bool(extrinsic) and bool(intrinsics),
        "warnings": warnings,
    }


def summarize_camera_calibration(calibration):
    calibration = calibration if isinstance(calibration, dict) else {}
    extrinsic = calibration.get("extrinsic") if isinstance(calibration.get("extrinsic"), dict) else {}
    intrinsic = calibration.get("intrinsic") if isinstance(calibration.get("intrinsic"), dict) else {}
    color = intrinsic.get("color") if isinstance(intrinsic.get("color"), dict) else {}
    depth = intrinsic.get("depth") if isinstance(intrinsic.get("depth"), dict) else {}
    return {
        "profile": calibration.get("profile"),
        "serial": calibration.get("serial"),
        "configured": bool(calibration.get("configured")),
        "extrinsic_loaded": bool(extrinsic),
        "intrinsic_loaded": bool(intrinsic),
        "T_base_camera": extrinsic.get("T_base_camera"),
        "color_intrinsic": {
            "fx": color.get("fx"),
            "fy": color.get("fy"),
            "cx": color.get("cx"),
            "cy": color.get("cy"),
            "width": color.get("width"),
            "height": color.get("height"),
        }
        if color
        else None,
        "depth_intrinsic": {
            "fx": depth.get("fx"),
            "fy": depth.get("fy"),
            "cx": depth.get("cx"),
            "cy": depth.get("cy"),
            "width": depth.get("width"),
            "height": depth.get("height"),
            "depth_scale": depth.get("depth_scale"),
        }
        if depth
        else None,
        "sources": {
            "settings_ini_path": calibration.get("settings_ini_path"),
            "camera_extrinsic_script_path": calibration.get("camera_extrinsic_script_path"),
            "realsense_calib_dir": calibration.get("realsense_calib_dir"),
            "intrinsic_file": intrinsic.get("source") if intrinsic else None,
        },
        "warnings": calibration.get("warnings", []),
    }


def build_calibration_prompt_block(calibration_summary):
    if not isinstance(calibration_summary, dict):
        return ""
    lines = []
    if calibration_summary.get("profile"):
        lines.append(f"- profile: {calibration_summary.get('profile')}")
    if calibration_summary.get("serial"):
        lines.append(f"- serial: {calibration_summary.get('serial')}")
    color = calibration_summary.get("color_intrinsic")
    if isinstance(color, dict):
        lines.append(
            "- color_intrinsic: "
            f"fx={color.get('fx')}, fy={color.get('fy')}, cx={color.get('cx')}, cy={color.get('cy')}, "
            f"width={color.get('width')}, height={color.get('height')}"
        )
    transform = calibration_summary.get("T_base_camera")
    if transform:
        lines.append(f"- T_base_camera(4x4): {json.dumps(transform, ensure_ascii=False)}")
    if not lines:
        return ""
    return "Camera calibration (robot base frame context):\n" + "\n".join(lines) + "\n"


def probe_tcp_endpoint(host, port, timeout_s=2.0):
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout_s)):
            return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def sanitize_json_from_text(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_plan_from_vlm_text(text):
    cleaned = sanitize_json_from_text(text)
    try:
        data = json.loads(cleaned)
    except Exception:
        return [], cleaned
    if isinstance(data, dict):
        actions = data.get("actions", [])
    elif isinstance(data, list):
        actions = data
    else:
        actions = []
    if not isinstance(actions, list):
        actions = []
    normalized = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("type", "")).strip().lower()
        if not action_type:
            continue
        normalized.append({"type": action_type, **item})
    return normalized, cleaned


def read_log_tail(log_path, max_chars=12000):
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def append_log_line(log_path, line):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def build_log_path(state_dir, prefix):
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(state_dir) / "logs" / f"{prefix}_{ts}.log"


def build_frame_path(state_dir, prefix):
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(state_dir) / "frames" / f"{prefix}_{ts}.png"


def _open_capture(camera_source):
    parsed_source = parse_camera_source(camera_source)
    cap = cv2.VideoCapture(parsed_source)
    return cap


def _capture_single_frame_realsense(camera_source, output_path, warmup_frames=6, timeout_s=8.0):
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        raise RuntimeError(f"pyrealsense2 is required for RealSense capture: {exc}") from exc

    source = parse_realsense_source(camera_source)
    if not source["enabled"]:
        raise RuntimeError(f"Not a RealSense source: {camera_source}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    if not serials:
        raise RuntimeError("No RealSense devices detected")
    if source["serial"] and source["serial"] not in serials:
        raise RuntimeError(
            f"Requested RealSense serial {source['serial']} not found; detected serials: {serials}"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    if source["serial"]:
        config.enable_device(source["serial"])
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    started = time.monotonic()
    frame = None
    try:
        pipeline.start(config)
        while time.monotonic() - started < timeout_s:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            warmup_frames -= 1
            if warmup_frames <= 0:
                break
        if frame is None:
            raise RuntimeError(f"Failed to capture RealSense frame from source: {camera_source}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write frame image: {output_path}")
        return output_path
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def _capture_single_frame_realsense_zmq(camera_source, output_path, warmup_frames=2, timeout_s=8.0):
    try:
        import zmq
    except Exception as exc:
        raise RuntimeError(f"pyzmq is required for RealSense ZMQ capture: {exc}") from exc

    source = parse_realsense_zmq_source(camera_source)
    if not source["enabled"]:
        raise RuntimeError(f"Not a RealSense ZMQ source: {camera_source}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    endpoint = f"tcp://{source['host']}:{source['port']}"
    context = zmq.Context.instance()
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.RCVTIMEO, max(100, int(float(timeout_s) * 1000)))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(endpoint)
    sock.setsockopt(zmq.SUBSCRIBE, source["topic"].encode("utf-8"))
    started = time.monotonic()
    frame = None
    try:
        while time.monotonic() - started < timeout_s:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            if len(parts) < 3:
                continue
            image_np = np.frombuffer(parts[2], dtype=np.uint8)
            decoded = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if decoded is None:
                continue
            frame = decoded
            warmup_frames -= 1
            if warmup_frames <= 0:
                break
        if frame is None:
            raise RuntimeError(f"Failed to capture RealSense ZMQ frame from source: {camera_source}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write frame image: {output_path}")
        return output_path
    finally:
        sock.close(0)


def capture_single_frame(camera_source, output_path, warmup_frames=6, timeout_s=8.0):
    if parse_realsense_zmq_source(camera_source)["enabled"]:
        return _capture_single_frame_realsense_zmq(
            camera_source=camera_source,
            output_path=output_path,
            warmup_frames=warmup_frames,
            timeout_s=timeout_s,
        )
    if parse_realsense_source(camera_source)["enabled"]:
        return _capture_single_frame_realsense(
            camera_source=camera_source,
            output_path=output_path,
            warmup_frames=warmup_frames,
            timeout_s=timeout_s,
        )
    cap = _open_capture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {camera_source}")
    started = time.monotonic()
    frame = None
    try:
        while time.monotonic() - started < timeout_s:
            ok, candidate = cap.read()
            if not ok or candidate is None:
                time.sleep(0.05)
                continue
            frame = candidate
            warmup_frames -= 1
            if warmup_frames <= 0:
                break
        if frame is None:
            raise RuntimeError(f"Failed to capture frame from camera source: {camera_source}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write frame image: {output_path}")
        return output_path
    finally:
        cap.release()


def capture_realsense_rgbd(camera_source, warmup_frames=6, timeout_s=8.0):
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        raise RuntimeError(f"pyrealsense2 is required for RealSense RGB-D capture: {exc}") from exc

    source = parse_realsense_source(camera_source)
    if not source["enabled"]:
        raise RuntimeError(f"capture_realsense_rgbd only supports RealSense sources, got: {camera_source}")

    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    if not serials:
        raise RuntimeError("No RealSense devices detected")
    if source["serial"] and source["serial"] not in serials:
        raise RuntimeError(f"Requested RealSense serial {source['serial']} not found; detected serials: {serials}")

    pipeline = rs.pipeline()
    config = rs.config()
    if source["serial"]:
        config.enable_device(source["serial"])
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    frame = None
    depth_image = None
    profile = None
    started = time.monotonic()
    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        while time.monotonic() - started < timeout_s:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_image = depth_raw.astype(np.float32) * depth_scale
            warmup_frames -= 1
            if warmup_frames <= 0:
                break
        if frame is None or depth_image is None:
            raise RuntimeError(f"Failed to capture RGB-D frame from RealSense source: {camera_source}")
        return frame, depth_image, {"source": str(camera_source), "camera_type": "realsense_rgbd", "depth_scale": depth_scale, "serial": source.get("serial")}
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass



def _decode_zmq_image_payload(payload, *, color=True):
    arr = np.frombuffer(payload, dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    decoded = cv2.imdecode(arr, flag)
    if decoded is not None:
        return decoded
    return None



def _decode_zmq_depth_payload(payload, depth_scale):
    depth = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(depth, np.ndarray):
        raise RuntimeError("invalid depth payload: expected numpy array")
    depth = depth.astype(np.float32)
    if np.nanmax(depth) > 10.0:
        depth = depth * float(depth_scale)
    return depth



def capture_realsense_zmq_rgbd(camera_source, warmup_frames=2, timeout_s=8.0):
    try:
        import zmq
    except Exception as exc:
        raise RuntimeError(f"pyzmq is required for RealSense ZMQ RGB-D capture: {exc}") from exc

    source = parse_realsense_zmq_source(camera_source)
    if not source["enabled"]:
        raise RuntimeError(f"capture_realsense_zmq_rgbd only supports RealSense ZMQ sources, got: {camera_source}")

    endpoint = f"tcp://{source['host']}:{source['port']}"
    context = zmq.Context.instance()
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.RCVTIMEO, max(100, int(float(timeout_s) * 1000)))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(endpoint)
    sock.setsockopt(zmq.SUBSCRIBE, source["topic"].encode("utf-8"))
    started = time.monotonic()
    frame = None
    depth_image = None
    capture_info = {"source": str(camera_source), "camera_type": "realsense_zmq_rgbd", "serial": None, "depth_scale": 0.001, "topic": source["topic"], "host": source["host"], "port": source["port"]}
    try:
        while time.monotonic() - started < timeout_s:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            if len(parts) < 3:
                continue
            if len(parts) != 4:
                continue
            meta = {}
            try:
                maybe_meta = json.loads(parts[1].decode("utf-8", errors="replace"))
                if isinstance(maybe_meta, dict):
                    meta = maybe_meta
            except Exception:
                meta = {}
            depth_scale = float(meta.get("depth_scale", capture_info.get("depth_scale", 0.001)) or 0.001)
            color_payload = parts[2]
            depth_payload = parts[3]
            decoded_color = _decode_zmq_image_payload(color_payload, color=True)
            try:
                decoded_depth = _decode_zmq_depth_payload(depth_payload, depth_scale)
            except Exception:
                decoded_depth = None
            if decoded_color is None or decoded_depth is None:
                continue
            frame = decoded_color
            depth_image = decoded_depth
            capture_info.update(meta)
            capture_info["depth_scale"] = depth_scale
            warmup_frames -= 1
            if warmup_frames <= 0:
                break
        if frame is None or depth_image is None:
            raise RuntimeError(f"Failed to capture RGB-D frame from RealSense ZMQ source: {camera_source}")
        return frame, depth_image, capture_info
    finally:
        sock.close(0)


class MockDobotAdapter(RobotAdapter):
    def __init__(self):
        self.connected = False
        self.gripper_closed = False
        self.gripper_position = 0.0

    def connect(self):
        self.connected = True
        return {"ok": True, "driver": "mock"}

    def close(self):
        self.connected = False

    def get_runtime_state(self):
        return normalize_robot_runtime_state(
            source="mock",
            connected=self.connected,
            busy=False,
            faulted=False,
            joint_state=[],
            tool_pose=[],
            gripper_closed=self.gripper_closed,
            gripper_position=self.gripper_position,
        )

    def execute_action(self, action, execute_motion=False):
        action_type = action.get("type", "")
        if action_type == "wait":
            duration = float(action.get("seconds", action.get("duration_s", 0.5)))
            time.sleep(max(0.0, min(duration, 5.0)))
        elif action_type == "open_gripper":
            self.gripper_closed = False
            self.gripper_position = 0.0
            time.sleep(0.1)
        elif action_type == "close_gripper":
            self.gripper_closed = True
            self.gripper_position = 1.0
            time.sleep(0.1)
        else:
            # Keep mocked latency so long procedures in OpenClaw show progress naturally.
            time.sleep(0.2)
        return normalize_action_result(
            driver="mock",
            action=action,
            executed=bool(execute_motion),
            ok=True,
        )


class DashboardTcpDobotAdapter(RobotAdapter):
    def __init__(self, host, port):
        self.host = host
        self.port = int(port)
        self.sock = None
        self.gripper_closed = False
        self.gripper_position = 0.0

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
        self.sock.settimeout(5.0)
        return {"ok": True, "driver": "dashboard_tcp", "host": self.host, "port": self.port}

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def get_runtime_state(self):
        return normalize_robot_runtime_state(
            source="dashboard_tcp",
            connected=self.sock is not None,
            busy=False,
            faulted=False,
            joint_state=[],
            tool_pose=[],
            gripper_closed=self.gripper_closed,
            gripper_position=self.gripper_position,
            extra={"host": self.host, "port": self.port},
        )

    def _send(self, command):
        if self.sock is None:
            raise RuntimeError("Dobot dashboard connection is not open")
        payload = f"{command}\n".encode("utf-8")
        self.sock.sendall(payload)
        data = self.sock.recv(4096)
        return data.decode("utf-8", errors="replace").strip()

    def execute_action(self, action, execute_motion=False):
        action_type = action.get("type", "")
        if action_type == "wait":
            duration = float(action.get("seconds", action.get("duration_s", 0.5)))
            time.sleep(max(0.0, min(duration, 30.0)))
            return normalize_action_result(
                driver="dashboard_tcp",
                action=action,
                executed=True,
                ok=True,
                command_response="wait",
            )

        if not execute_motion:
            return normalize_action_result(
                driver="dashboard_tcp",
                action=action,
                executed=False,
                ok=True,
                dry_run=True,
            )

        if action_type == "movej":
            joints = action.get("joints", [])
            if not isinstance(joints, list) or len(joints) < 4:
                raise RuntimeError("movej action requires 'joints' list with at least 4 values")
            cmd = f"JointMovJ({','.join(str(float(v)) for v in joints)})"
            response = self._send(cmd)
        elif action_type == "movel":
            pose = action.get("pose", [])
            if not isinstance(pose, list) or len(pose) < 4:
                raise RuntimeError("movel action requires 'pose' list with at least 4 values")
            cmd = f"MovL({','.join(str(float(v)) for v in pose)})"
            response = self._send(cmd)
        elif action_type == "open_gripper":
            response = self._send("DO(1,0)")
            self.gripper_closed = False
            self.gripper_position = 0.0
        elif action_type == "close_gripper":
            response = self._send("DO(1,1)")
            self.gripper_closed = True
            self.gripper_position = 1.0
        else:
            raise RuntimeError(f"Unsupported action type for dashboard_tcp: {action_type}")
        return normalize_action_result(
            driver="dashboard_tcp",
            action=action,
            executed=True,
            ok=True,
            command_response=response,
        )


class XtrainerSdkDobotAdapter(RobotAdapter):
    def __init__(self, sdk_dir, robot_ip, dashboard_port, move_port):
        self.sdk_dir = Path(sdk_dir).expanduser().resolve()
        self.robot_ip = robot_ip
        self.dashboard_port = int(dashboard_port)
        self.move_port = int(move_port)
        self.dashboard = None
        self.move = None
        self._dobot_api = None
        self.gripper_closed = False
        self.gripper_position = 0.0

    def connect(self):
        ensure_xtrainer_import_path(self.sdk_dir)
        from dobot_control.robots import dobot_api

        self._dobot_api = dobot_api
        self.dashboard = dobot_api.DobotApiDashboard(self.robot_ip, self.dashboard_port)
        self.move = dobot_api.DobotApiMove(self.robot_ip, self.move_port)
        enable_resp = self.dashboard.EnableRobot()
        speed_resp = self.dashboard.SpeedFactor(20)
        return {
            "ok": True,
            "driver": "xtrainer_sdk",
            "robot_ip": self.robot_ip,
            "dashboard_port": self.dashboard_port,
            "move_port": self.move_port,
            "sdk_dir": str(self.sdk_dir),
            "enable_response": enable_resp,
            "speed_response": speed_resp,
        }

    def close(self):
        for obj in (self.move, self.dashboard):
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:
                pass
        self.move = None
        self.dashboard = None

    def get_runtime_state(self):
        return normalize_robot_runtime_state(
            source="xtrainer_sdk",
            connected=self.dashboard is not None and self.move is not None,
            busy=False,
            faulted=False,
            joint_state=[],
            tool_pose=[],
            gripper_closed=self.gripper_closed,
            gripper_position=self.gripper_position,
            extra={
                "robot_ip": self.robot_ip,
                "dashboard_port": self.dashboard_port,
                "move_port": self.move_port,
                "sdk_dir": str(self.sdk_dir),
            },
        )

    def execute_action(self, action, execute_motion=False):
        action_type = str(action.get("type", "")).strip().lower()
        if action_type == "wait":
            duration = float(action.get("seconds", action.get("duration_s", 0.5)))
            time.sleep(max(0.0, min(duration, 30.0)))
            return normalize_action_result(
                driver="xtrainer_sdk",
                action=action,
                executed=True,
                ok=True,
                command_response="wait",
            )

        if not execute_motion:
            return normalize_action_result(
                driver="xtrainer_sdk",
                action=action,
                executed=False,
                ok=True,
                dry_run=True,
            )

        if self.dashboard is None or self.move is None:
            raise RuntimeError("xtrainer sdk connection is not open")

        if action_type == "movej":
            joints = action.get("joints", [])
            if not isinstance(joints, list) or len(joints) < 6:
                raise RuntimeError("movej action requires 'joints' list with 6 values")
            response = self.move.JointMovJ(*[float(v) for v in joints[:6]])
        elif action_type == "movel":
            pose = action.get("pose", [])
            if not isinstance(pose, list) or len(pose) < 6:
                raise RuntimeError("movel action requires 'pose' list with 6 values")
            response = self.move.MovL(*[float(v) for v in pose[:6]])
        elif action_type == "open_gripper":
            response = self.dashboard.DO(1, 0)
            self.gripper_closed = False
            self.gripper_position = 0.0
        elif action_type == "close_gripper":
            response = self.dashboard.DO(1, 1)
            self.gripper_closed = True
            self.gripper_position = 1.0
        else:
            raise RuntimeError(f"Unsupported action type for xtrainer_sdk: {action_type}")
        return normalize_action_result(
            driver="xtrainer_sdk",
            action=action,
            executed=True,
            ok=True,
            command_response=response,
        )


class XtrainerZmqDobotAdapter(RobotAdapter):
    def __init__(self, host, port, request_timeout_ms=3000):
        self.host = str(host)
        self.port = int(port)
        self.request_timeout_ms = int(request_timeout_ms)
        self._context = None
        self._socket = None
        self._pickle = None
        self._np = None
        self._left_joint_state = [0.0] * 7
        self._right_joint_state = [0.0] * 7
        self._left_tool_pose = []
        self._right_tool_pose = []
        self._last_active_arm = "right"

    def _open_socket(self):
        import pickle
        import zmq

        if self._context is None:
            self._context = zmq.Context.instance()
        sock = self._context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        sock.connect(f"tcp://{self.host}:{self.port}")
        self._socket = sock
        self._pickle = pickle
        self._np = np

    def _reset_socket(self):
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
            self._socket = None
        self._open_socket()

    def _rpc(self, method, args=None):
        if self._socket is None:
            raise RuntimeError("xtrainer_zmq connection is not open")
        request = {"method": method}
        if args:
            request["args"] = args
        try:
            self._socket.send(self._pickle.dumps(request))
            response = self._socket.recv()
            return self._pickle.loads(response)
        except Exception as exc:
            try:
                self._reset_socket()
            except Exception:
                pass
            raise RuntimeError(f"xtrainer_zmq rpc failed ({method}): {exc}") from exc

    def _set_left_joint_state_from_remote(self):
        remote = self._rpc("get_joint_state")
        arr = self._np.asarray(remote, dtype=float).flatten()
        if arr.size >= 14:
            self._left_joint_state = [float(v) for v in arr[:7]]
            self._right_joint_state = [float(v) for v in arr[7:14]]
        elif arr.size >= 7:
            self._left_joint_state = [float(v) for v in arr[:7]]
            self._right_joint_state = [0.0] * 7
        elif arr.size >= 6:
            self._left_joint_state = [float(v) for v in arr[:6]] + [1.0]
            self._right_joint_state = [0.0] * 7
        else:
            self._left_joint_state = [0.0] * 7
            self._right_joint_state = [0.0] * 7

    def _gripper_position_for_arm(self, arm_name):
        if arm_name == "left":
            return float(self._left_joint_state[6]) if len(self._left_joint_state) >= 7 else None
        if arm_name == "both":
            return None
        return float(self._right_joint_state[6]) if len(self._right_joint_state) >= 7 else None

    def _set_tool_pose_from_remote(self):
        remote = self._rpc("get_XYZrxryrz_state")
        arr = self._np.asarray(remote, dtype=float).flatten()
        if arr.size >= 12:
            self._left_tool_pose = [float(v) for v in arr[:6]]
            self._right_tool_pose = [float(v) for v in arr[6:12]]
        elif arr.size >= 6:
            pose = [float(v) for v in arr[:6]]
            if self._last_active_arm == "left":
                self._left_tool_pose = pose
            else:
                self._right_tool_pose = pose
        else:
            self._left_tool_pose = []
            self._right_tool_pose = []

    def _set_tool_pose_for_arm(self, arm_name, pose):
        pose_list = [float(v) for v in list(pose)[:6]]
        if len(pose_list) < 6:
            return
        if arm_name == "left":
            self._left_tool_pose = pose_list
        elif arm_name == "right":
            self._right_tool_pose = pose_list
        elif arm_name == "both":
            self._left_tool_pose = pose_list
            self._right_tool_pose = pose_list

    def _set_gripper_position_for_arm(self, arm_name, value):
        v = float(value)
        if arm_name in {"left", "both"} and len(self._left_joint_state) >= 7:
            self._left_joint_state[6] = v
        if arm_name in {"right", "both"} and len(self._right_joint_state) >= 7:
            self._right_joint_state[6] = v

    @staticmethod
    def _arm_value_for_do(arm_name):
        if arm_name == "left":
            return "left"
        if arm_name == "both":
            return "both"
        return "right"

    @staticmethod
    def _joint_state_for_arm(arm_name, left_joint_state, right_joint_state):
        if arm_name == "left":
            return list(left_joint_state)
        if arm_name == "both":
            return list(left_joint_state) + list(right_joint_state)
        return list(right_joint_state)

    def _set_joint_state_for_arm(self, arm_name, joint_values):
        if arm_name == "left":
            self._left_joint_state = list(joint_values)
        elif arm_name == "right":
            self._right_joint_state = list(joint_values)
        else:
            self._set_left_joint_state_from_remote()

    @staticmethod
    def _full_state_for_arm(arm_name, joint_values):
        if arm_name in {"left", "right"}:
            return list(joint_values)
        # both: duplicate for left and right when one payload is provided
        return list(joint_values) + list(joint_values)

    def _set_last_active_arm(self, arm_name):
        if arm_name in {"left", "right"}:
            self._last_active_arm = arm_name

    def _command_gripper_via_joint_state(self, arm_name, gripper_value):
        target = float(gripper_value)
        if arm_name == "both":
            left_state = list(self._left_joint_state) if len(self._left_joint_state) >= 7 else [0.0] * 7
            right_state = list(self._right_joint_state) if len(self._right_joint_state) >= 7 else [0.0] * 7
            left_state[6] = target
            right_state[6] = target
            response = self._rpc(
                "command_joint_state",
                {"joint_state": left_state + right_state, "flag_in": [1, 1]},
            )
            self._left_joint_state = left_state
            self._right_joint_state = right_state
            return response

        flag_in = [1, 0] if arm_name == "left" else [0, 1]
        current = self._left_joint_state if arm_name == "left" else self._right_joint_state
        if len(current) < 7 or all(abs(v) < 1e-9 for v in current[:6]):
            # Best effort refresh if local cache is not initialized.
            self._set_left_joint_state_from_remote()
            current = self._left_joint_state if arm_name == "left" else self._right_joint_state
        joint_state = list(current) if len(current) >= 7 else [0.0] * 7
        joint_state[6] = target
        response = self._rpc(
            "command_joint_state",
            {"joint_state": joint_state, "flag_in": flag_in},
        )
        if arm_name == "left":
            self._left_joint_state = joint_state
        else:
            self._right_joint_state = joint_state
        return response

    def _command_gripper_via_rpc(self, arm_name, gripper_value):
        target = float(gripper_value)
        if arm_name == "both":
            return self._rpc(
                "command_gripper",
                {"position": [target, target], "flag_in": [1, 1]},
            )
        flag_in = [1, 0] if arm_name == "left" else [0, 1]
        return self._rpc(
            "command_gripper",
            {"position": [target], "flag_in": flag_in},
        )

    def _command_gripper_with_fallback(self, arm_name, gripper_value, do_status):
        errors = {}
        try:
            response = self._command_gripper_via_rpc(arm_name, gripper_value)
            return response, "command_gripper_rpc", errors
        except Exception as exc:
            errors["command_gripper_rpc_error"] = str(exc)
        try:
            response = self._command_gripper_via_joint_state(arm_name, gripper_value)
            return response, "joint_state_fallback", errors
        except Exception as exc:
            errors["joint_state_fallback_error"] = str(exc)
        response = self._rpc(
            "set_do_status",
            {"which_do": do_status, "arm": self._arm_value_for_do(arm_name)},
        )
        return response, "do_fallback", errors

    def connect(self):
        self._open_socket()
        num_dofs = int(self._rpc("num_dofs"))
        self._set_left_joint_state_from_remote()
        try:
            self._set_tool_pose_from_remote()
        except Exception:
            pass
        return {
            "ok": True,
            "driver": "xtrainer_zmq",
            "host": self.host,
            "port": self.port,
            "num_dofs": num_dofs,
        }

    def close(self):
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
            self._socket = None

    def get_runtime_state(self):
        joint_state = self._joint_state_for_arm(self._last_active_arm, self._left_joint_state, self._right_joint_state)
        gripper_position = self._gripper_position_for_arm(self._last_active_arm)
        gripper_closed = None
        if gripper_position is not None:
            gripper_closed = gripper_position < 0.5
        tool_pose = self.get_tool_pose_mm_deg(self._last_active_arm)
        return normalize_robot_runtime_state(
            source="xtrainer_zmq",
            connected=self._socket is not None,
            busy=False,
            faulted=False,
            joint_state=joint_state,
            tool_pose=tool_pose,
            gripper_closed=gripper_closed,
            gripper_position=gripper_position,
            extra={"host": self.host, "port": self.port, "active_arm": self._last_active_arm},
        )

    def get_tool_pose_mm_deg(self, arm_name=None):
        target_arm = str(arm_name or self._last_active_arm or "right").strip().lower()
        try:
            self._set_tool_pose_from_remote()
        except Exception:
            pass
        if target_arm == "left":
            return list(self._left_tool_pose)
        if target_arm == "both":
            return list(self._left_tool_pose) + list(self._right_tool_pose)
        return list(self._right_tool_pose)

    @staticmethod
    def _normalize_joint_values(raw_joints):
        joints = [float(v) for v in raw_joints[:6]]
        if max(abs(v) for v in joints) > 7.0:
            joints = [float(np.deg2rad(v)) for v in joints]
        return joints

    @staticmethod
    def _resolve_action_arm(action, default="right"):
        arm = str((action or {}).get("arm", default)).strip().lower()
        if arm in {"left", "l"}:
            return [1, 0], "left"
        if arm in {"both", "bimanual", "lr"}:
            return [1, 1], "both"
        return [0, 1], "right"

    @staticmethod
    def _prepare_movel_pose(action):
        raw_pose = action.get("pose", [])
        if not isinstance(raw_pose, list) or len(raw_pose) < 6:
            raise RuntimeError("movel action requires 'pose' list with 6 values")
        pose = [float(v) for v in raw_pose[:6]]
        notes = []

        # ReKep default contract: MovL pose already uses mm + deg.
        # Keep passthrough by default; only convert when explicitly requested.
        normalize_units = parse_boolish(action.get("normalize_units"))
        if normalize_units is None:
            normalize_units = False
        units = str(action.get("units", "")).strip().lower()
        if units in {"mm_deg", "mm+deg", "mm/deg"}:
            normalize_units = False
        elif units in {"m_rad", "m+rad", "m/rad"}:
            normalize_units = True
            pose[:3] = [v * 1000.0 for v in pose[:3]]
            pose[3:] = [float(np.rad2deg(v)) for v in pose[3:]]
            notes.append("unit_explicit_m_rad_to_mm_deg")
            return pose, notes

        if normalize_units:
            # Backward-compatibility mode: infer likely m/rad and convert.
            if max(abs(v) for v in pose[:3]) < 5.0:
                pose[:3] = [v * 1000.0 for v in pose[:3]]
                notes.append("position_converted_m_to_mm")
            if max(abs(v) for v in pose[3:]) <= (2.0 * np.pi + 1e-6):
                pose[3:] = [float(np.rad2deg(v)) for v in pose[3:]]
                notes.append("rotation_converted_rad_to_deg")
        return pose, notes

    def execute_action(self, action, execute_motion=False):
        action_type = str(action.get("type", "")).strip().lower()
        if action_type == "wait":
            duration = float(action.get("seconds", action.get("duration_s", 0.5)))
            time.sleep(max(0.0, min(duration, 30.0)))
            return normalize_action_result(
                driver="xtrainer_zmq",
                action=action,
                executed=True,
                ok=True,
                command_response="wait",
            )

        if not execute_motion:
            return normalize_action_result(
                driver="xtrainer_zmq",
                action=action,
                executed=False,
                ok=True,
                dry_run=True,
            )

        if action_type == "movej":
            joints = action.get("joints", [])
            if not isinstance(joints, list) or len(joints) < 6:
                raise RuntimeError("movej action requires 'joints' list with 6 values")
            joint_values = self._normalize_joint_values(joints)
            flag_in, arm_name = self._resolve_action_arm(action, default="right")
            gripper_value = self._gripper_position_for_arm(arm_name)
            if gripper_value is None:
                gripper_value = 1.0
            full_state = joint_values + [float(gripper_value)]
            response = self._rpc(
                "command_joint_state",
                {"joint_state": full_state, "flag_in": flag_in},
            )
            self._set_joint_state_for_arm(arm_name, full_state)
            self._set_last_active_arm(arm_name)
            if isinstance(response, dict):
                response = dict(response)
                response["arm"] = arm_name
        elif action_type == "movel":
            pose, unit_notes = self._prepare_movel_pose(action)
            flag_in, arm_name = self._resolve_action_arm(action, default="right")
            try:
                response = self._rpc(
                    "command_movel",
                    {"pose": pose, "flag_in": flag_in},
                )
            except Exception as exc:
                raise RuntimeError(
                    "xtrainer_zmq command_movel failed. "
                    "Please update remote server to include `command_movel` RPC support."
                ) from exc
            self._set_tool_pose_for_arm(arm_name, pose)
            try:
                self._set_tool_pose_from_remote()
            except Exception:
                pass
            self._set_last_active_arm(arm_name)
            if isinstance(response, dict):
                response = dict(response)
                response["arm"] = arm_name
                if unit_notes:
                    response["unit_normalization"] = unit_notes
            elif unit_notes:
                response = {"result": response, "arm": arm_name, "unit_normalization": unit_notes}
            else:
                response = {"result": response, "arm": arm_name}
        elif action_type == "open_gripper":
            _, arm_name = self._resolve_action_arm(action, default="right")
            response, gripper_control_mode, gripper_errors = self._command_gripper_with_fallback(
                arm_name,
                1.0,
                [1, 0],
            )
            self._set_gripper_position_for_arm(arm_name, 1.0)
            self._set_last_active_arm(arm_name)
            if isinstance(response, dict):
                response = dict(response)
                response["arm"] = arm_name
                response["gripper_control"] = gripper_control_mode
                response.update(gripper_errors)
            else:
                response = {
                    "result": response,
                    "arm": arm_name,
                    "gripper_control": gripper_control_mode,
                    **gripper_errors,
                }
        elif action_type == "close_gripper":
            _, arm_name = self._resolve_action_arm(action, default="right")
            response, gripper_control_mode, gripper_errors = self._command_gripper_with_fallback(
                arm_name,
                0.0,
                [1, 1],
            )
            self._set_gripper_position_for_arm(arm_name, 0.0)
            self._set_last_active_arm(arm_name)
            if isinstance(response, dict):
                response = dict(response)
                response["arm"] = arm_name
                response["gripper_control"] = gripper_control_mode
                response.update(gripper_errors)
            else:
                response = {
                    "result": response,
                    "arm": arm_name,
                    "gripper_control": gripper_control_mode,
                    **gripper_errors,
                }
        else:
            raise RuntimeError(f"Unsupported action type for xtrainer_zmq: {action_type}")

        return normalize_action_result(
            driver="xtrainer_zmq",
            action=action,
            executed=True,
            ok=True,
            command_response=response,
        )


def normalize_robot_runtime_state(*, source, connected=None, busy=None, faulted=None, tool_pose=None, joint_state=None, gripper_closed=None, gripper_position=None, extra=None):
    payload = {
        "source": read_string(source, "unknown"),
        "connected": bool(connected) if connected is not None else False,
        "busy": bool(busy) if busy is not None else False,
        "faulted": bool(faulted) if faulted is not None else False,
        "tool_pose": list(tool_pose) if isinstance(tool_pose, (list, tuple)) else [],
        "joint_state": list(joint_state) if isinstance(joint_state, (list, tuple)) else [],
        "gripper_closed": gripper_closed if isinstance(gripper_closed, bool) or gripper_closed is None else None,
        "gripper_position": float(gripper_position) if isinstance(gripper_position, (int, float)) else None,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload



def probe_realsense_zmq_rgbd(camera_source, timeout_s=2.5):
    try:
        rgb, depth, info = capture_realsense_zmq_rgbd(camera_source, warmup_frames=1, timeout_s=timeout_s)
        return {
            "ok": True,
            "rgb_shape": list(rgb.shape) if isinstance(rgb, np.ndarray) else [],
            "depth_shape": list(depth.shape) if isinstance(depth, np.ndarray) else [],
            "capture_info": info if isinstance(info, dict) else {},
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }



def normalize_action_result(*, driver, action, executed, ok=True, dry_run=False, command_response=None, error=None, extra=None):
    action_type = read_string((action or {}).get("type"), "unknown") if isinstance(action, dict) else "unknown"
    payload = {
        "ok": bool(ok),
        "driver": read_string(driver, "unknown"),
        "executed": bool(executed),
        "dry_run": bool(dry_run),
        "action_type": action_type,
        "action": action if isinstance(action, dict) else {},
        "command_response": command_response,
        "error": read_string(error) or None,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload



def create_dobot_adapter(driver, host, port, move_port, xtrainer_sdk_dir) -> RobotAdapter:
    # Backward-compatible shim: keep symbol stable while construction is owned by robot_factory.
    from robot_factory import create_robot_adapter

    profile = build_hardware_profile(
        robot_family="dobot",
        robot_driver=driver,
        robot_host=host,
        robot_port=port,
        robot_move_port=move_port,
        xtrainer_sdk_dir=str(xtrainer_sdk_dir or ""),
        camera_source="0",
        camera_profile="global3",
    )
    return create_robot_adapter(hardware_profile=profile)


def run_preflight(args, require_vlm=False):
    hardware_profile = resolve_runtime_hardware_profile(args)
    family = read_string(hardware_profile.robot_family, "dobot").lower()
    driver = hardware_profile.robot_driver
    camera_source = hardware_profile.camera_source
    camera_calibration = resolve_real_camera_calibration(args, camera_source=camera_source)
    camera_calibration_summary = summarize_camera_calibration(camera_calibration)
    xtrainer_sdk_dir = Path(hardware_profile.xtrainer_sdk_dir).expanduser().resolve()
    realsense_source = parse_realsense_source(camera_source)
    realsense_zmq_source = parse_realsense_zmq_source(camera_source)
    blockers = []
    notes = []
    modules = {
        "cv2": module_available("cv2"),
        "requests": module_available("requests"),
        "pyrealsense2": module_available("pyrealsense2"),
        "zmq": module_available("zmq"),
    }
    vlm = resolve_vlm_config(default_model="gpt-5.4")
    if not modules["cv2"]:
        blockers.append("opencv-missing")
    if family == "dobot":
        if driver not in SUPPORTED_DOBOT_DRIVERS:
            blockers.append("unsupported-dobot-driver")
        if driver == "dashboard_tcp":
            if not hardware_profile.robot_host:
                blockers.append("dobot-host-missing")
            if not hardware_profile.robot_port:
                blockers.append("dobot-port-missing")
            notes.append("dashboard_tcp uses textual Dobot dashboard commands; verify firmware command compatibility.")
        if driver == "xtrainer_sdk":
            if not hardware_profile.robot_host:
                blockers.append("xtrainer-robot-ip-missing")
            if not xtrainer_sdk_dir.exists():
                blockers.append("xtrainer-sdk-dir-missing")
            elif not (xtrainer_sdk_dir / "dobot_control" / "robots" / "dobot_api.py").exists():
                blockers.append("xtrainer-sdk-invalid")
            else:
                try:
                    ensure_xtrainer_import_path(xtrainer_sdk_dir)
                    __import__("dobot_control.robots.dobot_api")
                except Exception as exc:
                    blockers.append("xtrainer-sdk-import-failed")
                    notes.append(f"xtrainer sdk import error: {exc}")
            notes.append(f"xtrainer sdk dir: {xtrainer_sdk_dir}")
            notes.append(f"xtrainer robot ip: {hardware_profile.robot_host}")
        if driver == "xtrainer_zmq":
            if not hardware_profile.robot_host:
                blockers.append("xtrainer-zmq-host-missing")
            if not hardware_profile.robot_port:
                blockers.append("xtrainer-zmq-port-missing")
            if not modules["zmq"]:
                blockers.append("pyzmq-missing")
            elif hardware_profile.robot_host and hardware_profile.robot_port:
                probe = probe_tcp_endpoint(hardware_profile.robot_host, hardware_profile.robot_port, timeout_s=2.0)
                if not probe["ok"]:
                    blockers.append("xtrainer-zmq-unreachable")
                    notes.append(f"xtrainer_zmq connect error: {probe['error']}")
                    notes.append("Remote ZMQ robot server must bind 0.0.0.0 (not 127.0.0.1).")
            notes.append(
                "Robot ZMQ server reference: dobot_xtrainer_remote/experiments/launch_nodes.py --hostname 0.0.0.0 --robot-port 6001"
            )
            notes.append(f"xtrainer_zmq endpoint: {hardware_profile.robot_host}:{hardware_profile.robot_port}")
    elif family == "cellbot":
        if not driver:
            blockers.append("cellbot-driver-missing")
        elif driver not in SUPPORTED_CELLBOT_DRIVERS:
            notes.append(
                f"cellbot driver '{driver}' is not in built-in list {sorted(SUPPORTED_CELLBOT_DRIVERS)}; "
                "assuming custom adapter registration."
            )
        notes.append("custom family=cellbot preflight uses generic checks; ensure adapter-specific checks are implemented.")
        if hardware_profile.robot_host and hardware_profile.robot_port:
            probe = probe_tcp_endpoint(hardware_profile.robot_host, hardware_profile.robot_port, timeout_s=1.5)
            if not probe["ok"]:
                notes.append(f"cellbot endpoint probe failed: {probe['error']}")
            else:
                notes.append(f"cellbot endpoint reachable: {hardware_profile.robot_host}:{hardware_profile.robot_port}")
    else:
        if not driver:
            blockers.append("robot-driver-missing")
        notes.append(
            f"custom family={family} detected; "
            "only generic preflight checks were applied. Add family-specific checks in run_preflight()."
        )
    if realsense_source["enabled"]:
        if not modules["pyrealsense2"]:
            blockers.append("pyrealsense2-missing")
        notes.append("RealSense source enabled; use camera_source='realsense' or 'realsense:<serial>'.")
    if realsense_zmq_source["enabled"]:
        if not modules["zmq"]:
            blockers.append("pyzmq-missing")
        else:
            probe = probe_tcp_endpoint(realsense_zmq_source["host"], realsense_zmq_source["port"], timeout_s=2.0)
            if not probe["ok"]:
                blockers.append("realsense-zmq-unreachable")
                notes.append(f"realsense_zmq connect error: {probe['error']}")
            else:
                rgbd_probe = probe_realsense_zmq_rgbd(camera_source, timeout_s=2.5)
                if not rgbd_probe.get("ok"):
                    blockers.append("realsense-zmq-rgbd-invalid")
                    notes.append(f"realsense_zmq rgbd probe error: {rgbd_probe.get('error')}")
                else:
                    notes.append(f"realsense_zmq rgb shape: {rgbd_probe.get('rgb_shape')}")
                    notes.append(f"realsense_zmq depth shape: {rgbd_probe.get('depth_shape')}")
                    probe_info = rgbd_probe.get("capture_info") if isinstance(rgbd_probe.get("capture_info"), dict) else {}
                    notes.append(f"realsense_zmq serial: {probe_info.get('serial') or probe_info.get('serial_number') or 'unknown'}")
                    notes.append(f"realsense_zmq depth_scale: {probe_info.get('depth_scale')}")
        notes.append(
            "RealSense ZMQ source enabled; use camera_source='realsense_zmq://<host>:<port>/<topic>'."
        )
        notes.append(
            "RealSense ZMQ server reference: dobot_xtrainer_remote/experiments/launch_realsense_server.py --host 0.0.0.0 --port 7001 --topic realsense"
        )
        notes.append(
            f"realsense_zmq endpoint: {realsense_zmq_source['host']}:{realsense_zmq_source['port']} topic={realsense_zmq_source['topic']}"
        )
    if require_vlm and not vlm_ready(default_model=vlm["model"]):
        blockers.append("vlm-api-key-missing")
    if camera_calibration_summary.get("profile"):
        notes.append(
            f"camera calibration profile={camera_calibration_summary.get('profile')} "
            f"serial={camera_calibration_summary.get('serial') or 'unknown'} "
            f"configured={camera_calibration_summary.get('configured')}"
        )
        if read_string(camera_calibration_summary.get('profile')).lower() == 'global3':
            notes.append("camera profile global3 selected (expected D455 remote camera with repo extrinsics)")
    for warning in camera_calibration_summary.get("warnings", []):
        notes.append(f"camera calibration warning: {warning}")

    return {
        "status": "ready" if not blockers else "blocked",
        "family": family,
        "driver": driver,
        "camera_source": camera_source,
        "hardware_profile": hardware_profile.to_dict(),
        "xtrainer_sdk_dir": str(xtrainer_sdk_dir),
        "execute_motion": bool(args.execute_motion),
        "modules": modules,
        "vlm": {
            "configured": bool(vlm["api_key"]),
            "model": vlm["model"],
            "base_url": vlm["base_url"],
            "api_key_env": vlm["api_key_env"],
        },
        "camera_calibration": camera_calibration_summary,
        "blockers": blockers,
        "notes": notes,
    }


def get_standby_state_path(state_dir):
    return Path(state_dir) / "standby.json"


def load_standby_state(state_dir):
    data = load_json_if_exists(get_standby_state_path(state_dir))
    return data if isinstance(data, dict) else {}


def save_standby_state(state_dir, payload):
    state_path = get_standby_state_path(state_dir)
    payload = dict(payload)
    payload["updated_at"] = now_iso()
    atomic_write_json(state_path, payload)
    return payload


def run_standby_worker(args):
    state_dir = resolve_state_dir(args.state_dir)
    state_path = get_standby_state_path(state_dir)
    hardware_profile = resolve_runtime_hardware_profile(args)
    camera_source = hardware_profile.camera_source
    interval_s = max(0.05, float(args.interval_s))
    latest_frame_path = Path(state_dir) / "latest_frame.jpg"
    stop_flag = {"value": False}

    def _handle_stop(_signum, _frame):
        stop_flag["value"] = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    save_standby_state(
        state_dir,
        {
            "status": "running",
            "error": None,
            "camera_source": camera_source,
            "pid": os.getpid(),
            "process_group_id": os.getpgrp(),
            "started_at": now_iso(),
            "heartbeat_at": now_iso(),
            "latest_frame_path": str(latest_frame_path),
            "interval_s": interval_s,
        },
    )
    emit_progress(f"[real] standby worker started camera_source={camera_source} interval_s={interval_s}")

    using_realsense = parse_realsense_source(camera_source)["enabled"]
    using_realsense_zmq = parse_realsense_zmq_source(camera_source)["enabled"]
    cap = None
    if not (using_realsense or using_realsense_zmq):
        cap = _open_capture(camera_source)
        if not cap.isOpened():
            save_standby_state(
                state_dir,
                {
                    "status": "failed",
                    "error": f"Failed to open camera source: {camera_source}",
                    "camera_source": camera_source,
                    "pid": os.getpid(),
                    "process_group_id": os.getpgrp(),
                    "started_at": now_iso(),
                    "finished_at": now_iso(),
                },
            )
            return {
                "ok": False,
                "action": "standby_worker",
                "error": f"Failed to open camera source: {camera_source}",
                "state_path": str(state_path),
            }, 1

    missed = 0
    try:
        while not stop_flag["value"]:
            heartbeat = now_iso()
            ok = False
            if using_realsense or using_realsense_zmq:
                try:
                    capture_single_frame(
                        camera_source=camera_source,
                        output_path=latest_frame_path,
                        warmup_frames=1,
                        timeout_s=max(1.0, float(args.camera_timeout_s)),
                    )
                    ok = True
                except Exception:
                    ok = False
            else:
                ok, frame = cap.read()
                if ok and frame is not None:
                    cv2.imwrite(str(latest_frame_path), frame)
                else:
                    ok = False

            if ok:
                missed = 0
                save_standby_state(
                    state_dir,
                    {
                        "status": "running",
                        "error": None,
                        "camera_source": camera_source,
                        "pid": os.getpid(),
                        "process_group_id": os.getpgrp(),
                        "heartbeat_at": heartbeat,
                        "latest_frame_path": str(latest_frame_path),
                        "interval_s": interval_s,
                    },
                )
            else:
                missed += 1
                save_standby_state(
                    state_dir,
                    {
                        "status": "running",
                        "error": f"camera-read-failed (consecutive={missed})",
                        "camera_source": camera_source,
                        "pid": os.getpid(),
                        "process_group_id": os.getpgrp(),
                        "heartbeat_at": heartbeat,
                        "latest_frame_path": str(latest_frame_path),
                        "interval_s": interval_s,
                    },
                )
            time.sleep(interval_s)
    finally:
        if cap is not None:
            cap.release()
        save_standby_state(
            state_dir,
            {
                "status": "stopped",
                "camera_source": camera_source,
                "pid": os.getpid(),
                "process_group_id": os.getpgrp(),
                "finished_at": now_iso(),
                "latest_frame_path": str(latest_frame_path),
                "interval_s": interval_s,
            },
        )
        emit_progress("[real] standby worker stopped")

    return {
        "ok": True,
        "action": "standby_worker",
        "state_path": str(state_path),
    }, 0


def run_standby_start(args):
    state_dir = resolve_state_dir(args.state_dir)
    hardware_profile = resolve_runtime_hardware_profile(args)
    camera_source = hardware_profile.camera_source
    current = load_standby_state(state_dir)
    if current.get("status") == "running":
        pid = current.get("pid")
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                return {
                    "ok": True,
                    "action": "standby_start",
                    "standby": {**current, "worker_alive": True},
                }, 0
            except ProcessLookupError:
                pass

    worker_argv = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "standby_worker",
        "--state_dir",
        str(state_dir),
        "--camera_source",
        camera_source,
        "--camera_timeout_s",
        str(args.camera_timeout_s),
        "--interval_s",
        str(args.interval_s),
    ]
    proc = subprocess.Popen(
        worker_argv,
        cwd=str(Path(__file__).resolve().parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    standby = save_standby_state(
        state_dir,
        {
            "status": "running",
            "camera_source": camera_source,
            "pid": proc.pid,
            "process_group_id": proc.pid,
            "started_at": now_iso(),
            "heartbeat_at": None,
            "latest_frame_path": str(Path(state_dir) / "latest_frame.jpg"),
            "interval_s": float(args.interval_s),
            "error": None,
        },
    )
    return {
        "ok": True,
        "action": "standby_start",
        "standby": standby,
    }, 0


def run_standby_status(args):
    state_dir = resolve_state_dir(args.state_dir)
    standby = load_standby_state(state_dir)
    if not standby:
        return {
            "ok": False,
            "action": "standby_status",
            "error": "Standby worker has not been started",
        }, 1
    worker_alive = False
    pid = standby.get("pid")
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
            worker_alive = True
        except ProcessLookupError:
            worker_alive = False
    standby["worker_alive"] = worker_alive
    stale_timeout_s = max(3.0, float(args.standby_stale_timeout_s))
    now_ts = time.time()
    last_heartbeat_ts = (
        parse_iso_ts(standby.get("heartbeat_at"))
        or parse_iso_ts(standby.get("updated_at"))
        or parse_iso_ts(standby.get("started_at"))
    )
    stale_heartbeat = (
        standby.get("status") == "running"
        and worker_alive
        and last_heartbeat_ts is not None
        and (now_ts - last_heartbeat_ts) > stale_timeout_s
    )
    if stale_heartbeat:
        pgid = standby.get("process_group_id")
        try:
            if isinstance(pgid, int) and pgid > 0:
                os.killpg(pgid, signal.SIGKILL)
            elif isinstance(pid, int) and pid > 0:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        standby = save_standby_state(
            state_dir,
            {
                **standby,
                "status": "failed",
                "error": standby.get("error")
                or f"Standby worker stale: no frame heartbeat for {int(now_ts - last_heartbeat_ts)}s",
                "finished_at": now_iso(),
                "worker_alive": False,
            },
        )
    elif standby.get("status") == "running" and not worker_alive:
        standby = save_standby_state(
            state_dir,
            {
                **standby,
                "status": "failed",
                "error": standby.get("error") or "Standby worker exited unexpectedly",
                "finished_at": now_iso(),
            },
        )
    return {
        "ok": True,
        "action": "standby_status",
        "standby": standby,
    }, 0


def run_standby_stop(args):
    state_dir = resolve_state_dir(args.state_dir)
    standby = load_standby_state(state_dir)
    if not standby:
        return {
            "ok": True,
            "action": "standby_stop",
            "standby": {"status": "stopped", "note": "no-existing-worker"},
        }, 0

    pgid = standby.get("process_group_id")
    pid = standby.get("pid")
    try:
        if isinstance(pgid, int) and pgid > 0:
            os.killpg(pgid, signal.SIGKILL)
        elif isinstance(pid, int) and pid > 0:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    standby = save_standby_state(
        state_dir,
        {
            **standby,
            "status": "stopped",
            "finished_at": now_iso(),
            "error": standby.get("error"),
        },
    )
    return {
        "ok": True,
        "action": "standby_stop",
        "standby": standby,
    }, 0


def resolve_frame_for_request(args, state_dir, prefix):
    state_dir = Path(state_dir)
    hardware_profile = resolve_runtime_hardware_profile(args)
    if parse_boolish(args.use_standby_frame):
        standby = load_standby_state(state_dir)
        latest = (standby or {}).get("latest_frame_path", "")
        if latest and Path(latest).exists():
            return Path(latest).resolve(), True
    frame_path = build_frame_path(state_dir, prefix)
    captured = capture_single_frame(
        camera_source=hardware_profile.camera_source,
        output_path=frame_path,
        warmup_frames=max(1, int(args.camera_warmup_frames)),
        timeout_s=float(args.camera_timeout_s),
    )
    return captured.resolve(), False


def run_scene_qa(args):
    state_dir = resolve_state_dir(args.state_dir)
    preflight = run_preflight(args, require_vlm=True)
    camera_calibration = preflight.get("camera_calibration", {})
    if preflight["status"] != "ready" and not args.force:
        return {
            "ok": False,
            "action": "scene_qa",
            "error": "Real scene QA preflight failed",
            "preflight": preflight,
        }, 2
    if not args.question or not args.question.strip():
        return {
            "ok": False,
            "action": "scene_qa",
            "error": "A non-empty --question is required",
            "preflight": preflight,
        }, 2

    log_path = build_log_path(state_dir, "real_scene_qa")
    frame_path, used_standby_frame = resolve_frame_for_request(args, state_dir, "scene_qa")
    emit_progress(f"[real] scene_qa frame={frame_path} standby_frame={used_standby_frame}")

    image_bytes = Path(frame_path).read_bytes()
    answer, vlm_cfg = ask_image_question(
        image_bytes=image_bytes,
        question=args.question.strip(),
        default_model=args.model,
        system_prompt=(
            "You are an embodied robotics operator for Dobot. "
            "Answer only what is visible in the current frame and be explicit about uncertainty."
        ),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    trace_path = Path(frame_path).with_suffix(".qa_trace.json")
    atomic_write_json(
        trace_path,
        {
            "question": args.question.strip(),
            "answer": answer,
            "frame_path": str(frame_path),
            "used_standby_frame": used_standby_frame,
            "vlm": {
                "model": vlm_cfg["model"],
                "base_url": vlm_cfg["base_url"],
                "api_key_env": vlm_cfg["api_key_env"],
            },
            "camera_calibration": camera_calibration,
            "timestamp": now_iso(),
        },
    )
    log_path.write_text(
        "\n".join(
            [
                f"[real] question: {args.question.strip()}",
                f"[real] frame: {frame_path}",
                f"[real] used_standby_frame: {used_standby_frame}",
                "[vlm] assistant_response_begin",
                answer,
                "[vlm] assistant_response_end",
                f"[vlm] trace_path: {trace_path}",
                f"[real] camera_calibration_profile: {camera_calibration.get('profile')}",
                f"[real] camera_calibration_serial: {camera_calibration.get('serial')}",
                f"[real] camera_calibration_configured: {camera_calibration.get('configured')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "action": "scene_qa",
        "preflight": preflight,
        "result": {
            "question": args.question.strip(),
            "answer": answer,
            "snapshot_path": str(frame_path),
            "vlm_trace_path": str(trace_path),
            "used_standby_frame": used_standby_frame,
            "vlm": {
                "model": vlm_cfg["model"],
                "base_url": vlm_cfg["base_url"],
                "api_key_env": vlm_cfg["api_key_env"],
            },
            "camera_calibration": camera_calibration,
        },
        "run_log_path": str(log_path),
    }, 0


def build_execution_prompt(instruction, camera_calibration=None):
    calibration_block = build_calibration_prompt_block(camera_calibration)
    return (
        "You are planning a safe manipulator routine for a Dobot arm from one image.\n"
        "Use camera calibration context only as geometric reference; keep plan conservative.\n"
        "Return strict JSON only. Schema:\n"
        "{\n"
        '  "actions":[\n'
        "    {\"type\":\"movej\",\"joints\":[j1,j2,j3,j4,j5,j6]},\n"
        "    {\"type\":\"movel\",\"pose\":[x,y,z,rx,ry,rz]},\n"
        "    {\"type\":\"open_gripper\"},\n"
        "    {\"type\":\"close_gripper\"},\n"
        "    {\"type\":\"wait\",\"seconds\":0.5}\n"
        "  ],\n"
        '  "notes":"short safety notes"\n'
        "}\n"
        "Units: movel pose must use x/y/z in millimeters and rx/ry/rz in degrees.\n"
        "Keep action count <= 12. Avoid unsafe motion. "
        "If uncertain, return conservative actions with waits and no aggressive move.\n"
        f"{calibration_block}"
        f"User instruction: {instruction}"
    )


def _execute_action_list(adapter, actions, emit_prefix, execute_motion, action_interval_s=0.0):
    execution_records = []
    execution_error = None
    for idx, action in enumerate(actions, start=1):
        try:
            step_result = adapter.execute_action(action, execute_motion=bool(execute_motion))
            execution_records.append({"index": idx, "ok": True, "result": step_result})
            emit_progress(f"{emit_prefix} action[{idx}] {action.get('type')} ok")
            if bool(execute_motion) and float(action_interval_s) > 0.0 and idx < len(actions):
                emit_progress(f"{emit_prefix} action[{idx}] cooldown {float(action_interval_s):.1f}s")
                time.sleep(float(action_interval_s))
        except Exception as exc:
            execution_records.append({"index": idx, "ok": False, "error": str(exc), "action": action})
            execution_error = str(exc)
            emit_progress(f"{emit_prefix} action[{idx}] {action.get('type')} failed: {exc}")
            break
    return execution_records, execution_error


def _run_execute_rekep_vlm_stage_task(args, state_dir, preflight):
    task_name = read_string(getattr(args, "task", None), "pen").lower()
    default_instruction = "reorient the white pen and drop it upright into the black pen holder" if task_name == "pen" else "complete the requested manipulation task"
    instruction = read_string(args.instruction, default_instruction)
    action_interval_s = resolve_action_interval_s(args)
    vlm_stage_grasp_descend_m = resolve_rekep_vlm_stage_grasp_descend_m(args)
    hardware_profile = resolve_profile_from_preflight_or_args(args, preflight)
    camera_calibration = preflight.get("camera_calibration", {})
    if not camera_calibration.get("configured") and not args.force:
        return {
            "ok": False,
            "action": "execute",
            "error": "ReKep execute requires configured camera calibration",
            "preflight": preflight,
        }, 2

    log_path = build_log_path(state_dir, "real_execute_rekep")
    frame_dir = Path(state_dir) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parent

    from camera_factory import create_camera_adapter

    env = RealReKepEnv(
        state_dir=state_dir,
        hardware_profile=hardware_profile,
        camera_calibration=camera_calibration,
        camera_warmup_frames=args.camera_warmup_frames,
        camera_timeout_s=args.camera_timeout_s,
        camera_adapter=create_camera_adapter(hardware_profile=hardware_profile),
    )
    rgb0, depth0, planning_capture = env.capture_rgbd("execute_pen_planning", capture_realsense_rgbd)
    planning_frame_path = Path(planning_capture.frame_path)
    planning_depth_path = Path(planning_capture.depth_path)
    planning_capture_info = planning_capture.capture_info or {}
    task_schema_info = infer_task_keypoint_schema(
        image_path=planning_frame_path,
        instruction=instruction,
        model=args.model,
        temperature=min(0.2, float(args.temperature)),
        max_tokens=max(1400, int(args.max_tokens)),
    )
    schema = task_schema_info.get("keypoints") or []
    if not schema and task_name == "pen":
        planning_keypoint_obs = localize_pen_keypoints(
            image_path=planning_frame_path,
            depth_image=depth0,
            camera_calibration=camera_calibration,
            model=args.model,
            temperature=min(0.2, float(args.temperature)),
            max_tokens=max(1200, int(args.max_tokens)),
        )
    else:
        planning_keypoint_obs = localize_schema_keypoints(
            image_path=planning_frame_path,
            depth_image=depth0,
            camera_calibration=camera_calibration,
            keypoint_schema=schema,
            model=args.model,
            temperature=min(0.2, float(args.temperature)),
            max_tokens=max(1600, int(args.max_tokens)),
        )
    planning_overlay_path = planning_frame_path.with_name(planning_frame_path.stem + ".keypoints.png")
    draw_keypoints_overlay(planning_frame_path, planning_keypoint_obs, planning_overlay_path)
    planning_capture = env.add_overlay(planning_capture, str(planning_overlay_path), planning_keypoint_obs)
    runtime = RealTaskRuntime(
        task=task_name,
        instruction=instruction,
        planning_observation=RealObservation(
            frame_path=planning_capture.frame_path,
            depth_path=planning_capture.depth_path,
            overlay_path=planning_capture.overlay_path,
            capture_info=planning_capture.capture_info or {},
            keypoint_obs=planning_capture.keypoint_obs or {},
        ),
    )

    generator = RealTimeConstraintGenerator(
        repo_dir=repo_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=max(2048, int(args.max_tokens)),
    )
    live_program_info = generator.generate(
        image_path=planning_overlay_path,
        instruction=instruction,
        metadata={
            "init_keypoint_positions": [planning_keypoint_obs["keypoints_3d"][str(i)] for i in sorted(map(int, planning_keypoint_obs["keypoints_3d"].keys()))],
            "num_keypoints": len(planning_keypoint_obs.get("keypoints_3d", {})),
            "camera_profile": hardware_profile.camera_profile,
            "camera_serial": camera_calibration.get("serial"),
            "planning_frame_path": str(planning_frame_path),
            "planning_overlay_path": str(planning_overlay_path),
            "task_schema": task_schema_info,
        },
    )
    generation_validation = {}
    if isinstance(live_program_info.get("metadata"), dict):
        maybe_validation = live_program_info["metadata"].get("generation_validation")
        if isinstance(maybe_validation, dict):
            generation_validation = maybe_validation
    if generation_validation and not bool(generation_validation.get("ok", False)):
        issues = generation_validation.get("issues") if isinstance(generation_validation.get("issues"), list) else []
        issue_preview = "; ".join(str(x) for x in issues[:3]) if issues else "unknown consistency issue"
        if not args.force:
            log_lines = [
                f"[real] instruction: {instruction}",
                f"[real] task: {task_name}",
                f"[real] driver: {hardware_profile.robot_driver}",
                f"[real] execute_motion: {bool(args.execute_motion)}",
                f"[rekep] planning_frame_path: {planning_frame_path}",
                f"[rekep] planning_overlay_path: {planning_overlay_path}",
                f"[rekep] program_dir: {live_program_info.get('program_dir')}",
                f"[rekep] generation_trace_path: {live_program_info.get('trace_path')}",
                f"[rekep] generation_validation_ok: False",
                f"[rekep] generation_validation_attempts: {generation_validation.get('attempts', 1)}",
                f"[rekep] generation_validation_issues: {json.dumps(issues, ensure_ascii=False)}",
                f"[real] execution_error: generated ReKep program failed index consistency checks: {issue_preview}",
            ]
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return {
                "ok": False,
                "action": "execute",
                "error": f"generated ReKep program failed index consistency checks: {issue_preview}",
                "preflight": preflight,
                "run_log_path": str(log_path),
                "program_generation": live_program_info,
            }, 2
        emit_progress(f"[rekep][warning] generation validation failed but --force enabled: {issue_preview}")
    program = load_generated_program(live_program_info["program_dir"])
    runtime.generated_program_dir = program.get("program_dir", "")
    runtime.generated_program_info = live_program_info

    from robot_factory import create_robot_adapter

    adapter = create_robot_adapter(hardware_profile=hardware_profile)
    connection_info = adapter.connect()
    stage_results = []
    execution_error = None
    localization_model = None
    planner_model = None
    keypoint_tracker = RealKeypointTracker(smoothing_alpha=0.6, max_jump_m=0.18)
    grasp_state_estimator = RealGraspStateEstimator(closed_threshold=0.5)
    constraint_evaluator = RealConstraintEvaluator(constraint_tolerance=0.10)
    constraint_monitor = RealConstraintMonitor(position_tolerance_m=0.08, grasp_required_penalty=1.0)
    recovery_manager = RealRecoveryManager(max_stage_retries=2)
    stage_runner = RealStageRunner(
        env=env,
        adapter=adapter,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        camera_calibration=camera_calibration,
        emit_progress=emit_progress,
        ask_image_question=ask_image_question,
        parse_plan_from_vlm_text=parse_plan_from_vlm_text,
        keypoint_tracker=keypoint_tracker,
        constraint_monitor=constraint_monitor,
        recovery_manager=recovery_manager,
        constraint_evaluator=constraint_evaluator,
        grasp_state_estimator=grasp_state_estimator,
        grasp_descend_m=vlm_stage_grasp_descend_m,
    )
    try:
        for stage_info in program.get("stages", []):
            stage = int(stage_info.get("stage", 0))

            def _localizer(image_path, depth_image, calibration):
                if schema:
                    return localize_schema_keypoints(
                        image_path=image_path,
                        depth_image=depth_image,
                        camera_calibration=calibration,
                        keypoint_schema=schema,
                        model=args.model,
                        temperature=min(0.2, float(args.temperature)),
                        max_tokens=max(1600, int(args.max_tokens)),
                    )
                return localize_pen_keypoints(
                    image_path=image_path,
                    depth_image=depth_image,
                    camera_calibration=calibration,
                    model=args.model,
                    temperature=min(0.2, float(args.temperature)),
                    max_tokens=max(1200, int(args.max_tokens)),
                )

            stage_execution = stage_runner.execute_stage(
                prefix=f"execute_pen_stage{stage}",
                stage_info=stage_info,
                stage_index=stage,
                total_stages=len(program.get("stages", [])),
                capture_fn=capture_realsense_rgbd,
                keypoint_localizer=lambda image_path, depth_image, calibration: _localizer(image_path, depth_image, calibration),
                overlay_drawer=draw_keypoints_overlay,
                planner_prompt_builder=build_generic_stage_execution_prompt,
                instruction=instruction,
                execute_motion=bool(args.execute_motion),
                action_interval_s=action_interval_s,
            )
            runtime.add_stage_execution(stage_execution)
            keypoint_obs = stage_execution.observation.keypoint_obs if stage_execution.observation else {}
            localization_model = (keypoint_obs.get("vlm") or {}).get("model") or localization_model
            planner_model = stage_execution.plan.notes or planner_model
            stage_results.append(
                {
                    "stage": stage_execution.stage,
                    "frame_path": stage_execution.observation.frame_path if stage_execution.observation else "",
                    "depth_path": stage_execution.observation.depth_path if stage_execution.observation else "",
                    "overlay_path": stage_execution.observation.overlay_path if stage_execution.observation else "",
                    "keypoint_obs": keypoint_obs,
                    "object_schema": keypoint_obs.get("schema", []) if isinstance(keypoint_obs, dict) else [],
                    "capture_info": stage_execution.observation.capture_info if stage_execution.observation else {},
                    "instruction": instruction,
                    "stage_constraints": stage_execution.plan.stage_constraints if stage_execution.plan else {},
                    "grasp_state": (stage_execution.plan.stage_constraints or {}).get("grasp_state", {}) if stage_execution.plan else {},
                    "constraint_eval": (stage_execution.plan.stage_constraints or {}).get("constraint_eval", {}) if stage_execution.plan else {},
                    "monitor_result": (stage_execution.plan.stage_constraints or {}).get("monitor_result", {}) if stage_execution.plan else {},
                    "recovery_result": (stage_execution.plan.stage_constraints or {}).get("recovery_result", {}) if stage_execution.plan else {},
                    "plan_actions": stage_execution.plan.plan_actions if stage_execution.plan else [],
                    "plan_raw_output_path": stage_execution.plan.plan_raw_output_path if stage_execution.plan else "",
                    "execution_records": stage_execution.execution_records,
                    "execution_error": stage_execution.execution_error,
                }
            )
            if stage_execution.execution_error:
                execution_error = f"stage {stage} failed: {stage_execution.execution_error}"
                break
    finally:
        adapter.close()

    log_lines = [
        f"[real] instruction: {instruction}",
        f"[real] task: {task_name}",
        f"[real] driver: {hardware_profile.robot_driver}",
        f"[real] execute_motion: {bool(args.execute_motion)}",
        f"[real] action_interval_s: {action_interval_s if bool(args.execute_motion) else 0.0}",
        f"[rekep] vlm_stage_grasp_descend_m: {vlm_stage_grasp_descend_m}",
        f"[real] camera_calibration_profile: {camera_calibration.get('profile')}",
        f"[real] camera_calibration_serial: {camera_calibration.get('serial')}",
        f"[real] camera_calibration_configured: {camera_calibration.get('configured')}",
        f"[rekep] planning_frame_path: {planning_frame_path}",
        f"[rekep] planning_overlay_path: {planning_overlay_path}",
        f"[rekep] program_dir: {program.get('program_dir')}",
        f"[rekep] generation_trace_path: {live_program_info.get('trace_path')}",
        f"[rekep] generation_validation_ok: {(generation_validation or {}).get('ok', True)}",
        f"[rekep] generation_validation_attempts: {(generation_validation or {}).get('attempts', 1)}",
        f"[rekep] generation_validation_issues: {json.dumps((generation_validation or {}).get('issues', []), ensure_ascii=False)}",
        f"[rekep] stages: {len(stage_results)}",
    ]
    for stage_result in stage_results:
        log_lines.extend(
            [
                f"[rekep][stage={stage_result['stage']}] frame_path: {stage_result['frame_path']}",
                f"[rekep][stage={stage_result['stage']}] overlay_path: {stage_result['overlay_path']}",
                f"[rekep][stage={stage_result['stage']}] plan_raw_output_path: {stage_result['plan_raw_output_path']}",
                f"[rekep][stage={stage_result['stage']}] keypoints_3d: {json.dumps((stage_result.get('keypoint_obs') or {}).get('keypoints_3d', {}), ensure_ascii=False)}",
            ]
        )
    if execution_error:
        log_lines.append(f"[real] execution_error: {execution_error}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    payload = {
        "ok": execution_error is None,
        "action": "execute",
        "preflight": preflight,
        "result": {
            "task": task_name,
            "instruction": instruction,
            "driver": hardware_profile.robot_driver,
            "execute_motion": bool(args.execute_motion),
            "action_interval_s": action_interval_s if bool(args.execute_motion) else 0.0,
            "connection_info": connection_info,
            "rekep_program_dir": program.get("program_dir"),
            "rekep_mode": "live_stage_keypoints_constraints",
            "hardware_profile": hardware_profile.to_dict(),
            "planning_observation": {
                "frame_path": str(planning_frame_path),
                "depth_path": str(planning_depth_path),
                "overlay_path": str(planning_overlay_path),
                "capture_info": planning_capture_info,
                "keypoint_obs": planning_keypoint_obs,
                "program_generation": live_program_info,
            },
            "stage_results": stage_results,
            "vlm": {
                "localization_model": localization_model,
                "planner_model": planner_model,
                "constraint_model": (live_program_info.get("client_config") or {}).get("model"),
            },
            "vlm_stage": {
                "grasp_descend_m": vlm_stage_grasp_descend_m,
            },
            "program_generation_validation": generation_validation,
            "camera_calibration": camera_calibration,
        },
        "run_log_path": str(log_path),
    }
    if execution_error:
        payload["error"] = execution_error
        return payload, 1
    return payload, 0


def _run_execute_rekep_solver_task(args, state_dir, preflight):
    try:
        from real_solver_rekep import (
            build_real_solver_config,
            execute_solver_program,
            propose_candidate_keypoints,
        )
    except Exception as exc:
        return {
            "ok": False,
            "action": "execute",
            "error": (
                "ReKep solver mode dependencies are not available. "
                f"Install solver extras or switch --rekep_execution_mode=vlm_stage. detail={exc}"
            ),
            "preflight": preflight,
        }, 2

    task_name = read_string(getattr(args, "task", None), "rekep").lower()
    instruction = read_string(args.instruction, "complete the requested manipulation task")
    action_interval_s = resolve_action_interval_s(args)
    grasp_depth_m = resolve_rekep_grasp_depth_m(args)
    hardware_profile = resolve_profile_from_preflight_or_args(args, preflight)
    camera_calibration = preflight.get("camera_calibration", {})
    if not camera_calibration.get("configured") and not args.force:
        return {
            "ok": False,
            "action": "execute",
            "error": "ReKep execute requires configured camera calibration",
            "preflight": preflight,
        }, 2

    log_path = build_log_path(state_dir, "real_execute_rekep_solver")
    frame_prefix = "execute_rekep"
    repo_dir = Path(__file__).resolve().parent

    from camera_factory import create_camera_adapter

    env = RealReKepEnv(
        state_dir=state_dir,
        hardware_profile=hardware_profile,
        camera_calibration=camera_calibration,
        camera_warmup_frames=args.camera_warmup_frames,
        camera_timeout_s=args.camera_timeout_s,
        camera_adapter=create_camera_adapter(hardware_profile=hardware_profile),
    )
    rgb0, depth0, planning_capture = env.capture_rgbd(f"{frame_prefix}_planning", capture_realsense_rgbd)
    planning_frame_path = Path(planning_capture.frame_path)
    planning_depth_path = Path(planning_capture.depth_path)
    planning_capture_info = planning_capture.capture_info or {}

    solver_config = build_real_solver_config(arm="right", grasp_depth_m=grasp_depth_m)
    planning_keypoint_obs = propose_candidate_keypoints(
        rgb_bgr=rgb0,
        depth_image=depth0,
        camera_calibration=camera_calibration,
        output_prefix=planning_frame_path.stem,
        output_dir=planning_frame_path.parent,
        config=solver_config,
    )
    planning_overlay_path = Path(planning_keypoint_obs.get("overlay_path") or planning_frame_path)
    planning_capture = env.add_overlay(planning_capture, str(planning_overlay_path), planning_keypoint_obs)

    ordered_keypoint_ids = sorted(int(k) for k in (planning_keypoint_obs.get("keypoints_3d") or {}).keys())
    candidate_positions = [
        planning_keypoint_obs["keypoints_3d"][str(idx)]
        for idx in ordered_keypoint_ids
    ]

    generator = RealTimeConstraintGenerator(
        repo_dir=repo_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=max(2048, int(args.max_tokens)),
    )
    live_program_info = generator.generate(
        image_path=planning_overlay_path,
        instruction=instruction,
        metadata={
            "init_keypoint_positions": candidate_positions,
            "num_keypoints": len(candidate_positions),
            "camera_profile": hardware_profile.camera_profile,
            "camera_serial": camera_calibration.get("serial"),
            "planning_frame_path": str(planning_frame_path),
            "planning_overlay_path": str(planning_overlay_path),
            "task_schema": {"keypoints": planning_keypoint_obs.get("schema", [])},
            "candidate_proposal": {
                "overlay_path": planning_keypoint_obs.get("overlay_path"),
                "mask_overlay_path": planning_keypoint_obs.get("mask_overlay_path"),
                "proposal_debug_path": planning_keypoint_obs.get("proposal_debug_path"),
            },
        },
    )
    generation_validation = {}
    if isinstance(live_program_info.get("metadata"), dict):
        maybe_validation = live_program_info["metadata"].get("generation_validation")
        if isinstance(maybe_validation, dict):
            generation_validation = maybe_validation
    if generation_validation and not bool(generation_validation.get("ok", False)):
        issues = generation_validation.get("issues") if isinstance(generation_validation.get("issues"), list) else []
        issue_preview = "; ".join(str(x) for x in issues[:3]) if issues else "unknown consistency issue"
        if not args.force:
            log_lines = [
                f"[real] instruction: {instruction}",
                f"[real] task: {task_name}",
                f"[real] driver: {hardware_profile.robot_driver}",
                f"[real] execute_motion: {bool(args.execute_motion)}",
                f"[rekep] execution_mode: solver",
                f"[rekep] planning_frame_path: {planning_frame_path}",
                f"[rekep] planning_overlay_path: {planning_overlay_path}",
                f"[rekep] proposal_debug_path: {planning_keypoint_obs.get('proposal_debug_path')}",
                f"[rekep] program_dir: {live_program_info.get('program_dir')}",
                f"[rekep] generation_trace_path: {live_program_info.get('trace_path')}",
                f"[rekep] generation_validation_ok: False",
                f"[rekep] generation_validation_attempts: {generation_validation.get('attempts', 1)}",
                f"[rekep] generation_validation_issues: {json.dumps(issues, ensure_ascii=False)}",
                f"[real] execution_error: generated ReKep program failed index consistency checks: {issue_preview}",
            ]
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return {
                "ok": False,
                "action": "execute",
                "error": f"generated ReKep program failed index consistency checks: {issue_preview}",
                "preflight": preflight,
                "run_log_path": str(log_path),
                "program_generation": live_program_info,
            }, 2
        emit_progress(f"[rekep][warning] generation validation failed but --force enabled: {issue_preview}")

    program = load_generated_program(live_program_info["program_dir"])

    from robot_factory import create_robot_adapter

    adapter = create_robot_adapter(hardware_profile=hardware_profile)
    connection_info = adapter.connect()
    try:
        solver_result = execute_solver_program(
            program=program,
            planning_keypoint_obs=planning_keypoint_obs,
            adapter=adapter,
            execute_motion=bool(args.execute_motion),
            action_interval_s=action_interval_s,
            state_dir=state_dir,
            frame_prefix=frame_prefix,
            emit_progress=emit_progress,
            arm="right",
            grasp_depth_m=grasp_depth_m,
        )
    finally:
        adapter.close()

    stage_results = solver_result.get("stage_results", [])
    execution_error = solver_result.get("execution_error")
    log_lines = [
        f"[real] instruction: {instruction}",
        f"[real] task: {task_name}",
        f"[real] driver: {hardware_profile.robot_driver}",
        f"[real] execute_motion: {bool(args.execute_motion)}",
        f"[real] action_interval_s: {action_interval_s if bool(args.execute_motion) else 0.0}",
        f"[real] camera_calibration_profile: {camera_calibration.get('profile')}",
        f"[real] camera_calibration_serial: {camera_calibration.get('serial')}",
        f"[real] camera_calibration_configured: {camera_calibration.get('configured')}",
        f"[rekep] execution_mode: solver",
        f"[rekep] grasp_depth_m: {grasp_depth_m}",
        f"[rekep] planning_frame_path: {planning_frame_path}",
        f"[rekep] planning_overlay_path: {planning_overlay_path}",
        f"[rekep] planning_mask_overlay_path: {planning_keypoint_obs.get('mask_overlay_path')}",
        f"[rekep] proposal_debug_path: {planning_keypoint_obs.get('proposal_debug_path')}",
        f"[rekep] program_dir: {program.get('program_dir')}",
        f"[rekep] generation_trace_path: {live_program_info.get('trace_path')}",
        f"[rekep] generation_validation_ok: {(generation_validation or {}).get('ok', True)}",
        f"[rekep] generation_validation_attempts: {(generation_validation or {}).get('attempts', 1)}",
        f"[rekep] generation_validation_issues: {json.dumps((generation_validation or {}).get('issues', []), ensure_ascii=False)}",
        f"[rekep] stages: {len(stage_results)}",
    ]
    for stage_result in stage_results:
        plan_actions = stage_result.get("plan_actions") or []
        log_lines.extend(
            [
                f"[rekep][stage={stage_result['stage']}] overlay_path: {stage_result.get('overlay_path', '')}",
                f"[rekep][stage={stage_result['stage']}] plan_raw_output_path: {stage_result.get('plan_raw_output_path', '')}",
                f"[rekep][stage={stage_result['stage']}] action_count: {len(plan_actions)}",
                f"[rekep][stage={stage_result['stage']}] keypoints_3d: {json.dumps((stage_result.get('keypoint_obs') or {}).get('keypoints_3d', {}), ensure_ascii=False)}",
            ]
        )
    if execution_error:
        log_lines.append(f"[real] execution_error: {execution_error}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    payload = {
        "ok": execution_error is None,
        "action": "execute",
        "preflight": preflight,
        "result": {
            "task": task_name,
            "instruction": instruction,
            "driver": hardware_profile.robot_driver,
            "execute_motion": bool(args.execute_motion),
            "action_interval_s": action_interval_s if bool(args.execute_motion) else 0.0,
            "connection_info": connection_info,
            "rekep_program_dir": program.get("program_dir"),
            "rekep_mode": "live_solver_candidates_constraints",
            "hardware_profile": hardware_profile.to_dict(),
            "planning_observation": {
                "frame_path": str(planning_frame_path),
                "depth_path": str(planning_depth_path),
                "overlay_path": str(planning_overlay_path),
                "mask_overlay_path": planning_keypoint_obs.get("mask_overlay_path"),
                "capture_info": planning_capture_info,
                "keypoint_obs": planning_capture.keypoint_obs or planning_keypoint_obs,
                "program_generation": live_program_info,
            },
            "stage_results": stage_results,
            "solver": {
                "grasp_depth_m": grasp_depth_m,
                "config": solver_result.get("config"),
                "current_ee_pose": solver_result.get("current_ee_pose"),
            },
            "vlm": {
                "constraint_model": (live_program_info.get("client_config") or {}).get("model"),
            },
            "program_generation_validation": generation_validation,
            "camera_calibration": camera_calibration,
        },
        "run_log_path": str(log_path),
    }
    if execution_error:
        payload["error"] = execution_error
        return payload, 1
    return payload, 0


def _run_execute_rekep_live_task(args, state_dir, preflight):
    mode = resolve_rekep_execution_mode(args)
    if mode == "vlm_stage":
        return _run_execute_rekep_vlm_stage_task(args, state_dir, preflight)
    return _run_execute_rekep_solver_task(args, state_dir, preflight)


def run_execute(args):
    state_dir = resolve_state_dir(args.state_dir)
    preflight = run_preflight(args, require_vlm=True)
    camera_calibration = preflight.get("camera_calibration", {})
    action_interval_s = resolve_action_interval_s(args)
    hardware_profile = resolve_profile_from_preflight_or_args(args, preflight)
    if preflight["status"] != "ready" and not args.force:
        return {
            "ok": False,
            "action": "execute",
            "error": "Real execute preflight failed",
            "preflight": preflight,
        }, 2
    if not args.instruction or not args.instruction.strip():
        return {
            "ok": False,
            "action": "execute",
            "error": "A non-empty --instruction is required",
            "preflight": preflight,
        }, 2

    task_name = read_string(getattr(args, "task", None), "pen").lower()
    if camera_calibration.get("configured"):
        return _run_execute_rekep_live_task(args, state_dir, preflight)

    log_path = build_log_path(state_dir, "real_execute")
    frame_path, used_standby_frame = resolve_frame_for_request(args, state_dir, "execute")
    instruction = args.instruction.strip()
    emit_progress(f"[real] execute instruction={instruction!r} frame={frame_path}")

    raw_answer, vlm_cfg = ask_image_question(
        image_bytes=Path(frame_path).read_bytes(),
        question=build_execution_prompt(instruction, camera_calibration=camera_calibration),
        default_model=args.model,
        system_prompt=(
            "You are a precise robot planner. Output strict JSON only and avoid markdown fences."
        ),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    actions, raw_json_like = parse_plan_from_vlm_text(raw_answer)
    plan_path = Path(frame_path).with_suffix(".plan_raw.txt")
    plan_path.write_text(raw_answer + "\n", encoding="utf-8")
    emit_progress(f"[real] planned_actions={len(actions)} driver={hardware_profile.robot_driver}")

    from robot_factory import create_robot_adapter

    adapter = create_robot_adapter(hardware_profile=hardware_profile)
    connection_info = adapter.connect()
    try:
        execution_records, execution_error = _execute_action_list(
            adapter,
            actions,
            emit_prefix="[real]",
            execute_motion=bool(args.execute_motion),
            action_interval_s=action_interval_s,
        )
    finally:
        adapter.close()

    log_lines = [
        f"[real] instruction: {instruction}",
        f"[real] frame: {frame_path}",
        f"[real] used_standby_frame: {used_standby_frame}",
        f"[real] driver: {hardware_profile.robot_driver}",
        f"[real] execute_motion: {bool(args.execute_motion)}",
        f"[real] action_interval_s: {action_interval_s if bool(args.execute_motion) else 0.0}",
        f"[real] planned_actions: {len(actions)}",
        f"[real] camera_calibration_profile: {camera_calibration.get('profile')}",
        f"[real] camera_calibration_serial: {camera_calibration.get('serial')}",
        f"[real] camera_calibration_configured: {camera_calibration.get('configured')}",
        "[vlm] planner_raw_begin",
        raw_answer,
        "[vlm] planner_raw_end",
        f"[vlm] plan_raw_path: {plan_path}",
    ]
    if execution_error:
        log_lines.append(f"[real] execution_error: {execution_error}")
    log_lines.append(f"[real] executed_steps: {len(execution_records)}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    payload = {
        "ok": execution_error is None,
        "action": "execute",
        "preflight": preflight,
        "result": {
            "instruction": instruction,
            "driver": hardware_profile.robot_driver,
            "execute_motion": bool(args.execute_motion),
            "action_interval_s": action_interval_s if bool(args.execute_motion) else 0.0,
            "frame_path": str(frame_path),
            "used_standby_frame": used_standby_frame,
            "connection_info": connection_info,
            "plan_actions": actions,
            "plan_raw_output_path": str(plan_path),
            "execution_records": execution_records,
            "hardware_profile": hardware_profile.to_dict(),
            "vlm": {
                "model": vlm_cfg["model"],
                "base_url": vlm_cfg["base_url"],
                "api_key_env": vlm_cfg["api_key_env"],
            },
            "camera_calibration": camera_calibration,
        },
        "run_log_path": str(log_path),
    }
    if execution_error:
        payload["error"] = execution_error
        return payload, 1
    return payload, 0


def job_state_path(state_dir, job_id):
    return Path(state_dir) / "jobs" / f"{job_id}.json"


def save_job_state(path, payload):
    payload = dict(payload)
    payload["updated_at"] = now_iso()
    atomic_write_json(path, payload)
    return payload


def update_job_state(path, **updates):
    current = load_json_if_exists(path) or {}
    current.update(updates)
    return save_job_state(path, current)


def latest_job_path(state_dir):
    jobs_dir = Path(state_dir) / "jobs"
    jobs = sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jobs[0] if jobs else None


def build_execute_worker_argv(args, job_id, state_dir):
    hardware_profile = resolve_runtime_hardware_profile(args)
    argv = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "execute_worker",
        "--job_id",
        job_id,
        "--state_dir",
        str(state_dir),
        "--instruction",
        args.instruction or "",
        "--task",
        read_string(getattr(args, "task", None), "rekep"),
        "--camera_profile",
        resolve_camera_profile(args),
        "--camera_serial",
        resolve_camera_serial(args),
        "--camera_source",
        hardware_profile.camera_source,
        "--dobot_settings_ini",
        str(resolve_dobot_settings_ini(args)),
        "--camera_extrinsic_script",
        str(resolve_camera_extrinsic_script(args)),
        "--realsense_calib_dir",
        str(resolve_realsense_calib_dir(args)),
        "--robot_family",
        read_string(hardware_profile.robot_family, "dobot"),
        "--robot_driver",
        read_string(hardware_profile.robot_driver),
        "--dobot_driver",
        hardware_profile.robot_driver,
        "--dobot_host",
        hardware_profile.robot_host,
        "--xtrainer_sdk_dir",
        str(hardware_profile.xtrainer_sdk_dir),
        "--camera_timeout_s",
        str(args.camera_timeout_s),
        "--camera_warmup_frames",
        str(args.camera_warmup_frames),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--action_interval_s",
        str(resolve_action_interval_s(args)),
        "--rekep_execution_mode",
        resolve_rekep_execution_mode(args),
        "--rekep_grasp_depth_m",
        str(resolve_rekep_grasp_depth_m(args)),
        "--rekep_vlm_stage_grasp_descend_m",
        str(resolve_rekep_vlm_stage_grasp_descend_m(args)),
    ]
    if read_string(hardware_profile.robot_host):
        argv.extend(["--robot_host", read_string(hardware_profile.robot_host)])
        argv.extend(["--dobot_host", read_string(hardware_profile.robot_host)])
    if hardware_profile.robot_port is not None:
        argv.extend(["--robot_port", str(int(hardware_profile.robot_port))])
        argv.extend(["--dobot_port", str(int(hardware_profile.robot_port))])
    if hardware_profile.robot_move_port is not None:
        argv.extend(["--robot_move_port", str(int(hardware_profile.robot_move_port))])
        argv.extend(["--dobot_move_port", str(int(hardware_profile.robot_move_port))])
    if args.execute_motion:
        argv.append("--execute_motion")
    if args.use_standby_frame:
        argv.append("--use_standby_frame")
    if args.force:
        argv.append("--force")
    return argv


def run_execute_background(args):
    state_dir = resolve_state_dir(args.state_dir)
    preflight = run_preflight(args, require_vlm=True)
    hardware_profile = resolve_profile_from_preflight_or_args(args, preflight)
    if preflight["status"] != "ready" and not args.force:
        return {
            "ok": False,
            "action": "execute_background",
            "error": "Real execute preflight failed",
            "preflight": preflight,
        }, 2
    if not args.instruction or not args.instruction.strip():
        return {
            "ok": False,
            "action": "execute_background",
            "error": "A non-empty --instruction is required",
            "preflight": preflight,
        }, 2

    job_id = args.job_id or f"real-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = job_state_path(state_dir, job_id)
    job_payload = save_job_state(
        path,
        {
            "job_id": job_id,
            "status": "queued",
            "instruction": args.instruction.strip(),
            "camera_source": hardware_profile.camera_source,
            "driver": hardware_profile.robot_driver,
            "robot_ip": hardware_profile.robot_host,
            "dashboard_port": int(hardware_profile.robot_port) if hardware_profile.robot_port is not None else None,
            "move_port": int(hardware_profile.robot_move_port) if hardware_profile.robot_move_port is not None else None,
            "xtrainer_sdk_dir": str(hardware_profile.xtrainer_sdk_dir),
            "hardware_profile": hardware_profile.to_dict(),
            "execute_motion": bool(args.execute_motion),
            "action_interval_s": resolve_action_interval_s(args) if bool(args.execute_motion) else 0.0,
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "process_group_id": None,
            "result": None,
            "error": None,
            "exit_code": None,
            "preflight": preflight,
            "state_path": str(path),
        },
    )

    worker_argv = build_execute_worker_argv(args, job_id, state_dir)
    worker_log_path = Path(state_dir) / "logs" / f"{job_id}.worker.log"
    with worker_log_path.open("ab") as worker_log:
        proc = subprocess.Popen(
            worker_argv,
            cwd=str(Path(__file__).resolve().parent),
            stdin=subprocess.DEVNULL,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    job_payload = update_job_state(
        path,
        status="running",
        started_at=now_iso(),
        pid=proc.pid,
        process_group_id=proc.pid,
        worker_log_path=str(worker_log_path),
    )
    return {
        "ok": True,
        "action": "execute_background",
        "job": job_payload,
    }, 0


def run_execute_worker(args):
    state_dir = resolve_state_dir(args.state_dir)
    path = job_state_path(state_dir, args.job_id)
    update_job_state(
        path,
        status="running",
        started_at=(load_json_if_exists(path) or {}).get("started_at") or now_iso(),
        pid=os.getpid(),
        process_group_id=os.getpgrp(),
        error=None,
    )
    payload, exit_code = run_execute(args)
    latest = load_json_if_exists(path) or {}
    if latest.get("status") == "cancelled":
        return {
            "ok": False,
            "action": "execute_worker",
            "error": "Job was cancelled",
            "job": latest,
        }, 1
    update_job_state(
        path,
        status="succeeded" if exit_code == 0 and payload.get("ok") else "failed",
        finished_at=now_iso(),
        result=payload.get("result"),
        error=None if exit_code == 0 and payload.get("ok") else payload.get("error", "unknown error"),
        exit_code=exit_code,
        run_log_path=payload.get("run_log_path"),
    )
    return payload, exit_code


def run_job_status(args):
    state_dir = resolve_state_dir(args.state_dir)
    path = job_state_path(state_dir, args.job_id) if args.job_id else latest_job_path(state_dir)
    if path is None or not Path(path).exists():
        return {
            "ok": False,
            "action": "job_status",
            "error": "No real-robot jobs were found" if not args.job_id else f"Unknown job id: {args.job_id}",
        }, 1
    payload = load_json_if_exists(path) or {}
    pid = payload.get("pid")
    worker_alive = False
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
            worker_alive = True
        except ProcessLookupError:
            worker_alive = False
    payload["worker_alive"] = worker_alive
    if payload.get("status") == "running" and not worker_alive:
        payload = update_job_state(
            path,
            status="failed",
            finished_at=now_iso(),
            error=payload.get("error") or "Background worker exited unexpectedly",
            exit_code=payload.get("exit_code") or 1,
        )
    return {
        "ok": True,
        "action": "job_status",
        "job": payload,
        "log_tail": read_log_tail(payload.get("run_log_path") or payload.get("worker_log_path")),
    }, 0


def run_job_cancel(args):
    state_dir = resolve_state_dir(args.state_dir)
    path = job_state_path(state_dir, args.job_id) if args.job_id else latest_job_path(state_dir)
    if path is None or not Path(path).exists():
        return {
            "ok": False,
            "action": "job_cancel",
            "error": "No real-robot jobs were found" if not args.job_id else f"Unknown job id: {args.job_id}",
        }, 1
    payload = load_json_if_exists(path) or {}
    if payload.get("status") in {"succeeded", "failed", "cancelled"}:
        return {
            "ok": True,
            "action": "job_cancel",
            "job": payload,
        }, 0
    pgid = payload.get("process_group_id")
    pid = payload.get("pid")
    try:
        if isinstance(pgid, int) and pgid > 0:
            os.killpg(pgid, signal.SIGKILL)
        elif isinstance(pid, int) and pid > 0:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    payload = update_job_state(
        path,
        status="cancelled",
        finished_at=now_iso(),
        error="Cancelled by user",
        exit_code=130,
    )
    return {
        "ok": True,
        "action": "job_cancel",
        "job": payload,
        "log_tail": read_log_tail(payload.get("run_log_path")),
    }, 0


def longrun_command_path(state_dir, job_id):
    return Path(state_dir) / LONGRUN_COMMAND_DIRNAME / f"{job_id}.jsonl"


def latest_longrun_job_path(state_dir):
    jobs_dir = Path(state_dir) / "jobs"
    for path in sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = load_json_if_exists(path) or {}
        if payload.get("kind") == "longrun":
            return path
    return None


def read_longrun_commands_since(command_path, offset):
    records = read_jsonl(command_path)
    safe_offset = max(0, int(offset or 0))
    if safe_offset >= len(records):
        return [], len(records)
    return records[safe_offset:], len(records)


def split_subtasks_text(instruction):
    text = read_string(instruction)
    if not text:
        return []
    candidates = []
    for chunk in re.split(r"[。\n；;]+", text):
        cleaned = chunk.strip(" ，,")
        if not cleaned:
            continue
        for nested in re.split(r"\s*(?:然后|接着|再|并且|并)\s*", cleaned):
            final = nested.strip(" ，,")
            if final:
                candidates.append(final)
    deduped = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped or [text]


def infer_longrun_command_from_text(text):
    raw = read_string(text).lower()
    if not raw:
        return ""
    if any(token in raw for token in ["暂停", "pause", "stop for now", "先停"]):
        return "pause"
    if any(token in raw for token in ["继续", "resume", "continue", "接着"]):
        return "resume"
    if any(token in raw for token in ["停止任务", "终止", "结束任务", "stop task", "cancel task"]):
        return "stop"
    if any(token in raw for token in ["跳过", "skip"]):
        return "skip_current"
    if any(token in raw for token in ["换成", "改成", "别抓", "instead", "不要抓"]):
        return "replace_current"
    if any(token in raw for token in ["重排", "替换任务", "new plan", "replace plan"]):
        return "replace_plan"
    if any(token in raw for token in ["追加", "增加", "append", "add subtask"]):
        return "append_subtask"
    return "append_subtask"


def normalize_longrun_command(command, command_text):
    normalized = read_string(command).lower()
    if not normalized:
        normalized = infer_longrun_command_from_text(command_text)
    if normalized == "continue":
        normalized = "resume"
    allowed = {
        "pause",
        "resume",
        "stop",
        "append_subtask",
        "replace_plan",
        "replace_current",
        "skip_current",
    }
    if normalized not in allowed:
        return ""
    return normalized


def infer_stage_type_from_runtime(stage_runtime):
    stage_runtime = as_dict(stage_runtime)
    stage_constraints = as_dict(stage_runtime.get("stage_constraints"))
    grasp_state = as_dict(stage_runtime.get("grasp_state"))
    plan_actions = stage_runtime.get("plan_actions") if isinstance(stage_runtime.get("plan_actions"), list) else []

    grasp_keypoint = stage_constraints.get("grasp_keypoint")
    release_keypoint = stage_constraints.get("release_keypoint")
    grasped_keypoints = grasp_state.get("grasped_keypoints") if isinstance(grasp_state.get("grasped_keypoints"), list) else []

    if release_keypoint not in (-1, None, ""):
        return "release"
    if grasp_keypoint not in (-1, None, ""):
        return "grasp"
    if grasped_keypoints:
        return "transport"

    action_types = {read_string(action.get("type")).lower() for action in plan_actions if isinstance(action, dict)}
    if any(token in action_types for token in {"close_gripper", "grasp", "pick", "pick_grasp"}):
        return "grasp"
    if any(token in action_types for token in {"open_gripper", "release", "drop", "place_release"}):
        return "release"
    return "move"



def infer_operation_type(stage_runtime):
    stage_runtime = as_dict(stage_runtime)
    instruction = read_string(stage_runtime.get("instruction")).lower()
    keypoint_obs = as_dict(stage_runtime.get("keypoint_obs"))
    schema = keypoint_obs.get("schema") if isinstance(keypoint_obs.get("schema"), list) else []
    object_names = " ".join(read_string(item.get("object")).lower() for item in schema if isinstance(item, dict))
    text = f"{instruction} {object_names}"
    if any(token in text for token in ["slide", "sliding", "push", "pull", "drawer", "slider", "滑", "推", "拉", "抽屉"]):
        return "slide"
    if any(token in text for token in ["rotate", "turn", "twist", "knob", "旋", "拧"]):
        return "rotate"
    if any(token in text for token in ["hinge", "door", "open lid", "open door", "铰", "开门", "开盖"]):
        return "hinge"
    return "pick_place"



def evaluate_sliding_progress(stage_runtime):
    stage_runtime = as_dict(stage_runtime)
    keypoint_obs = as_dict(stage_runtime.get("keypoint_obs"))
    keypoints_3d = keypoint_obs.get("keypoints_3d") if isinstance(keypoint_obs.get("keypoints_3d"), dict) else {}
    schema = keypoint_obs.get("schema") if isinstance(keypoint_obs.get("schema"), list) else []
    stage_constraints = as_dict(stage_runtime.get("stage_constraints"))
    initial_keypoints_3d = stage_constraints.get("initial_keypoints_3d") if isinstance(stage_constraints.get("initial_keypoints_3d"), dict) else {}
    if len(keypoints_3d) < 2 or not schema:
        return {"state": "unknown", "ok": False, "reason": "insufficient keypoints for slide evaluation"}

    roles = infer_object_roles_from_schema(schema, grasp_state=stage_runtime.get("grasp_state"), instruction=stage_runtime.get("instruction"))
    manipulated = read_string(roles.get("slider_object") or roles.get("manipulated_object"))
    rail = read_string(roles.get("rail_object") or roles.get("target_container"))

    current_points = []
    initial_points = []
    manipulated_current = []
    rail_current = []
    for item in schema:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id"))
        point = keypoints_3d.get(key)
        if isinstance(point, list) and len(point) >= 3:
            point_np = np.asarray(point[:3], dtype=float)
            current_points.append(point_np)
            if read_string(item.get("object")) == manipulated:
                manipulated_current.append(point_np)
            elif read_string(item.get("object")) == rail:
                rail_current.append(point_np)
        point0 = initial_keypoints_3d.get(key)
        if isinstance(point0, list) and len(point0) >= 3:
            initial_points.append(np.asarray(point0[:3], dtype=float))
    if len(current_points) < 2:
        return {"state": "unknown", "ok": False, "reason": "not enough localized points"}

    arr = np.stack(current_points, axis=0)
    spans = np.ptp(arr, axis=0)
    dominant_axis = int(np.argmax(spans[:2]))
    current_span = float(spans[dominant_axis])
    stable_cross_axis = float(spans[1 - dominant_axis]) if dominant_axis in (0, 1) else 0.0

    relative_progress = 0.0
    if len(initial_points) == len(current_points) and initial_points:
        init_arr = np.stack(initial_points, axis=0)
        current_center = np.mean(arr, axis=0)
        initial_center = np.mean(init_arr, axis=0)
        relative_progress = float(abs(current_center[dominant_axis] - initial_center[dominant_axis]))

    manipulated_progress = relative_progress
    rail_offset = 0.0
    target_displacement = 0.05
    if manipulated_current:
        manipulated_center = np.mean(np.stack(manipulated_current, axis=0), axis=0)
        manipulated_progress = float(abs(manipulated_center[dominant_axis] - np.mean(arr, axis=0)[dominant_axis]))
    if rail_current and manipulated_current:
        rail_center = np.mean(np.stack(rail_current, axis=0), axis=0)
        manipulated_center = np.mean(np.stack(manipulated_current, axis=0), axis=0)
        rail_offset = float(abs(manipulated_center[1 - dominant_axis] - rail_center[1 - dominant_axis]))
    slide_ok = (relative_progress > target_displacement or current_span > target_displacement) and stable_cross_axis < 0.08 and rail_offset < 0.10
    return {
        "state": "slide_progress_ok" if slide_ok else "slide_insufficient_progress",
        "ok": slide_ok,
        "reason": f"axis={dominant_axis}, relative_progress={relative_progress:.4f}, target_displacement={target_displacement:.4f}, current_span={current_span:.4f}, cross_axis_span={stable_cross_axis:.4f}, rail_offset={rail_offset:.4f}",
        "roles": {"slider_object": manipulated, "rail_object": rail},
        "metrics": {
            "dominant_axis": dominant_axis,
            "relative_progress": relative_progress,
            "manipulated_progress": manipulated_progress,
            "target_displacement": target_displacement,
            "current_span": current_span,
            "cross_axis_span": stable_cross_axis,
            "rail_offset": rail_offset,
        },
    }



def infer_longrun_fault_type(monitor, stage_runtime):
    monitor = as_dict(monitor)
    stage_runtime = as_dict(stage_runtime)
    status = read_string(monitor.get("status"), "unknown").lower()
    if status in {"dropped", "grasp_failed", "target_lost", "blocked", "monitor_failed"}:
        return status

    operation_type = infer_operation_type(stage_runtime)
    if operation_type == "slide":
        slide_eval = evaluate_sliding_progress(stage_runtime)
        if status == "deviation" and not slide_eval.get("ok"):
            return "slide_insufficient_progress"
        if status == "blocked":
            return "slide_blocked"

    stage_type = infer_stage_type_from_runtime(stage_runtime)
    recovery_result = as_dict(stage_runtime.get("recovery_result"))
    recovery_action = read_string(recovery_result.get("action")).lower()
    grasp_state = as_dict(stage_runtime.get("grasp_state"))
    stage_constraints = as_dict(stage_runtime.get("stage_constraints"))
    keypoint_obs = as_dict(stage_runtime.get("keypoint_obs"))
    schema = keypoint_obs.get("schema") if isinstance(keypoint_obs.get("schema"), list) else []
    grasped_keypoints = grasp_state.get("grasped_keypoints") if isinstance(grasp_state.get("grasped_keypoints"), list) else []
    object_bindings = grasp_state.get("object_bindings") if isinstance(grasp_state.get("object_bindings"), dict) else {}
    release_keypoint = stage_constraints.get("release_keypoint")

    bound_objects = {str(v) for v in object_bindings.values() if str(v)}
    release_object = ""
    if release_keypoint not in (-1, None, ""):
        for item in schema:
            if not isinstance(item, dict):
                continue
            try:
                if int(item.get("id", -1)) == int(release_keypoint):
                    release_object = str(item.get("object", ""))
                    break
            except Exception:
                continue

    if stage_type == "grasp" and recovery_action in {"retry_stage", "replan", "reobserve"}:
        return "grasp_failed"
    if stage_type == "transport" and not grasped_keypoints and bound_objects:
        return "dropped"
    if stage_type == "transport" and not grasped_keypoints and status in {"deviation", "unknown"}:
        return "dropped"
    if stage_type == "release" and release_object and release_object in bound_objects:
        return "release_failed"
    if stage_type == "release" and not bound_objects and status in {"deviation", "unknown"}:
        return "released_or_placed"
    if status == "deviation":
        return "deviation"
    return "execution_failed"



def choose_longrun_recovery_policy(fault_type, stage_type):
    fault = read_string(fault_type, "execution_failed").lower()
    stage = read_string(stage_type).lower()
    default_policy = {
        "mode": "replace_current",
        "retry_budget": 2,
        "cancel_running_job": True,
        "next_action": "replan",
    }
    policy_matrix = {
        ("grasp_failed", "grasp"): {"mode": "replace_current", "retry_budget": 3, "cancel_running_job": True, "next_action": "reobserve_then_regrasp"},
        ("target_lost", "grasp"): {"mode": "replace_current", "retry_budget": 3, "cancel_running_job": True, "next_action": "reobserve_target"},
        ("slide_insufficient_progress", "move"): {"mode": "replace_plan", "retry_budget": 2, "cancel_running_job": False, "next_action": "increase_slide_progress"},
        ("slide_blocked", "move"): {"mode": "replace_plan", "retry_budget": 1, "cancel_running_job": True, "next_action": "replan_slide_path"},
        ("dropped", "transport"): {"mode": "replace_current", "retry_budget": 2, "cancel_running_job": True, "next_action": "reacquire_and_regrasp"},
        ("deviation", "transport"): {"mode": "replace_current", "retry_budget": 2, "cancel_running_job": False, "next_action": "return_to_safe_transport_pose"},
        ("blocked", "transport"): {"mode": "replace_plan", "retry_budget": 1, "cancel_running_job": True, "next_action": "replan_around_obstacle"},
        ("deviation", "release"): {"mode": "replace_current", "retry_budget": 2, "cancel_running_job": False, "next_action": "realign_release_pose"},
        ("blocked", "release"): {"mode": "replace_current", "retry_budget": 1, "cancel_running_job": True, "next_action": "backoff_and_realign_release"},
        ("release_failed", "release"): {"mode": "replace_plan", "retry_budget": 1, "cancel_running_job": True, "next_action": "backoff_and_realign_release"},
        ("released_or_placed", "release"): {"mode": "append_subtask", "retry_budget": 1, "cancel_running_job": False, "next_action": "verify_final_placement"},
        ("execution_failed", "move"): {"mode": "replace_current", "retry_budget": 2, "cancel_running_job": True, "next_action": "retry_motion_plan"},
        ("blocked", "move"): {"mode": "skip_current", "retry_budget": 0, "cancel_running_job": True, "next_action": "skip_subtask"},
    }
    return {
        **default_policy,
        **policy_matrix.get((fault, stage), {}),
        "fault_type": fault,
        "stage_type": stage,
    }



def apply_longrun_recovery_policy(*, policy, subtask, reason, stage_type, fault_type):
    policy = as_dict(policy)
    mode = read_string(policy.get("mode"), "replace_current")
    next_action = read_string(policy.get("next_action"), "replan")
    recovery_instruction = build_recovery_instruction(fault_type, subtask, reason, stage_type=stage_type)
    result = {
        "mode": mode,
        "next_action": next_action,
        "replacement_subtask": recovery_instruction,
        "replacement_plan": [],
        "skip_current": False,
        "append_subtasks": [],
    }

    if next_action == "reobserve_then_regrasp":
        result["replacement_plan"] = [
            "重新观察目标和夹爪的相对位姿，确认最佳抓取点。",
            recovery_instruction,
        ]
    elif next_action == "reobserve_target":
        result["replacement_plan"] = [
            "重新搜索并确认目标当前可见、可抓取。",
            recovery_instruction,
        ]
    elif next_action == "reacquire_and_regrasp":
        result["replacement_plan"] = [
            "重新定位掉落目标。",
            "重新抓取目标并确认抓稳。",
            recovery_instruction,
        ]
    elif next_action == "return_to_safe_transport_pose":
        result["replacement_plan"] = [
            "先回到安全搬运位姿并稳定夹持目标。",
            recovery_instruction,
        ]
    elif next_action == "increase_slide_progress":
        result["replacement_plan"] = [
            "重新建立稳定接触，并确认主要滑动方向。",
            "沿滑动主轴继续施加小步长位移，避免偏离轨迹。",
            recovery_instruction,
        ]
    elif next_action == "replan_slide_path":
        result["replacement_plan"] = [
            "重新观察滑轨、接触点和阻挡关系。",
            "规划新的滑动接触路径，优先避免横向偏移。",
            recovery_instruction,
        ]
    elif next_action == "replan_around_obstacle":
        result["replacement_plan"] = [
            "重新观察阻挡关系并规划绕障路径。",
            recovery_instruction,
        ]
    elif next_action == "realign_release_pose":
        result["replacement_plan"] = [
            "重新对齐释放位姿，确认目标与放置区域匹配。",
            recovery_instruction,
        ]
    elif next_action == "backoff_and_realign_release":
        result["replacement_plan"] = [
            "先后退到安全释放前位姿。",
            "重新对齐放置点与目标姿态。",
            recovery_instruction,
        ]
    elif next_action == "verify_final_placement":
        result["append_subtasks"] = [
            "检查目标是否已经放置到正确位置，并确认姿态满足任务要求。",
        ]
        result["replacement_plan"] = [recovery_instruction]
    elif next_action == "retry_motion_plan":
        result["replacement_plan"] = [recovery_instruction]
    elif next_action == "skip_subtask":
        result["skip_current"] = True
        result["replacement_subtask"] = ""
        result["replacement_plan"] = []

    if mode == "skip_current":
        result["skip_current"] = True
        result["replacement_subtask"] = ""
        result["replacement_plan"] = []
    elif mode == "append_subtask":
        result["append_subtasks"] = result["replacement_plan"] or [recovery_instruction]
        result["replacement_subtask"] = read_string(subtask)
        result["replacement_plan"] = []
    elif mode == "replace_plan":
        result["replacement_subtask"] = ""
    elif not result["replacement_plan"]:
        result["replacement_plan"] = [recovery_instruction]
    return result



def build_recovery_instruction(fault_type, subtask, reason="", stage_type=""):
    base = read_string(subtask) or "继续执行当前目标"
    fault = read_string(fault_type, "execution_failed").lower()
    stage = read_string(stage_type).lower()
    reason_text = read_string(reason)
    templates = {
        "dropped": "目标疑似掉落。先重新定位并抓取目标，然后继续：{base}",
        "grasp_failed": "抓取失败。调整抓取位姿并重抓，然后继续：{base}",
        "target_lost": "目标丢失。先重新搜索目标，再执行：{base}",
        "blocked": "路径受阻。绕开阻挡后执行：{base}",
        "deviation": "出现偏差。先纠偏到目标位姿，再执行：{base}",
        "release_failed": "释放阶段失败。先重新对齐放置位姿，确认目标仍受控，再重新释放：{base}",
        "released_or_placed": "目标已释放或已基本放到位。先确认最终放置状态，再决定是否继续后续目标：{base}",
        "slide_insufficient_progress": "滑动进展不足。先重新贴合接触点，沿滑动方向继续推进：{base}",
        "slide_blocked": "滑动路径受阻。先重新观察滑轨/接触关系，再规划新的滑动路径：{base}",
        "execution_failed": "动作执行失败。保守重试并继续：{base}",
        "monitor_failed": "监控异常。先恢复可观测状态，再执行：{base}",
    }
    stage_templates = {
        ("grasp_failed", "grasp"): "当前卡在抓取阶段。先重新观察目标与夹爪对位，微调抓取位姿并重抓，然后继续：{base}",
        ("deviation", "transport"): "当前卡在搬运阶段。保持已抓稳目标，先回到安全搬运位姿，再沿更稳路径继续：{base}",
        ("blocked", "transport"): "当前卡在搬运阶段且路径受阻。保持抓取，先抬高避障并绕开阻挡，再继续：{base}",
        ("slide_insufficient_progress", "move"): "当前卡在滑动阶段。保持稳定接触，重新对齐滑动方向并增加有效位移，然后继续：{base}",
        ("slide_blocked", "move"): "当前卡在滑动阶段且路径受阻。先回到安全接触位姿，重新观察滑轨/障碍关系，再继续：{base}",
        ("deviation", "release"): "当前卡在释放阶段。先对齐放置位姿，确认目标到位后再缓慢释放，然后继续：{base}",
        ("target_lost", "grasp"): "抓取阶段目标丢失。先重新搜索目标并确认可抓取姿态，再继续：{base}",
        ("dropped", "transport"): "搬运阶段疑似掉物。先重新确认目标位置并重新抓取，再回到搬运轨迹继续：{base}",
        ("blocked", "release"): "释放阶段受阻。先退回到安全释放前位姿，重新对齐放置点后再释放：{base}",
    }
    template = stage_templates.get((fault, stage), templates.get(fault, templates["execution_failed"]))
    instruction = template.format(base=base)
    if reason_text:
        instruction = f"{instruction}（原因：{reason_text}）"
    return instruction


def send_feishu_alert(webhook, text):
    url = read_string(webhook)
    if not url:
        return {"sent": False, "reason": "webhook-not-configured"}
    payload = {
        "msg_type": "text",
        "content": {
            "text": read_string(text, "ReKep longrun fault"),
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib_request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return {"sent": True, "status_code": getattr(resp, "status", 200), "response": raw[:2000]}
    except urllib_error.URLError as exc:
        return {"sent": False, "reason": f"request-failed: {exc}"}
    except Exception as exc:
        return {"sent": False, "reason": str(exc)}



def infer_object_roles_from_schema(schema, grasp_state=None, instruction=""):
    schema = schema if isinstance(schema, list) else []
    grasp_state = as_dict(grasp_state)
    bindings = grasp_state.get("object_bindings") if isinstance(grasp_state.get("object_bindings"), dict) else {}
    instruction_text = read_string(instruction).lower()
    manipulated = ""
    if bindings:
        manipulated = str(next(iter(bindings.values()), ""))
    object_counts = {}
    object_names = []
    for item in schema:
        if not isinstance(item, dict):
            continue
        name = read_string(item.get("object"))
        if not name:
            continue
        object_counts[name] = object_counts.get(name, 0) + 1
        if name not in object_names:
            object_names.append(name)

    slider_keywords = ["drawer", "slider", "handle", "knob", "抽屉", "滑块", "把手"]
    rail_keywords = ["cabinet", "frame", "rail", "track", "base", "柜", "框", "轨"]
    slider_object = ""
    rail_object = ""
    for name in object_names:
        lname = name.lower()
        if not slider_object and any(token in lname for token in slider_keywords):
            slider_object = name
        if not rail_object and any(token in lname for token in rail_keywords):
            rail_object = name
    if not slider_object and manipulated:
        slider_object = manipulated
    if not slider_object and object_names:
        slider_object = object_names[0]
    if not rail_object:
        for name in object_names:
            if name != slider_object:
                rail_object = name
                break
    if not slider_object and any(token in instruction_text for token in ["drawer", "抽屉"]) and object_names:
        slider_object = object_names[0]
    return {
        "manipulated_object": manipulated or slider_object,
        "target_container": rail_object,
        "slider_object": slider_object,
        "rail_object": rail_object,
        "objects": object_names,
    }



def evaluate_container_placement(stage_runtime):
    stage_runtime = as_dict(stage_runtime)
    keypoint_obs = as_dict(stage_runtime.get("keypoint_obs"))
    keypoints_3d = keypoint_obs.get("keypoints_3d") if isinstance(keypoint_obs.get("keypoints_3d"), dict) else {}
    schema = keypoint_obs.get("schema") if isinstance(keypoint_obs.get("schema"), list) else []
    grasp_state = as_dict(stage_runtime.get("grasp_state"))
    roles = infer_object_roles_from_schema(schema, grasp_state=grasp_state)
    manipulated = read_string(roles.get("manipulated_object"))
    container = read_string(roles.get("target_container"))
    if not manipulated or not container or not keypoints_3d:
        return {"state": "unknown", "ok": False, "reason": "insufficient schema for placement evaluation", "roles": roles}

    obj_points = []
    container_points = []
    for item in schema:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id"))
        point = keypoints_3d.get(key)
        if not isinstance(point, list) or len(point) < 3:
            continue
        point_np = np.asarray(point[:3], dtype=float)
        if read_string(item.get("object")) == manipulated:
            obj_points.append(point_np)
        elif read_string(item.get("object")) == container:
            container_points.append(point_np)
    if not obj_points or not container_points:
        return {"state": "unknown", "ok": False, "reason": "missing object/container keypoints", "roles": roles}

    obj_center = np.mean(np.stack(obj_points, axis=0), axis=0)
    container_center = np.mean(np.stack(container_points, axis=0), axis=0)
    horizontal_dist = float(np.linalg.norm(obj_center[:2] - container_center[:2]))
    vertical_offset = float(obj_center[2] - container_center[2])
    aligned_xy = horizontal_dist < 0.05
    inserted_depth = vertical_offset < 0.03
    placed_ok = aligned_xy and inserted_depth
    return {
        "state": "placed_correctly" if placed_ok else "released_wrong_place",
        "ok": placed_ok,
        "reason": f"horizontal_dist={horizontal_dist:.4f}, vertical_offset={vertical_offset:.4f}",
        "roles": roles,
        "metrics": {
            "horizontal_dist": horizontal_dist,
            "vertical_offset": vertical_offset,
            "aligned_xy": aligned_xy,
            "inserted_depth": inserted_depth,
        },
    }



def judge_container_placement_with_vlm(args, frame_path, subtask):
    if not frame_path or not Path(frame_path).exists():
        return {"state": "unknown", "ok": False, "reason": "frame unavailable", "source": "vlm_placement"}
    prompt = (
        "你是机械臂放置结果判定器。判断被操作物体是否已经放到目标容器/目标区域。"
        "仅返回严格 JSON："
        "{\"state\":\"placed_correctly|released_wrong_place|unknown\",\"confidence\":0.0,\"reason\":\"...\"}。"
        f"当前子任务：{read_string(subtask)}"
    )
    raw_answer, vlm_cfg = ask_image_question(
        image_bytes=Path(frame_path).read_bytes(),
        question=prompt,
        default_model=args.model,
        system_prompt="You are a strict JSON judge for placement success. Return strict JSON only.",
        temperature=min(0.2, float(args.temperature)),
        max_tokens=max(220, int(args.max_tokens)),
    )
    parsed = parse_json_object_from_text(raw_answer)
    state = read_string(parsed.get("state"), "unknown").lower()
    if state not in {"placed_correctly", "released_wrong_place", "unknown"}:
        state = "unknown"
    return {
        "state": state,
        "ok": state == "placed_correctly",
        "confidence": float(parsed.get("confidence", 0.0)) if isinstance(parsed.get("confidence"), (int, float)) else 0.0,
        "reason": read_string(parsed.get("reason")),
        "raw_output": raw_answer,
        "model": vlm_cfg.get("model"),
        "source": "vlm_placement",
    }



def judge_slide_progress_with_vlm(args, frame_path, subtask):
    if not frame_path or not Path(frame_path).exists():
        return {"state": "unknown", "ok": False, "reason": "frame unavailable", "source": "vlm_slide"}
    prompt = (
        "你是机械臂滑动任务判定器。判断当前滑动对象是否已经沿正确方向滑到目标位置或足够进展。"
        "仅返回严格 JSON："
        "{\"state\":\"slide_progress_ok|slide_insufficient_progress|slide_blocked|unknown\",\"confidence\":0.0,\"reason\":\"...\"}。"
        f"当前子任务：{read_string(subtask)}"
    )
    raw_answer, vlm_cfg = ask_image_question(
        image_bytes=Path(frame_path).read_bytes(),
        question=prompt,
        default_model=args.model,
        system_prompt="You are a strict JSON judge for slide-task progress. Return strict JSON only.",
        temperature=min(0.2, float(args.temperature)),
        max_tokens=max(220, int(args.max_tokens)),
    )
    parsed = parse_json_object_from_text(raw_answer)
    state = read_string(parsed.get("state"), "unknown").lower()
    if state not in {"slide_progress_ok", "slide_insufficient_progress", "slide_blocked", "unknown"}:
        state = "unknown"
    return {
        "state": state,
        "ok": state == "slide_progress_ok",
        "confidence": float(parsed.get("confidence", 0.0)) if isinstance(parsed.get("confidence"), (int, float)) else 0.0,
        "reason": read_string(parsed.get("reason")),
        "raw_output": raw_answer,
        "model": vlm_cfg.get("model"),
        "source": "vlm_slide",
    }



def run_longrun_monitor_eval(args, state_dir, subtask, exec_job=None):
    exec_job = exec_job if isinstance(exec_job, dict) else {}
    exec_result = exec_job.get("result") if isinstance(exec_job.get("result"), dict) else {}
    stage_results = exec_result.get("stage_results") if isinstance(exec_result.get("stage_results"), list) else []
    if stage_results:
        last_stage = stage_results[-1] if isinstance(stage_results[-1], dict) else {}
        monitor_result = last_stage.get("monitor_result") if isinstance(last_stage.get("monitor_result"), dict) else {}
        recovery_result = last_stage.get("recovery_result") if isinstance(last_stage.get("recovery_result"), dict) else {}
        constraint_eval = last_stage.get("constraint_eval") if isinstance(last_stage.get("constraint_eval"), dict) else {}
        grasp_state = last_stage.get("grasp_state") if isinstance(last_stage.get("grasp_state"), dict) else {}
        stage_type = infer_stage_type_from_runtime(last_stage)
        operation_type = infer_operation_type(last_stage)
        stage_status = read_string(monitor_result.get("status"), "on_track")
        placement_eval = {"state": "unknown", "ok": False, "reason": "not-evaluated", "source": "geometry"}
        placement_vlm = {"state": "unknown", "ok": False, "reason": "not-evaluated", "source": "vlm_placement"}
        slide_eval = {"state": "unknown", "ok": False, "reason": "not-evaluated", "source": "geometry_slide"}
        slide_vlm = {"state": "unknown", "ok": False, "reason": "not-evaluated", "source": "vlm_slide"}
        if recovery_result.get("action") == "abort":
            stage_status = "blocked"
        elif recovery_result.get("action") in {"retry_stage", "replan", "reobserve"}:
            stage_status = "deviation"
        elif constraint_eval and not bool(constraint_eval.get("ok", True)):
            stage_status = "deviation"
        if operation_type == "slide":
            slide_eval = evaluate_sliding_progress(last_stage)
            slide_vlm = judge_slide_progress_with_vlm(args, read_string(last_stage.get("frame_path")), subtask)
            if slide_eval.get("ok") or slide_vlm.get("ok"):
                stage_status = "goal_done"
            elif read_string(slide_vlm.get("state")) == "slide_blocked":
                stage_status = "blocked"
            elif stage_status in {"on_track", "unknown"}:
                stage_status = "deviation"
        if stage_type == "release":
            placement_eval = evaluate_container_placement(last_stage)
            placement_vlm = judge_container_placement_with_vlm(args, read_string(last_stage.get("frame_path")), subtask)
            if placement_eval.get("ok") or placement_vlm.get("ok"):
                stage_status = "goal_done"
        return {
            "status": stage_status,
            "confidence": 0.95 if stage_status in {"on_track", "goal_done"} else 0.75,
            "reason": read_string(monitor_result.get("reasons", [""])[0] if isinstance(monitor_result.get("reasons"), list) and monitor_result.get("reasons") else monitor_result.get("reason")),
            "suggested_recovery": read_string(recovery_result.get("action"), monitor_result.get("suggested_action")),
            "stage": last_stage.get("stage"),
            "stage_type": stage_type,
            "operation_type": operation_type,
            "slide_eval": slide_eval,
            "slide_vlm": slide_vlm,
            "placement_eval": placement_eval,
            "placement_vlm": placement_vlm,
            "frame_path": read_string(last_stage.get("frame_path")),
            "used_standby_frame": False,
            "model": read_string((exec_result.get("vlm") or {}).get("planner_model")),
            "raw_output": json.dumps({
                "monitor_result": monitor_result,
                "recovery_result": recovery_result,
                "constraint_eval": constraint_eval,
                "grasp_state": grasp_state,
                "stage_type": stage_type,
                "operation_type": operation_type,
                "slide_eval": slide_eval,
                "slide_vlm": slide_vlm,
                "placement_eval": placement_eval,
                "placement_vlm": placement_vlm,
            }, ensure_ascii=False),
            "timestamp": now_iso(),
            "source": "live_runtime",
        }

    eval_args = clone_args(
        args,
        use_standby_frame=True,
        camera_warmup_frames=max(1, int(args.camera_warmup_frames)),
    )
    frame_path, used_standby_frame = resolve_frame_for_request(eval_args, state_dir, "longrun_monitor")
    prompt = (
        "你是机械臂在线监控器。请基于当前相机画面，判断当前子任务执行状态。"
        "仅返回严格 JSON："
        "{\"status\":\"on_track|goal_done|deviation|dropped|grasp_failed|target_lost|blocked|unknown\","
        "\"confidence\":0.0,"
        "\"reason\":\"...\","
        "\"suggested_recovery\":\"...\"}。"
        f"当前子任务：{read_string(subtask)}"
    )
    raw_answer, vlm_cfg = ask_image_question(
        image_bytes=Path(frame_path).read_bytes(),
        question=prompt,
        default_model=args.model,
        system_prompt="You are a strict JSON monitor for robotics execution state.",
        temperature=min(0.2, float(args.temperature)),
        max_tokens=max(256, int(args.max_tokens)),
    )
    parsed = parse_json_object_from_text(raw_answer)
    status = read_string(parsed.get("status"), "unknown").lower()
    if status not in {"on_track", "goal_done", "deviation", "dropped", "grasp_failed", "target_lost", "blocked", "unknown"}:
        status = "unknown"
    placement_vlm = {"state": "unknown", "ok": False, "reason": "not-evaluated", "source": "vlm_placement"}
    if status in {"goal_done", "on_track", "unknown"}:
        placement_vlm = judge_container_placement_with_vlm(args, str(frame_path), subtask)
        if placement_vlm.get("ok"):
            status = "goal_done"
    return {
        "status": status,
        "confidence": float(parsed.get("confidence", 0.0)) if isinstance(parsed.get("confidence"), (int, float)) else 0.0,
        "reason": read_string(parsed.get("reason")),
        "suggested_recovery": read_string(parsed.get("suggested_recovery")),
        "stage": None,
        "stage_type": "unknown",
        "placement_eval": {"state": "unknown", "ok": False, "reason": "not-available", "source": "geometry"},
        "placement_vlm": placement_vlm,
        "frame_path": str(frame_path),
        "used_standby_frame": bool(used_standby_frame),
        "model": vlm_cfg.get("model"),
        "raw_output": raw_answer,
        "timestamp": now_iso(),
        "source": "vlm_scene_monitor",
    }


def build_longrun_worker_argv(args, state_dir, job_id):
    hardware_profile = resolve_runtime_hardware_profile(args)
    worker_argv = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "longrun_worker",
        "--state_dir",
        str(state_dir),
        "--job_id",
        str(job_id),
        "--instruction",
        read_string(args.instruction),
        "--camera_source",
        hardware_profile.camera_source,
        "--camera_timeout_s",
        str(args.camera_timeout_s),
        "--camera_warmup_frames",
        str(args.camera_warmup_frames),
        "--robot_family",
        read_string(hardware_profile.robot_family, "dobot"),
        "--robot_driver",
        read_string(hardware_profile.robot_driver),
        "--dobot_driver",
        hardware_profile.robot_driver,
        "--xtrainer_sdk_dir",
        str(hardware_profile.xtrainer_sdk_dir),
        "--interval_s",
        str(args.interval_s),
        "--standby_stale_timeout_s",
        str(args.standby_stale_timeout_s),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--max_runtime_minutes",
        str(args.max_runtime_minutes),
        "--monitor_interval_s",
        str(args.monitor_interval_s),
        "--retry_limit",
        str(args.retry_limit),
        "--action_interval_s",
        str(resolve_action_interval_s(args)),
        "--feishu_webhook",
        read_string(args.feishu_webhook),
    ]
    if read_string(hardware_profile.robot_host):
        worker_argv.extend(["--robot_host", read_string(hardware_profile.robot_host)])
        worker_argv.extend(["--dobot_host", read_string(hardware_profile.robot_host)])
    if hardware_profile.robot_port is not None:
        worker_argv.extend(["--robot_port", str(int(hardware_profile.robot_port))])
        worker_argv.extend(["--dobot_port", str(int(hardware_profile.robot_port))])
    if hardware_profile.robot_move_port is not None:
        worker_argv.extend(["--robot_move_port", str(int(hardware_profile.robot_move_port))])
        worker_argv.extend(["--dobot_move_port", str(int(hardware_profile.robot_move_port))])
    if args.execute_motion:
        worker_argv.append("--execute_motion")
    if args.force:
        worker_argv.append("--force")
    return worker_argv


def run_longrun_start(args):
    state_dir = resolve_state_dir(args.state_dir)
    preflight = run_preflight(args, require_vlm=True)
    hardware_profile = resolve_profile_from_preflight_or_args(args, preflight)
    if preflight["status"] != "ready" and not args.force:
        return {
            "ok": False,
            "action": "longrun_start",
            "error": "Longrun preflight failed",
            "preflight": preflight,
        }, 2
    if not read_string(args.instruction):
        return {
            "ok": False,
            "action": "longrun_start",
            "error": "A non-empty --instruction is required",
            "preflight": preflight,
        }, 2

    subtasks = split_subtasks_text(args.instruction)
    job_id = args.job_id or f"longrun-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = job_state_path(state_dir, job_id)
    command_path = longrun_command_path(state_dir, job_id)
    log_path = build_log_path(state_dir, "real_longrun")
    now = now_iso()
    payload = save_job_state(
        path,
        {
            "job_id": job_id,
            "kind": "longrun",
            "status": "queued",
            "phase": "queued",
            "instruction": read_string(args.instruction),
            "current_subtask": None,
            "pending_subtasks": subtasks,
            "completed_subtasks": [],
            "slide_stability_hits": 0,
            "failed_subtasks": [],
            "paused": False,
            "active_execute_job_id": None,
            "last_monitor": None,
            "last_stage_runtime": None,
            "command_offset": 0,
            "command_path": str(command_path),
            "run_log_path": str(log_path),
            "max_runtime_minutes": float(args.max_runtime_minutes),
            "monitor_interval_s": float(args.monitor_interval_s),
            "retry_limit": int(args.retry_limit),
            "driver": hardware_profile.robot_driver,
            "camera_source": hardware_profile.camera_source,
            "hardware_profile": hardware_profile.to_dict(),
            "execute_motion": bool(args.execute_motion),
            "action_interval_s": resolve_action_interval_s(args) if bool(args.execute_motion) else 0.0,
            "feishu_webhook": read_string(args.feishu_webhook),
            "alerts": [],
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "process_group_id": None,
            "error": None,
            "exit_code": None,
            "preflight": preflight,
            "state_path": str(path),
        },
    )

    worker_argv = build_longrun_worker_argv(args, state_dir, job_id)
    proc = subprocess.Popen(
        worker_argv,
        cwd=str(Path(__file__).resolve().parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    payload = update_job_state(
        path,
        status="running",
        phase="starting",
        started_at=now_iso(),
        pid=proc.pid,
        process_group_id=proc.pid,
    )
    append_log_line(log_path, f"[longrun] job started id={job_id} subtasks={len(subtasks)}")
    return {
        "ok": True,
        "action": "longrun_start",
        "job": payload,
    }, 0


def run_longrun_worker(args):
    state_dir = resolve_state_dir(args.state_dir)
    path = job_state_path(state_dir, args.job_id)
    if not path.exists():
        return {
            "ok": False,
            "action": "longrun_worker",
            "error": f"Unknown longrun job id: {args.job_id}",
        }, 1
    job = load_json_if_exists(path) or {}
    if job.get("kind") != "longrun":
        return {
            "ok": False,
            "action": "longrun_worker",
            "error": f"Job is not a longrun job: {args.job_id}",
        }, 1
    log_path = read_string(job.get("run_log_path"), str(build_log_path(state_dir, "real_longrun")))
    command_path = read_string(job.get("command_path"), str(longrun_command_path(state_dir, args.job_id)))
    start_ts = time.time()
    max_runtime_s = max(60.0, float(job.get("max_runtime_minutes", args.max_runtime_minutes)) * 60.0)
    monitor_interval_s = max(0.5, float(job.get("monitor_interval_s", args.monitor_interval_s)))
    retry_limit = max(0, int(job.get("retry_limit", args.retry_limit)))
    command_offset = int(job.get("command_offset", 0) or 0)
    current_subtask = read_string(job.get("current_subtask"))
    pending_subtasks = list(job.get("pending_subtasks") or [])
    completed_subtasks = list(job.get("completed_subtasks") or [])
    failed_subtasks = list(job.get("failed_subtasks") or [])
    paused = bool(job.get("paused", False))
    active_execute_job_id = read_string(job.get("active_execute_job_id"))
    active_retry = int(job.get("active_retry", 0) or 0)
    alerts = list(job.get("alerts") or [])

    update_job_state(
        path,
        status="running",
        phase="running",
        pid=os.getpid(),
        process_group_id=os.getpgrp(),
        started_at=job.get("started_at") or now_iso(),
        command_path=command_path,
        run_log_path=log_path,
        command_offset=command_offset,
        current_subtask=current_subtask or None,
        pending_subtasks=pending_subtasks,
        completed_subtasks=completed_subtasks,
        failed_subtasks=failed_subtasks,
        active_execute_job_id=active_execute_job_id or None,
        active_retry=active_retry,
        paused=paused,
    )
    append_log_line(log_path, f"[longrun] worker online pid={os.getpid()} monitor_interval_s={monitor_interval_s}")
    emit_progress(f"[longrun] worker started job_id={args.job_id}")

    standby_payload, standby_code = run_standby_start(clone_args(args, state_dir=str(state_dir)))
    if standby_code != 0:
        failure = update_job_state(
            path,
            status="failed",
            phase="failed",
            finished_at=now_iso(),
            error=read_string(standby_payload.get("error"), "failed to start standby stream"),
            exit_code=1,
        )
        append_log_line(log_path, f"[longrun] failed to start standby: {failure.get('error')}")
        return {"ok": False, "action": "longrun_worker", "job": failure}, 1

    next_monitor_at = 0.0
    terminal_payload = None

    while True:
        latest = load_json_if_exists(path) or {}
        if latest.get("status") in {"cancelled", "stopped"}:
            terminal_payload = latest
            break
        elapsed_s = time.time() - start_ts
        if elapsed_s > max_runtime_s:
            message = f"[longrun] timeout after {int(elapsed_s)}s"
            append_log_line(log_path, message)
            alert_result = send_feishu_alert(
                read_string(latest.get("feishu_webhook"), read_string(args.feishu_webhook)),
                f"长任务超时: job={args.job_id}, elapsed={int(elapsed_s)}s, subtask={current_subtask or 'n/a'}",
            )
            alerts.append({"timestamp": now_iso(), "type": "timeout", "result": alert_result})
            terminal_payload = update_job_state(
                path,
                status="failed",
                phase="failed",
                finished_at=now_iso(),
                error=f"Longrun timeout after {int(elapsed_s)} seconds",
                exit_code=1,
                alerts=alerts,
            )
            break

        commands, command_offset = read_longrun_commands_since(command_path, command_offset)
        for event in commands:
            cmd = normalize_longrun_command(event.get("command"), event.get("text"))
            text = read_string(event.get("text"))
            append_log_line(log_path, f"[longrun] command {cmd or 'unknown'} text={text!r}")
            if cmd == "pause":
                paused = True
                if active_execute_job_id:
                    run_job_cancel(clone_args(args, state_dir=str(state_dir), job_id=active_execute_job_id))
                    active_execute_job_id = ""
                emit_progress("[longrun] paused by human command")
            elif cmd == "resume":
                paused = False
                emit_progress("[longrun] resumed by human command")
            elif cmd == "stop":
                if active_execute_job_id:
                    run_job_cancel(clone_args(args, state_dir=str(state_dir), job_id=active_execute_job_id))
                    active_execute_job_id = ""
                terminal_payload = update_job_state(
                    path,
                    status="cancelled",
                    phase="cancelled",
                    finished_at=now_iso(),
                    error="Stopped by human command",
                    exit_code=130,
                    command_offset=command_offset,
                )
                break
            elif cmd == "append_subtask" and text:
                pending_subtasks.append(text)
            elif cmd == "replace_plan" and text:
                if active_execute_job_id:
                    run_job_cancel(clone_args(args, state_dir=str(state_dir), job_id=active_execute_job_id))
                    active_execute_job_id = ""
                pending_subtasks = split_subtasks_text(text)
                current_subtask = ""
                active_retry = 0
            elif cmd == "replace_current" and text:
                if active_execute_job_id:
                    run_job_cancel(clone_args(args, state_dir=str(state_dir), job_id=active_execute_job_id))
                    active_execute_job_id = ""
                current_subtask = text
                active_retry = 0
            elif cmd == "skip_current":
                if active_execute_job_id:
                    run_job_cancel(clone_args(args, state_dir=str(state_dir), job_id=active_execute_job_id))
                    active_execute_job_id = ""
                current_subtask = ""
                active_retry = 0
        if terminal_payload is not None:
            break

        if paused:
            update_job_state(
                path,
                phase="paused",
                paused=True,
                command_offset=command_offset,
                current_subtask=current_subtask or None,
                pending_subtasks=pending_subtasks,
                completed_subtasks=completed_subtasks,
                failed_subtasks=failed_subtasks,
                active_execute_job_id=active_execute_job_id or None,
                active_retry=active_retry,
            )
            time.sleep(min(0.5, monitor_interval_s))
            continue

        if not current_subtask:
            if pending_subtasks:
                current_subtask = read_string(pending_subtasks.pop(0))
                active_retry = 0
                append_log_line(log_path, f"[longrun] pick subtask: {current_subtask}")
                emit_progress(f"[longrun] subtask started: {current_subtask}")
            else:
                terminal_payload = update_job_state(
                    path,
                    status="succeeded",
                    phase="completed",
                    finished_at=now_iso(),
                    error=None,
                    exit_code=0,
                    paused=False,
                    current_subtask=None,
                    pending_subtasks=[],
                    completed_subtasks=completed_subtasks,
                    failed_subtasks=failed_subtasks,
                    active_execute_job_id=None,
                    command_offset=command_offset,
                    alerts=alerts,
                )
                append_log_line(log_path, "[longrun] all subtasks completed")
                break

        if not active_execute_job_id:
            exec_job_id = f"exec-{dt.datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
            exec_args = clone_args(
                args,
                state_dir=str(state_dir),
                job_id=exec_job_id,
                instruction=current_subtask,
                use_standby_frame=True,
            )
            launch_payload, launch_code = run_execute_background(exec_args)
            if launch_code != 0 or not launch_payload.get("ok"):
                reason = read_string(launch_payload.get("error"), "failed to launch execute_background")
                if active_retry < retry_limit:
                    active_retry += 1
                    current_subtask = build_recovery_instruction("execution_failed", current_subtask, reason)
                    append_log_line(log_path, f"[longrun] launch failed, retry={active_retry}, reason={reason}")
                    emit_progress(f"[longrun] retry launch ({active_retry}/{retry_limit})")
                else:
                    failed_subtasks.append({"subtask": current_subtask, "reason": reason})
                    alert_result = send_feishu_alert(
                        read_string(latest.get("feishu_webhook"), read_string(args.feishu_webhook)),
                        f"长任务失败: job={args.job_id}, subtask={current_subtask}, reason={reason}",
                    )
                    alerts.append({"timestamp": now_iso(), "type": "launch_failed", "result": alert_result})
                    terminal_payload = update_job_state(
                        path,
                        status="failed",
                        phase="failed",
                        finished_at=now_iso(),
                        error=reason,
                        exit_code=1,
                        failed_subtasks=failed_subtasks,
                        alerts=alerts,
                        command_offset=command_offset,
                    )
                    break
            else:
                active_execute_job_id = read_string(as_dict(launch_payload.get("job", {})).get("job_id"), exec_job_id)
                next_monitor_at = 0.0
                update_job_state(
                    path,
                    phase="executing",
                    active_execute_job_id=active_execute_job_id,
                    current_subtask=current_subtask,
                    pending_subtasks=pending_subtasks,
                    completed_subtasks=completed_subtasks,
                    failed_subtasks=failed_subtasks,
                    active_retry=active_retry,
                    paused=False,
                    command_offset=command_offset,
                    slide_stability_hits=0,
                )
                emit_progress(f"[longrun] execute job started: {active_execute_job_id}")
            time.sleep(0.2)
            continue

        status_payload, _ = run_job_status(clone_args(args, state_dir=str(state_dir), job_id=active_execute_job_id))
        exec_job = as_dict(status_payload.get("job", {}))
        exec_status = read_string(exec_job.get("status"), "unknown").lower()
        now_ts = time.time()

        if now_ts >= next_monitor_at:
            latest_exec_job = exec_job if isinstance(exec_job, dict) else {}
            latest_exec_result = latest_exec_job.get("result") if isinstance(latest_exec_job.get("result"), dict) else {}
            latest_stage_runtime = None
            if isinstance(latest_exec_result.get("stage_results"), list) and latest_exec_result.get("stage_results"):
                latest_stage_runtime = latest_exec_result.get("stage_results")[-1]
            try:
                monitor = run_longrun_monitor_eval(args, state_dir, current_subtask, exec_job=latest_exec_job)
            except Exception as exc:
                monitor = {
                    "status": "monitor_failed",
                    "reason": str(exc),
                    "timestamp": now_iso(),
                    "source": "monitor_exception",
                }
            slide_stability_hits = int((load_json_if_exists(path) or {}).get("slide_stability_hits", 0) or 0)
            if read_string(monitor.get("operation_type")) == "slide":
                if as_dict(monitor.get("slide_eval")).get("ok"):
                    slide_stability_hits += 1
                else:
                    slide_stability_hits = 0
                if slide_stability_hits >= 2:
                    monitor["status"] = "goal_done"
                    monitor["reason"] = read_string(monitor.get("reason"), "slide progress stable across checks")
            else:
                slide_stability_hits = 0
            update_job_state(
                path,
                last_monitor=monitor,
                last_stage_runtime=latest_stage_runtime,
                phase="monitoring" if exec_status == "running" else "executing",
                active_execute_job_id=active_execute_job_id,
                command_offset=command_offset,
                slide_stability_hits=slide_stability_hits,
            )
            emit_progress(
                f"[longrun] monitor status={read_string(monitor.get('status'),'unknown')} reason={read_string(monitor.get('reason'))}"
            )
            next_monitor_at = now_ts + monitor_interval_s

            if exec_status == "running":
                stage_type = read_string(monitor.get("stage_type")) or infer_stage_type_from_runtime(latest_stage_runtime)
                fault_type = infer_longrun_fault_type(monitor, latest_stage_runtime)
                recovery_policy = choose_longrun_recovery_policy(fault_type, stage_type)
                if read_string(monitor.get("status")) in {"dropped", "grasp_failed", "target_lost", "blocked"} and recovery_policy.get("cancel_running_job", True):
                    run_job_cancel(clone_args(args, state_dir=str(state_dir), job_id=active_execute_job_id))
                    exec_status = "failed"
                    exec_job["error"] = f"monitor-detected-{fault_type or read_string(monitor.get('status'))}"
                    append_log_line(log_path, f"[longrun] monitor interrupted execute job due to {monitor.get('status')} stage_type={stage_type or 'unknown'} fault={fault_type or 'unknown'}")

        if exec_status == "running":
            time.sleep(min(0.5, monitor_interval_s))
            continue

        active_execute_job_id = ""
        reason = read_string(exec_job.get("error"))
        if exec_status == "succeeded":
            longrun_snapshot = load_json_if_exists(path) or {}
            monitor = as_dict(longrun_snapshot.get("last_monitor", {}))
            last_stage_runtime = as_dict(longrun_snapshot.get("last_stage_runtime", {}))
            recovery_result = as_dict(last_stage_runtime.get("recovery_result", {}))
            monitor_status = read_string(monitor.get("status"), "unknown")
            recovery_action = read_string(recovery_result.get("action"))
            stage_type = read_string(monitor.get("stage_type")) or infer_stage_type_from_runtime(last_stage_runtime)
            if recovery_action in {"retry_stage", "replan", "reobserve"} or monitor_status in {"deviation", "dropped", "grasp_failed", "target_lost", "blocked"}:
                reason = read_string(recovery_result.get("reason"), read_string(monitor.get("reason"), monitor_status))
                fault_type = infer_longrun_fault_type(monitor, last_stage_runtime)
                recovery_policy = choose_longrun_recovery_policy(fault_type, stage_type)
                policy_result = apply_longrun_recovery_policy(
                    policy=recovery_policy,
                    subtask=current_subtask,
                    reason=reason,
                    stage_type=stage_type,
                    fault_type=fault_type or monitor_status or recovery_action or "execution_failed",
                )
                allowed_retry_budget = min(retry_limit, int(recovery_policy.get("retry_budget", retry_limit)))
                if active_retry < allowed_retry_budget:
                    active_retry += 1
                    if policy_result.get("skip_current"):
                        append_log_line(log_path, f"[longrun] runtime recovery decided to skip current subtask stage_type={stage_type or 'unknown'} fault={fault_type}")
                        completed_subtasks.append(f"[skipped] {current_subtask}")
                        current_subtask = ""
                        active_retry = 0
                    else:
                        replacement_plan = [read_string(x) for x in (policy_result.get("replacement_plan") or []) if read_string(x)]
                        if read_string(policy_result.get("mode")) == "replace_plan" and replacement_plan:
                            pending_subtasks = replacement_plan + pending_subtasks
                            current_subtask = ""
                        else:
                            if replacement_plan:
                                current_subtask = replacement_plan[0]
                                for extra in reversed(replacement_plan[1:]):
                                    pending_subtasks.insert(0, extra)
                            else:
                                current_subtask = read_string(policy_result.get("replacement_subtask"), current_subtask)
                            for extra in policy_result.get("append_subtasks") or []:
                                pending_subtasks.insert(0, read_string(extra))
                        append_log_line(log_path, f"[longrun] runtime recovery retry={active_retry}/{allowed_retry_budget} stage_type={stage_type or 'unknown'} action={recovery_action or monitor_status} fault={fault_type} policy={json.dumps(recovery_policy, ensure_ascii=False)} policy_result={json.dumps(policy_result, ensure_ascii=False)} reason={reason}")
                        emit_progress(f"[longrun] recovery retry ({active_retry}/{allowed_retry_budget})")
                else:
                    failed_subtasks.append({"subtask": current_subtask, "reason": reason, "stage_type": stage_type or "unknown", "fault_type": fault_type or monitor_status or recovery_action or "execution_failed", "recovery_policy": recovery_policy, "policy_result": policy_result})
                    alert_result = send_feishu_alert(
                        read_string(latest.get("feishu_webhook"), read_string(args.feishu_webhook)),
                        f"长任务失败: job={args.job_id}, subtask={current_subtask}, reason={reason}",
                    )
                    alerts.append({"timestamp": now_iso(), "type": "deviation", "result": alert_result})
                    terminal_payload = update_job_state(
                        path,
                        status="failed",
                        phase="failed",
                        finished_at=now_iso(),
                        error=reason or "deviation unrecoverable",
                        exit_code=1,
                        failed_subtasks=failed_subtasks,
                        alerts=alerts,
                        command_offset=command_offset,
                    )
                    break
            else:
                completed_subtasks.append(current_subtask)
                append_log_line(log_path, f"[longrun] subtask completed: {current_subtask}")
                current_subtask = ""
                active_retry = 0
        else:
            reason = reason or f"execute job ended with status {exec_status}"
            longrun_snapshot = load_json_if_exists(path) or {}
            monitor = as_dict(longrun_snapshot.get("last_monitor", {}))
            last_stage_runtime = as_dict(longrun_snapshot.get("last_stage_runtime", {}))
            stage_type = read_string(monitor.get("stage_type")) or infer_stage_type_from_runtime(last_stage_runtime)
            fault_type = infer_longrun_fault_type(monitor, last_stage_runtime)
            recovery_policy = choose_longrun_recovery_policy(fault_type, stage_type)
            policy_result = apply_longrun_recovery_policy(
                policy=recovery_policy,
                subtask=current_subtask,
                reason=reason,
                stage_type=stage_type,
                fault_type=fault_type or "execution_failed",
            )
            allowed_retry_budget = min(retry_limit, int(recovery_policy.get("retry_budget", retry_limit)))
            if active_retry < allowed_retry_budget:
                active_retry += 1
                if policy_result.get("skip_current"):
                    append_log_line(log_path, f"[longrun] execute failure caused skip_current stage_type={stage_type or 'unknown'} fault={fault_type or 'execution_failed'}")
                    completed_subtasks.append(f"[skipped] {current_subtask}")
                    current_subtask = ""
                    active_retry = 0
                else:
                    replacement_plan = [read_string(x) for x in (policy_result.get("replacement_plan") or []) if read_string(x)]
                    if read_string(policy_result.get("mode")) == "replace_plan" and replacement_plan:
                        pending_subtasks = replacement_plan + pending_subtasks
                        current_subtask = ""
                    else:
                        if replacement_plan:
                            current_subtask = replacement_plan[0]
                            for extra in reversed(replacement_plan[1:]):
                                pending_subtasks.insert(0, extra)
                        else:
                            current_subtask = read_string(policy_result.get("replacement_subtask"), current_subtask)
                        for extra in policy_result.get("append_subtasks") or []:
                            pending_subtasks.insert(0, read_string(extra))
                    append_log_line(log_path, f"[longrun] execute failed, retry={active_retry}/{allowed_retry_budget} stage_type={stage_type or 'unknown'} fault={fault_type or 'execution_failed'} policy={json.dumps(recovery_policy, ensure_ascii=False)} policy_result={json.dumps(policy_result, ensure_ascii=False)} reason={reason}")
                    emit_progress(f"[longrun] execute retry ({active_retry}/{allowed_retry_budget})")
            else:
                failed_subtasks.append({"subtask": current_subtask, "reason": reason, "stage_type": stage_type or "unknown", "fault_type": fault_type or "execution_failed", "recovery_policy": recovery_policy, "policy_result": policy_result})
                alert_result = send_feishu_alert(
                    read_string(latest.get("feishu_webhook"), read_string(args.feishu_webhook)),
                    f"长任务失败: job={args.job_id}, subtask={current_subtask}, reason={reason}",
                )
                alerts.append({"timestamp": now_iso(), "type": "execute_failed", "result": alert_result})
                terminal_payload = update_job_state(
                    path,
                    status="failed",
                    phase="failed",
                    finished_at=now_iso(),
                    error=reason,
                    exit_code=1,
                    failed_subtasks=failed_subtasks,
                    alerts=alerts,
                    command_offset=command_offset,
                )
                break

        update_job_state(
            path,
            phase="running",
            active_execute_job_id=None,
            current_subtask=current_subtask or None,
            pending_subtasks=pending_subtasks,
            completed_subtasks=completed_subtasks,
            failed_subtasks=failed_subtasks,
            active_retry=active_retry,
            paused=False,
            command_offset=command_offset,
            alerts=alerts,
        )
        time.sleep(0.1)

    final_state = terminal_payload or (load_json_if_exists(path) or {})
    if read_string(final_state.get("status")) in {"running", "queued"}:
        final_state = update_job_state(
            path,
            status="failed",
            phase="failed",
            finished_at=now_iso(),
            error="Longrun worker exited unexpectedly",
            exit_code=1,
        )
    emit_progress(f"[longrun] worker finished status={read_string(final_state.get('status'),'unknown')}")
    return {
        "ok": read_string(final_state.get("status")) == "succeeded",
        "action": "longrun_worker",
        "job": final_state,
        "log_tail": read_log_tail(log_path),
    }, 0 if read_string(final_state.get("status")) == "succeeded" else 1


def run_longrun_status(args):
    state_dir = resolve_state_dir(args.state_dir)
    path = job_state_path(state_dir, args.job_id) if args.job_id else latest_longrun_job_path(state_dir)
    if path is None or not Path(path).exists():
        return {
            "ok": False,
            "action": "longrun_status",
            "error": "No longrun jobs were found" if not args.job_id else f"Unknown longrun job id: {args.job_id}",
        }, 1
    payload = load_json_if_exists(path) or {}
    pid = payload.get("pid")
    worker_alive = False
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
            worker_alive = True
        except ProcessLookupError:
            worker_alive = False
    payload["worker_alive"] = worker_alive
    command_path = read_string(payload.get("command_path"), str(longrun_command_path(state_dir, payload.get("job_id"))))
    command_offset = int(payload.get("command_offset", 0) or 0)
    total_commands = len(read_jsonl(command_path))
    payload["pending_commands"] = max(0, total_commands - command_offset)
    if payload.get("status") == "running" and not worker_alive:
        payload = update_job_state(
            path,
            status="failed",
            phase="failed",
            finished_at=now_iso(),
            error=payload.get("error") or "Longrun worker exited unexpectedly",
            exit_code=payload.get("exit_code") or 1,
        )
        payload["worker_alive"] = False
        payload["pending_commands"] = max(0, total_commands - int(payload.get("command_offset", 0) or 0))
    return {
        "ok": True,
        "action": "longrun_status",
        "job": payload,
        "last_stage_runtime": payload.get("last_stage_runtime"),
        "log_tail": read_log_tail(payload.get("run_log_path")),
    }, 0


def run_longrun_command(args):
    state_dir = resolve_state_dir(args.state_dir)
    path = job_state_path(state_dir, args.job_id) if args.job_id else latest_longrun_job_path(state_dir)
    if path is None or not Path(path).exists():
        return {
            "ok": False,
            "action": "longrun_command",
            "error": "No longrun jobs were found" if not args.job_id else f"Unknown longrun job id: {args.job_id}",
        }, 1
    payload = load_json_if_exists(path) or {}
    if payload.get("kind") != "longrun":
        return {
            "ok": False,
            "action": "longrun_command",
            "error": f"Job is not longrun: {payload.get('job_id')}",
        }, 1
    if payload.get("status") in {"succeeded", "failed", "cancelled"}:
        return {
            "ok": False,
            "action": "longrun_command",
            "error": f"Longrun job already finished with status={payload.get('status')}",
            "job": payload,
        }, 1
    command = normalize_longrun_command(args.command, args.command_text)
    if not command:
        return {
            "ok": False,
            "action": "longrun_command",
            "error": "Unsupported command. Use pause/resume/stop/append_subtask/replace_plan/replace_current/skip_current.",
        }, 2
    event = {
        "id": f"cmd-{uuid.uuid4().hex[:8]}",
        "timestamp": now_iso(),
        "command": command,
        "text": read_string(args.command_text),
    }
    command_file = read_string(payload.get("command_path"), str(longrun_command_path(state_dir, payload.get("job_id"))))
    append_jsonl(command_file, event)
    payload = update_job_state(path, command_path=command_file)
    append_log_line(read_string(payload.get("run_log_path")), f"[longrun] enqueue command={command} text={event['text']!r}")
    return {
        "ok": True,
        "action": "longrun_command",
        "job": payload,
        "command_event": event,
    }, 0


def run_longrun_stop(args):
    state_dir = resolve_state_dir(args.state_dir)
    path = job_state_path(state_dir, args.job_id) if args.job_id else latest_longrun_job_path(state_dir)
    if path is None or not Path(path).exists():
        return {
            "ok": False,
            "action": "longrun_stop",
            "error": "No longrun jobs were found" if not args.job_id else f"Unknown longrun job id: {args.job_id}",
        }, 1
    payload = load_json_if_exists(path) or {}
    if payload.get("kind") != "longrun":
        return {
            "ok": False,
            "action": "longrun_stop",
            "error": f"Job is not longrun: {payload.get('job_id')}",
        }, 1
    command_file = read_string(payload.get("command_path"), str(longrun_command_path(state_dir, payload.get("job_id"))))
    event = {
        "id": f"cmd-{uuid.uuid4().hex[:8]}",
        "timestamp": now_iso(),
        "command": "stop",
        "text": read_string(args.command_text, "stop"),
    }
    append_jsonl(command_file, event)
    pgid = payload.get("process_group_id")
    pid = payload.get("pid")
    try:
        if isinstance(pgid, int) and pgid > 0:
            os.killpg(pgid, signal.SIGTERM)
        elif isinstance(pid, int) and pid > 0:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    payload = update_job_state(
        path,
        status="cancelled",
        phase="cancelled",
        finished_at=now_iso(),
        error="Stopped by user",
        exit_code=130,
        command_path=command_file,
    )
    append_log_line(read_string(payload.get("run_log_path")), "[longrun] stop requested by user")
    return {
        "ok": True,
        "action": "longrun_stop",
        "job": payload,
        "command_event": event,
        "log_tail": read_log_tail(payload.get("run_log_path")),
    }, 0


def main():
    parser = argparse.ArgumentParser(description="Real robot bridge for Dobot + ReKep + OpenClaw")
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "standby_start",
            "standby_status",
            "standby_stop",
            "standby_worker",
            "scene_qa",
            "execute",
            "execute_background",
            "execute_worker",
            "job_status",
            "job_cancel",
            "longrun_start",
            "longrun_worker",
            "longrun_status",
            "longrun_command",
            "longrun_stop",
        ],
    )
    parser.add_argument("--state_dir", type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--task", type=str, default="rekep")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--command", type=str, default=None)
    parser.add_argument("--command_text", type=str, default=None)
    parser.add_argument("--camera_profile", type=str, default=None)
    parser.add_argument("--camera_serial", type=str, default=None)
    parser.add_argument("--dobot_settings_ini", type=str, default=None)
    parser.add_argument("--camera_extrinsic_script", type=str, default=None)
    parser.add_argument("--realsense_calib_dir", type=str, default=None)
    parser.add_argument("--camera_source", type=str, default=None)
    parser.add_argument("--camera_timeout_s", type=float, default=8.0)
    parser.add_argument("--camera_warmup_frames", type=int, default=6)
    parser.add_argument("--use_standby_frame", action="store_true")
    parser.add_argument("--robot_family", type=str, default=None)
    parser.add_argument("--robot_driver", type=str, default=None)
    parser.add_argument("--robot_host", type=str, default=None)
    parser.add_argument("--robot_port", type=int, default=None)
    parser.add_argument("--robot_move_port", type=int, default=None)
    parser.add_argument("--dobot_driver", type=str, default=None)
    parser.add_argument("--dobot_host", type=str, default=None)
    parser.add_argument("--dobot_port", type=int, default=None)
    parser.add_argument("--dobot_move_port", type=int, default=None)
    parser.add_argument("--xtrainer_sdk_dir", type=str, default=None)
    parser.add_argument("--interval_s", type=float, default=0.2)
    parser.add_argument("--standby_stale_timeout_s", type=float, default=15.0)
    parser.add_argument("--execute_motion", action="store_true")
    parser.add_argument(
        "--action_interval_s",
        type=float,
        default=DEFAULT_ACTION_INTERVAL_S,
        help="Seconds to wait between consecutive actions when --execute_motion is enabled.",
    )
    parser.add_argument(
        "--rekep_execution_mode",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_REKEP_EXECUTION_MODES),
        help="ReKep real execution backend: solver uses DINOv2 candidates + constraint solver; vlm_stage keeps the legacy VLM stage-action planner.",
    )
    parser.add_argument(
        "--rekep_grasp_depth_m",
        type=float,
        default=DEFAULT_REAL_GRASP_DEPTH_M,
        help="Local grasp insertion depth used by solver mode after reaching the optimized pre-grasp pose.",
    )
    parser.add_argument(
        "--rekep_vlm_stage_grasp_descend_m",
        type=float,
        default=None,
        help="Legacy vlm_stage only: insert one extra downward movel before the first close_gripper in grasp stages.",
    )
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_runtime_minutes", type=float, default=DEFAULT_LONGRUN_MAX_MINUTES)
    parser.add_argument("--monitor_interval_s", type=float, default=DEFAULT_LONGRUN_MONITOR_INTERVAL_S)
    parser.add_argument("--retry_limit", type=int, default=DEFAULT_LONGRUN_RETRY_LIMIT)
    parser.add_argument("--feishu_webhook", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    try:
        if args.action == "preflight":
            payload, exit_code = {"ok": True, "action": "preflight", "preflight": run_preflight(args)}, 0
        elif args.action == "standby_start":
            payload, exit_code = run_standby_start(args)
        elif args.action == "standby_status":
            payload, exit_code = run_standby_status(args)
        elif args.action == "standby_stop":
            payload, exit_code = run_standby_stop(args)
        elif args.action == "standby_worker":
            payload, exit_code = run_standby_worker(args)
        elif args.action == "scene_qa":
            payload, exit_code = run_scene_qa(args)
        elif args.action == "execute":
            payload, exit_code = run_execute(args)
        elif args.action == "execute_background":
            payload, exit_code = run_execute_background(args)
        elif args.action == "execute_worker":
            payload, exit_code = run_execute_worker(args)
        elif args.action == "job_status":
            payload, exit_code = run_job_status(args)
        elif args.action == "job_cancel":
            payload, exit_code = run_job_cancel(args)
        elif args.action == "longrun_start":
            payload, exit_code = run_longrun_start(args)
        elif args.action == "longrun_worker":
            payload, exit_code = run_longrun_worker(args)
        elif args.action == "longrun_status":
            payload, exit_code = run_longrun_status(args)
        elif args.action == "longrun_command":
            payload, exit_code = run_longrun_command(args)
        elif args.action == "longrun_stop":
            payload, exit_code = run_longrun_stop(args)
        else:
            payload, exit_code = {
                "ok": False,
                "action": args.action,
                "error": f"Unsupported action: {args.action}",
            }, 2
    except Exception as exc:
        payload = {
            "ok": False,
            "action": args.action,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    print_json(payload, pretty=args.pretty)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

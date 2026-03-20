"""Real-robot ReKep driver backed by ``tools/rekep/dobot_bridge.py``."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from hal.base_driver import BaseDriver

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TOOL_ROOT = _REPO_ROOT / "tools" / "rekep"
_DEFAULT_STATE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "rekep_real_state"

_ACTION_ALIASES: dict[str, str] = {
    "preflight": "preflight",
    "execute": "execute",
    "execute_background": "execute_background",
    "scene_qa": "scene_qa",
    "standby_start": "standby_start",
    "standby_status": "standby_status",
    "standby_stop": "standby_stop",
    "job_status": "job_status",
    "job_cancel": "job_cancel",
    "longrun_start": "longrun_start",
    "longrun_status": "longrun_status",
    "longrun_command": "longrun_command",
    "longrun_stop": "longrun_stop",
    "real_preflight": "preflight",
    "real_execute": "execute",
    "real_execute_background": "execute_background",
    "real_scene_qa": "scene_qa",
    "real_standby_start": "standby_start",
    "real_standby_status": "standby_status",
    "real_standby_stop": "standby_stop",
    "real_job_status": "job_status",
    "real_job_cancel": "job_cancel",
    "real_longrun_start": "longrun_start",
    "real_longrun_status": "longrun_status",
    "real_longrun_command": "longrun_command",
    "real_longrun_stop": "longrun_stop",
}

_HIGH_LEVEL_ACTIONS = {
    "move_to",
    "pick_up",
    "put_down",
    "push",
    "point_to",
    "open_gripper",
    "close_gripper",
    "rekep_instruction",
}

_STRING_ARGS: dict[str, str] = {
    "task": "--task",
    "question": "--question",
    "instruction": "--instruction",
    "command": "--command",
    "command_text": "--command_text",
    "camera_profile": "--camera_profile",
    "camera_serial": "--camera_serial",
    "dobot_settings_ini": "--dobot_settings_ini",
    "camera_extrinsic_script": "--camera_extrinsic_script",
    "realsense_calib_dir": "--realsense_calib_dir",
    "camera_source": "--camera_source",
    "dobot_driver": "--dobot_driver",
    "dobot_host": "--dobot_host",
    "xtrainer_sdk_dir": "--xtrainer_sdk_dir",
    "model": "--model",
    "job_id": "--job_id",
    "rekep_execution_mode": "--rekep_execution_mode",
}

_NUMERIC_ARGS: dict[str, str] = {
    "camera_timeout_s": "--camera_timeout_s",
    "camera_warmup_frames": "--camera_warmup_frames",
    "dobot_port": "--dobot_port",
    "dobot_move_port": "--dobot_move_port",
    "interval_s": "--interval_s",
    "standby_stale_timeout_s": "--standby_stale_timeout_s",
    "action_interval_s": "--action_interval_s",
    "rekep_grasp_depth_m": "--rekep_grasp_depth_m",
    "rekep_vlm_stage_grasp_descend_m": "--rekep_vlm_stage_grasp_descend_m",
    "temperature": "--temperature",
    "max_tokens": "--max_tokens",
    "max_runtime_minutes": "--max_runtime_minutes",
    "monitor_interval_s": "--monitor_interval_s",
    "retry_limit": "--retry_limit",
}

_BOOL_FLAGS: dict[str, str] = {
    "execute_motion": "--execute_motion",
    "use_standby_frame": "--use_standby_frame",
    "force": "--force",
    "pretty": "--pretty",
}


class ReKepRealDriver(BaseDriver):
    """HAL driver that delegates real-world manipulation to the ReKep bridge."""

    def __init__(self, gui: bool = False, **kwargs: Any) -> None:
        self._gui = gui
        self._objects: dict[str, dict] = {}
        self._last_runtime: dict[str, Any] = {}
        self._tool_root = self._resolve_tool_root(kwargs.get("rekep_root"))
        self._bridge_script = self._tool_root / "dobot_bridge.py"
        self._python_bin = str(kwargs.get("rekep_python") or os.environ.get("REKEP_PYTHON") or sys.executable)

        state_dir = kwargs.get("state_dir") or os.environ.get("REKEP_REAL_STATE_DIR")
        self._state_dir = Path(state_dir).expanduser().resolve() if state_dir else _DEFAULT_STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def get_profile_path(self) -> Path:
        return _PROFILES_DIR / "rekep_real.md"

    def load_scene(self, scene: dict[str, dict]) -> None:
        self._objects = dict(scene)

    def execute_action(self, action_type: str, params: dict) -> str:
        action = str(action_type or "").strip().lower()
        params = params if isinstance(params, dict) else {}

        bridge_action = _ACTION_ALIASES.get(action)
        bridge_params = dict(params)
        if bridge_action is None and action in _HIGH_LEVEL_ACTIONS:
            bridge_action = "execute"
            bridge_params.setdefault("instruction", self._build_instruction(action, bridge_params))

        if bridge_action is None:
            return f"Unknown action: {action_type}"

        payload, error = self._invoke_bridge(bridge_action, bridge_params)
        self._last_runtime = {
            "action_type": action_type,
            "bridge_action": bridge_action,
            "ok": bool(payload.get("ok")) if isinstance(payload, dict) else False,
            "error": error or "",
            "payload": payload if isinstance(payload, dict) else {},
        }

        if error:
            return f"ReKep {bridge_action} failed: {error}"
        if not isinstance(payload, dict):
            return f"ReKep {bridge_action} failed: invalid bridge response"
        if not bool(payload.get("ok")):
            reason = self._extract_error(payload)
            return f"ReKep {bridge_action} failed: {reason}"
        return self._format_success(bridge_action, payload)

    def get_scene(self) -> dict[str, dict]:
        scene = dict(self._objects)
        if self._last_runtime:
            scene["_rekep_runtime"] = {
                "action_type": self._last_runtime.get("action_type", ""),
                "bridge_action": self._last_runtime.get("bridge_action", ""),
                "ok": bool(self._last_runtime.get("ok", False)),
                "error": self._last_runtime.get("error", ""),
            }
        return scene

    def _invoke_bridge(self, action: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not self._bridge_script.exists():
            return None, f"bridge script not found: {self._bridge_script}"

        argv: list[str] = [
            self._python_bin,
            "-u",
            str(self._bridge_script),
            action,
            "--state_dir",
            str(self._state_dir),
        ]
        self._append_bridge_args(argv, params)
        timeout_s = self._resolve_timeout_s(action, params)

        try:
            proc = subprocess.run(
                argv,
                cwd=str(self._tool_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout after {timeout_s}s"
        except Exception as exc:
            return None, str(exc)

        payload = self._parse_payload(proc.stdout)
        if payload is None:
            detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
            return None, detail

        if proc.returncode != 0 and not bool(payload.get("ok")):
            return payload, self._extract_error(payload)

        return payload, None

    @staticmethod
    def _resolve_tool_root(explicit_root: Any) -> Path:
        raw = explicit_root or os.environ.get("REKEP_TOOL_ROOT")
        root = Path(raw).expanduser().resolve() if raw else _DEFAULT_TOOL_ROOT
        return root

    @staticmethod
    def _append_bridge_args(argv: list[str], params: dict[str, Any]) -> None:
        for key, flag in _STRING_ARGS.items():
            value = params.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    argv.extend([flag, cleaned])

        for key, flag in _NUMERIC_ARGS.items():
            value = params.get(key)
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                argv.extend([flag, str(value)])
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if key.endswith("_frames") or key.endswith("_tokens") or key.endswith("_limit") or key in {
                "dobot_port",
                "dobot_move_port",
                "max_tokens",
                "retry_limit",
                "camera_warmup_frames",
            }:
                argv.extend([flag, str(int(numeric))])
            else:
                argv.extend([flag, str(numeric)])

        for key, flag in _BOOL_FLAGS.items():
            if bool(params.get(key, False)):
                argv.append(flag)

    @staticmethod
    def _parse_payload(stdout_text: str) -> dict[str, Any] | None:
        text = (stdout_text or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        match = re.search(r"({[\s\S]*})\s*$", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _extract_error(payload: dict[str, Any]) -> str:
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        preflight = payload.get("preflight")
        if isinstance(preflight, dict):
            blockers = preflight.get("blockers")
            if isinstance(blockers, list) and blockers:
                return ", ".join(str(item) for item in blockers[:5])
        return "unknown error"

    @staticmethod
    def _format_success(action: str, payload: dict[str, Any]) -> str:
        if action == "preflight":
            preflight = payload.get("preflight", {})
            status = preflight.get("status", "unknown")
            blockers = preflight.get("blockers", [])
            if isinstance(blockers, list) and blockers:
                return f"ReKep preflight status: {status} ({', '.join(str(x) for x in blockers[:5])})"
            return f"ReKep preflight status: {status}"

        result = payload.get("result")
        if action == "scene_qa" and isinstance(result, dict):
            answer = result.get("answer")
            if isinstance(answer, str) and answer.strip():
                return f"Scene QA: {answer.strip()}"

        if action in {"execute_background", "job_status", "job_cancel", "longrun_start", "longrun_status"}:
            job = payload.get("job")
            if isinstance(job, dict):
                job_id = job.get("job_id", "")
                status = job.get("status", "unknown")
                if job_id:
                    return f"ReKep {action} success: job_id={job_id}, status={status}"
                return f"ReKep {action} success: status={status}"

        if action == "longrun_command":
            command_event = payload.get("command_event")
            if isinstance(command_event, dict):
                cmd = command_event.get("command", "unknown")
                return f"ReKep longrun command accepted: {cmd}"

        return f"ReKep {action} succeeded."

    @staticmethod
    def _resolve_timeout_s(action: str, params: dict[str, Any]) -> int:
        raw = params.get("timeout_s")
        if isinstance(raw, (int, float)) and raw > 0:
            return max(5, int(raw))
        if action in {"preflight", "standby_status", "standby_stop", "job_status", "job_cancel"}:
            return 30
        if action in {"scene_qa", "standby_start"}:
            return 120
        if action in {"execute_background", "longrun_start", "longrun_command", "longrun_stop"}:
            return 180
        if action in {"execute", "longrun_status"}:
            return 1800
        return 120

    @staticmethod
    def _build_instruction(action: str, params: dict[str, Any]) -> str:
        if action == "rekep_instruction":
            instruction = params.get("instruction")
            if isinstance(instruction, str) and instruction.strip():
                return instruction.strip()
            return "complete the requested manipulation task safely"

        if action == "move_to":
            x = params.get("x")
            y = params.get("y")
            z = params.get("z")
            rx = params.get("rx")
            ry = params.get("ry")
            rz = params.get("rz")
            components = [f"x={x}", f"y={y}", f"z={z}"]
            if rx is not None and ry is not None and rz is not None:
                components.extend([f"rx={rx}", f"ry={ry}", f"rz={rz}"])
            return "move the end effector to target pose (" + ", ".join(components) + ")"

        if action == "pick_up":
            target = params.get("target", "target object")
            return f"pick up {target}"
        if action == "put_down":
            target = params.get("target", "held object")
            location = params.get("location", "target location")
            return f"place {target} at {location}"
        if action == "push":
            target = params.get("target", "target object")
            direction = params.get("direction", "forward")
            return f"push {target} toward {direction}"
        if action == "point_to":
            target = params.get("target", "target object")
            return f"point to {target}"
        if action == "open_gripper":
            return "open the gripper"
        if action == "close_gripper":
            return "close the gripper"
        return "complete the requested manipulation task safely"

from __future__ import annotations

import time
from typing import Any, Dict

from robot_adapter import RobotAdapter


class CellbotAdapter(RobotAdapter):
    """Template adapter for integrating a new robot backend.

    Replace the TODO blocks with real SDK / RPC logic for your robot.
    The class intentionally supports dry-run out of the box so integration
    can start without moving hardware.
    """

    def __init__(
        self,
        *,
        host: str = "",
        port: int | None = None,
        driver: str = "cellbot_sdk",
        extras: Dict[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port) if port is not None else None
        self.driver = str(driver or "cellbot_sdk")
        self.extras = extras if isinstance(extras, dict) else {}
        self._connected = False
        self._gripper_closed: bool | None = None
        self._gripper_position: float | None = None
        self._tool_pose: list[float] = []
        self._joint_state: list[float] = []

    def connect(self) -> Dict[str, Any]:
        # TODO: initialize your real robot SDK / socket / RPC session here.
        self._connected = True
        return {
            "ok": True,
            "driver": self.driver,
            "host": self.host,
            "port": self.port,
        }

    def close(self) -> None:
        # TODO: release your real robot connection here.
        self._connected = False

    def get_runtime_state(self) -> Dict[str, Any]:
        return {
            "source": self.driver,
            "connected": self._connected,
            "busy": False,
            "faulted": False,
            "tool_pose": list(self._tool_pose),
            "joint_state": list(self._joint_state),
            "gripper_closed": self._gripper_closed,
            "gripper_position": self._gripper_position,
        }

    def execute_action(self, action: Dict[str, Any], execute_motion: bool = False) -> Dict[str, Any]:
        action = action if isinstance(action, dict) else {}
        action_type = str(action.get("type", "")).strip().lower()

        if action_type == "wait":
            seconds = float(action.get("seconds", action.get("duration_s", 0.5)))
            time.sleep(max(0.0, min(seconds, 30.0)))
            return {
                "ok": True,
                "driver": self.driver,
                "action_type": action_type,
                "executed": True,
                "dry_run": False,
            }

        if not execute_motion:
            return {
                "ok": True,
                "driver": self.driver,
                "action_type": action_type,
                "executed": False,
                "dry_run": True,
            }

        if action_type == "movej":
            joints = action.get("joints", [])
            if isinstance(joints, list):
                self._joint_state = [float(v) for v in joints]
            # TODO: call your SDK joint-space motion API.
        elif action_type == "movel":
            pose = action.get("pose", [])
            if isinstance(pose, list):
                self._tool_pose = [float(v) for v in pose]
            # TODO: call your SDK Cartesian motion API.
        elif action_type == "open_gripper":
            self._gripper_closed = False
            self._gripper_position = 1.0
            # TODO: call your SDK gripper-open API.
        elif action_type == "close_gripper":
            self._gripper_closed = True
            self._gripper_position = 0.0
            # TODO: call your SDK gripper-close API.
        else:
            raise RuntimeError(f"Unsupported action type for {self.driver}: {action_type}")

        return {
            "ok": True,
            "driver": self.driver,
            "action_type": action_type,
            "executed": True,
            "dry_run": False,
        }

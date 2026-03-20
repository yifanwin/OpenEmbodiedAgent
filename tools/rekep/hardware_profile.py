from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class HardwareProfile:
    robot_family: str = "dobot"
    robot_driver: str = "xtrainer_zmq"
    robot_host: str = ""
    robot_port: int | None = None
    robot_move_port: int | None = None
    xtrainer_sdk_dir: str = ""
    camera_family: str = "realsense"
    camera_source: str = "0"
    camera_profile: str = "global3"
    camera_model: str = "D455"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_int(value: Any, fallback: int | None = None) -> int | None:
    if isinstance(value, bool) or value is None:
        return fallback
    try:
        return int(value)
    except Exception:
        return fallback


def _infer_camera_family(camera_source: str) -> str:
    source = str(camera_source)
    lowered = source.lower()
    if "realsense_zmq" in lowered or "rs_zmq" in lowered:
        return "realsense_zmq"
    if "realsense" in lowered or "rs:" in lowered or lowered in {"rs", "d455", "realsense"}:
        return "realsense"
    return "generic"


def build_hardware_profile(
    *,
    robot_driver: str,
    camera_source: str,
    camera_profile: str,
    robot_family: str = "dobot",
    robot_host: str | None = None,
    robot_port: int | None = None,
    robot_move_port: int | None = None,
    xtrainer_sdk_dir: str = "",
    extras: Dict[str, Any] | None = None,
) -> HardwareProfile:
    source = str(camera_source)
    camera_family = _infer_camera_family(source)
    camera_model = "D455" if camera_profile == "global3" or camera_family.startswith("realsense") else "unknown"
    return HardwareProfile(
        robot_family=str(robot_family or "dobot"),
        robot_driver=str(robot_driver),
        robot_host=str(robot_host or ""),
        robot_port=_to_int(robot_port),
        robot_move_port=_to_int(robot_move_port),
        xtrainer_sdk_dir=str(xtrainer_sdk_dir or ""),
        camera_family=camera_family,
        camera_source=source,
        camera_profile=str(camera_profile),
        camera_model=camera_model,
        extras=extras or {},
    )


def coerce_hardware_profile(profile: HardwareProfile | Dict[str, Any] | None) -> HardwareProfile:
    if isinstance(profile, HardwareProfile):
        return profile
    payload = profile if isinstance(profile, dict) else {}
    return build_hardware_profile(
        robot_family=str(payload.get("robot_family", "dobot")),
        robot_driver=str(payload.get("robot_driver", "xtrainer_zmq")),
        robot_host=str(payload.get("robot_host", "")),
        robot_port=_to_int(payload.get("robot_port")),
        robot_move_port=_to_int(payload.get("robot_move_port")),
        xtrainer_sdk_dir=str(payload.get("xtrainer_sdk_dir", "")),
        camera_source=str(payload.get("camera_source", "0")),
        camera_profile=str(payload.get("camera_profile", "global3")),
        extras=payload.get("extras", {}) if isinstance(payload.get("extras"), dict) else {},
    )

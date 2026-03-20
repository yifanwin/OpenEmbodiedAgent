#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = REPO_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from hardware_profile import build_hardware_profile
from robot_factory import create_robot_adapter


# From dobot_xtrainer_remote/experiments/run_control.py (right arm safe zone).
RIGHT_X_RANGE = (-250.0, 450.0)
RIGHT_Y_RANGE = (-750.0, -160.0)
RIGHT_Z_MIN = 40.0


@dataclass
class ActionItem:
    plan_path: Path
    stage: int
    action_index: int
    action: Dict[str, Any]


def _default_host_for_driver(driver: str) -> str:
    return "127.0.0.1" if driver == "xtrainer_zmq" else "192.168.5.1"


def _default_port_for_driver(driver: str) -> int:
    return 6001 if driver == "xtrainer_zmq" else 29999


def _resolve_plan_files(raw_paths: List[str], plan_glob: str) -> List[Path]:
    paths: List[Path] = []
    for raw in raw_paths:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = WORKSPACE_DIR / p
        if p.is_file():
            paths.append(p.resolve())
            continue
        raise FileNotFoundError(f"plan file not found: {raw}")

    if plan_glob:
        for p in sorted(WORKSPACE_DIR.glob(plan_glob)):
            if p.is_file():
                paths.append(p.resolve())

    dedup: List[Path] = []
    seen = set()
    for p in paths:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        dedup.append(p)
    return dedup


def _parse_stage_from_filename(path: Path) -> int:
    stem = path.stem
    marker = "_stage"
    if marker not in stem:
        return 0
    tail = stem.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return 0
    return int("".join(digits))


def _load_actions_from_plan(path: Path) -> List[ActionItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError(f"invalid plan format (missing actions list): {path}")
    stage = _parse_stage_from_filename(path)
    items: List[ActionItem] = []
    for idx, action in enumerate(raw_actions, start=1):
        if not isinstance(action, dict):
            raise ValueError(f"invalid action at {path}#{idx}")
        items.append(ActionItem(plan_path=path, stage=stage, action_index=idx, action=action))
    return items


def _normalize_arm_name(raw: Any, default: str = "right") -> str:
    arm = str(raw if raw is not None else default).strip().lower()
    if arm in {"left", "l"}:
        return "left"
    if arm in {"both", "bimanual", "lr"}:
        return "both"
    return "right"


def _safe_check_movel(action: Dict[str, Any], *, target_arm: str = "right") -> Dict[str, Any]:
    action_type = str(action.get("type", "")).strip().lower()
    result = {"checked": False, "ok": True, "reason": "", "x": None, "y": None, "z": None, "arm": target_arm}
    if action_type != "movel":
        return result
    if target_arm == "left":
        result["checked"] = True
        result["ok"] = True
        result["reason"] = "skip right-arm safe zone check for left-arm action"
        return result
    pose = action.get("pose")
    if not isinstance(pose, list) or len(pose) < 3:
        result["checked"] = True
        result["ok"] = False
        result["reason"] = "movel pose missing xyz"
        return result
    x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
    result.update({"checked": True, "x": x, "y": y, "z": z})
    if not (RIGHT_X_RANGE[0] <= x <= RIGHT_X_RANGE[1]):
        result["ok"] = False
        result["reason"] = f"x out of range {RIGHT_X_RANGE}: {x}"
    elif not (RIGHT_Y_RANGE[0] <= y <= RIGHT_Y_RANGE[1]):
        result["ok"] = False
        result["reason"] = f"y out of range {RIGHT_Y_RANGE}: {y}"
    elif not (z > RIGHT_Z_MIN):
        result["ok"] = False
        result["reason"] = f"z must be > {RIGHT_Z_MIN}: {z}"
    return result


def _format_action(action: Dict[str, Any]) -> str:
    action_type = str(action.get("type", "")).strip().lower()
    if action_type == "movel":
        return f"movel pose={action.get('pose')}"
    if action_type == "movej":
        return f"movej joints={action.get('joints')}"
    if action_type == "wait":
        return f"wait seconds={action.get('seconds', action.get('duration_s', 0.5))}"
    return json.dumps(action, ensure_ascii=False)


def _prompt_next(run_all: bool) -> str:
    if run_all:
        return "y"
    answer = input("[Enter/y] run, [s] skip, [a] run-all, [q] quit > ").strip().lower()
    return answer or "y"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay stage_plan actions one-by-one for manual execution testing."
    )
    parser.add_argument("--plan", action="append", default=[], help="Path to a *.stage_plan.txt file (repeatable).")
    parser.add_argument(
        "--plan-glob",
        default="",
        help="Glob relative to workspace root, e.g. openclaw-runtime/state/rekep/real/frames/execute_pen_stage*.stage_plan.txt",
    )
    parser.add_argument("--dobot-driver", default="xtrainer_zmq", choices=["mock", "dashboard_tcp", "xtrainer_sdk", "xtrainer_zmq"])
    parser.add_argument("--dobot-host", default="")
    parser.add_argument("--dobot-port", type=int, default=0)
    parser.add_argument("--dobot-move-port", type=int, default=30003)
    parser.add_argument(
        "--arm",
        default="right",
        choices=["left", "right", "both"],
        help="Target arm for movej/movel/gripper actions.",
    )
    parser.add_argument(
        "--xtrainer-sdk-dir",
        default=str((WORKSPACE_DIR / ".downloads/third_party/dobot_xtrainer").resolve()),
    )
    parser.add_argument("--execute-motion", action="store_true", help="Actually execute motion commands.")
    parser.add_argument("--no-step-confirm", action="store_true", help="Run without per-step prompt.")
    parser.add_argument("--allow-out-of-safe-zone", action="store_true", help="Do not block movel points outside right-arm safe zone.")
    parser.add_argument("--task-tag", default="manual_replay", help="Tag for output log filename.")
    parser.add_argument("--log-path", default="", help="Optional output json log path.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    plan_files = _resolve_plan_files(args.plan, args.plan_glob)
    if not plan_files:
        parser.error("No plan files found. Use --plan and/or --plan-glob.")

    all_actions: List[ActionItem] = []
    for plan_path in plan_files:
        all_actions.extend(_load_actions_from_plan(plan_path))

    if not all_actions:
        parser.error("No actions found in plan files.")

    host = args.dobot_host.strip() or _default_host_for_driver(args.dobot_driver)
    port = int(args.dobot_port) if int(args.dobot_port) > 0 else _default_port_for_driver(args.dobot_driver)
    move_port = int(args.dobot_move_port)

    hardware_profile = build_hardware_profile(
        robot_family="dobot",
        robot_driver=args.dobot_driver,
        robot_host=host,
        robot_port=port,
        robot_move_port=move_port,
        xtrainer_sdk_dir=str(Path(args.xtrainer_sdk_dir).expanduser().resolve()),
        camera_source="0",
        camera_profile="global3",
    )
    adapter = create_robot_adapter(hardware_profile=hardware_profile)
    conn = adapter.connect()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_path:
        log_path = Path(args.log_path).expanduser().resolve()
    else:
        log_path = (
            WORKSPACE_DIR
            / "openclaw-runtime/state/rekep/real/logs"
            / f"{args.task_tag}_{timestamp}.json"
        ).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[replay] connected: {json.dumps(conn, ensure_ascii=False)}")
    print(f"[replay] plans: {len(plan_files)}, actions: {len(all_actions)}")
    print(f"[replay] execute_motion={bool(args.execute_motion)} driver={args.dobot_driver}")
    print(f"[replay] log_path={log_path}")

    records: List[Dict[str, Any]] = []
    run_all = bool(args.no_step_confirm)
    started_at = time.time()

    try:
        for global_index, item in enumerate(all_actions, start=1):
            action = item.action
            action_type = str(action.get("type", "")).strip().lower()
            effective_arm = _normalize_arm_name(action.get("arm"), default=args.arm)
            safe = _safe_check_movel(action, target_arm=effective_arm)
            action_line = (
                f"[{global_index}/{len(all_actions)}] "
                f"stage={item.stage} file={item.plan_path.name} action#{item.action_index} "
                f"{_format_action(action)}"
            )
            print(action_line)

            if safe["checked"]:
                print(
                    "[safe] "
                    f"arm={safe['arm']} ok={safe['ok']} x={safe['x']} y={safe['y']} z={safe['z']} reason={safe['reason']}"
                )
                if not safe["ok"] and not args.allow_out_of_safe_zone:
                    raise RuntimeError(
                        "movel point is outside right-arm safe zone "
                        "(set --allow-out-of-safe-zone to bypass)"
                    )

            decision = _prompt_next(run_all)
            if decision in {"q", "quit"}:
                print("[replay] aborted by user.")
                break
            if decision in {"a", "all"}:
                run_all = True
                decision = "y"
            if decision in {"s", "skip"}:
                records.append(
                    {
                        "index": global_index,
                        "stage": item.stage,
                        "plan_path": str(item.plan_path),
                        "action_index": item.action_index,
                        "action": action,
                        "skipped": True,
                        "safe_check": safe,
                    }
                )
                continue

            action_payload = dict(action)
            if action_type in {"movej", "movel", "open_gripper", "close_gripper"} and "arm" not in action_payload:
                action_payload["arm"] = args.arm
            result = adapter.execute_action(action_payload, execute_motion=bool(args.execute_motion))
            print(f"[result] ok={result.get('ok')} executed={result.get('executed')} dry_run={result.get('dry_run')}")
            if result.get("command_response") is not None:
                print(f"[result] command_response={result.get('command_response')}")

            records.append(
                {
                    "index": global_index,
                    "stage": item.stage,
                    "plan_path": str(item.plan_path),
                    "action_index": item.action_index,
                    "action": action_payload,
                    "skipped": False,
                    "safe_check": safe,
                    "result": result,
                }
            )
    finally:
        try:
            runtime_state = adapter.get_runtime_state()
        except Exception as exc:
            runtime_state = {"error": str(exc)}
        adapter.close()

        payload = {
            "ok": True,
            "task_tag": args.task_tag,
            "started_at": datetime.fromtimestamp(started_at).isoformat(),
            "finished_at": datetime.now().isoformat(),
            "elapsed_s": time.time() - started_at,
            "execute_motion": bool(args.execute_motion),
            "driver": args.dobot_driver,
            "host": host,
            "port": port,
            "move_port": move_port,
            "plans": [str(p) for p in plan_files],
            "total_actions": len(all_actions),
            "records": records,
            "runtime_state_after_close": runtime_state,
        }
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    executed = sum(1 for row in records if not row.get("skipped"))
    skipped = sum(1 for row in records if row.get("skipped"))
    print(f"[summary] executed={executed} skipped={skipped} total={len(all_actions)}")
    print(f"[summary] log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

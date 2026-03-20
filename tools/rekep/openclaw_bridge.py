#!/usr/bin/env python3
import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from vlm_client import resolve_vlm_config, vlm_ready


REPO_DIR = Path(__file__).resolve().parent
TARGET_STACK = {
    "rekep_repo": "huangwl18/ReKep",
    "isaac_sim": "2023.1.1",
    "omnigibson_commit": "cc0316a0574018a3cb2956fcbff3be75c07cdf0f",
}
TASKS = {
    "pen": {
        "scene_file": str(REPO_DIR / "configs" / "og_scene_file_pen.json"),
        "instruction": "reorient the white pen and drop it upright into the black pen holder",
        "rekep_program_dir": str(REPO_DIR / "vlm_query" / "pen"),
    },
}
MODULES_TO_CHECK = [
    "isaacsim",
    "omnigibson",
    "torch",
    "open3d",
    "trimesh",
    "imageio",
    "cv2",
    "parse",
    "sklearn",
    "kmeans_pytorch",
    "yaml",
    "scipy",
    "numba",
]
ESSENTIAL_MODULE_BLOCKERS = {
    "parse": "parse-missing",
    "kmeans_pytorch": "kmeans-pytorch-missing",
    "sklearn": "scikit-learn-missing",
    "torch": "torch-missing",
    "open3d": "open3d-missing",
    "trimesh": "trimesh-missing",
    "imageio": "imageio-missing",
    "cv2": "opencv-missing",
    "scipy": "scipy-missing",
    "numba": "numba-missing",
}
DEFAULT_JOB_STATE_ENV = "REKEP_JOB_STATE_DIR"
DEFAULT_JOB_STATE_DIR = Path("/tmp/rekep_jobs")
REAL_ACTION_MAP = {
    "real_preflight": "preflight",
    "real_standby_start": "standby_start",
    "real_standby_status": "standby_status",
    "real_standby_stop": "standby_stop",
    "real_scene_qa": "scene_qa",
    "real_execute": "execute",
    "real_execute_background": "execute_background",
    "real_job_status": "job_status",
    "real_job_cancel": "job_cancel",
    "real_longrun_start": "longrun_start",
    "real_longrun_status": "longrun_status",
    "real_longrun_command": "longrun_command",
    "real_longrun_stop": "longrun_stop",
}
DEFAULT_REKEP_CONDA_ENV = os.environ.get("REKEP_CONDA_ENV", "rekep")


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


def is_real_action(action_name):
    return str(action_name).startswith("real_")


def resolve_headless_setting(requested=None):
    if requested is not None:
        return bool(requested)
    from_env = parse_boolish(os.environ.get("OMNIGIBSON_HEADLESS"))
    if from_env is not None:
        return from_env
    return True


def build_runtime_env(headless):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env["REKEP_HEADLESS"] = "1" if headless else "0"
    env["OMNIGIBSON_HEADLESS"] = env["REKEP_HEADLESS"]
    if headless:
        env.pop("OMNIGIBSON_REMOTE_STREAMING", None)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        env["QT_QPA_PLATFORM"] = "offscreen"
    else:
        env.pop("QT_QPA_PLATFORM", None)
    return env


def normalize_runtime_args(args):
    args.headless = resolve_headless_setting(getattr(args, "headless", None))
    if not is_real_action(getattr(args, "action", "")):
        if getattr(args, "visualize", False) and args.headless:
            raise ValueError("visualize requires --no-headless")
        if getattr(args, "hold_ui_seconds", 0.0) and args.headless:
            args.hold_ui_seconds = 0.0
    return args


def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def resolve_rekep_python():
    explicit = os.environ.get("REKEP_PYTHON")
    if explicit:
        return explicit
    conda_root = os.environ.get("CONDA_EXE")
    if conda_root:
        root = Path(conda_root).resolve().parent.parent
        candidate = root / "envs" / DEFAULT_REKEP_CONDA_ENV / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    for root in [Path.home() / "APP" / "miniconda3", Path.home() / "miniconda3", Path.home() / "anaconda3"]:
        candidate = root / "envs" / DEFAULT_REKEP_CONDA_ENV / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_command(argv, timeout=5):
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "argv": argv,
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "argv": argv,
    }


def detect_os_release():
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return {}
    result = {}
    for line in os_release.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def build_preflight(use_cached_query, require_vlm=False, headless=None):
    modules = {name: module_available(name) for name in MODULES_TO_CHECK}
    nvidia_smi = run_command(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        timeout=5,
    )
    os_release = detect_os_release()
    vlm = resolve_vlm_config(default_model="gpt-5.4")
    resolved_headless = resolve_headless_setting(headless)
    blockers = []
    notes = []

    if not nvidia_smi["ok"]:
        blockers.append("nvidia-driver-unavailable")
    if not modules["isaacsim"]:
        blockers.append("isaacsim-missing")
    if not modules["omnigibson"]:
        blockers.append("omnigibson-missing")
    for module_name, blocker_name in ESSENTIAL_MODULE_BLOCKERS.items():
        if not modules[module_name]:
            blockers.append(blocker_name)
    if (require_vlm or not use_cached_query) and not vlm_ready(default_model=vlm["model"]):
        blockers.append("vlm-api-key-missing")
    if os_release.get("VERSION_ID") == "24.04":
        notes.append(
            "This repo targets Isaac Sim 2023.1.1 + a pinned OmniGibson commit; this machine runs a locally patched compatibility stack on Ubuntu 24.04."
        )

    return {
        "status": "ready" if not blockers else "blocked",
        "repo_dir": str(REPO_DIR),
        "cwd": os.getcwd(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "rekep_executable": resolve_rekep_python(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "os_release": os_release,
        },
        "display": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
            "headless_requested": resolved_headless,
        },
        "vlm": {
            "configured": bool(vlm["api_key"]),
            "model": vlm["model"],
            "base_url": vlm["base_url"],
            "api_key_env": vlm["api_key_env"],
        },
        "modules": modules,
        "nvidia_smi": nvidia_smi,
        "tasks": sorted(TASKS.keys()),
        "target_stack": TARGET_STACK,
        "blockers": blockers,
        "notes": notes,
    }


def print_json(payload, pretty):
    indent = 2 if pretty else None
    print(json.dumps(payload, indent=indent, ensure_ascii=False))


def emit_progress(message):
    print(message, file=sys.stderr, flush=True)


def build_run_log_path(task_name):
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("/tmp") / f"rekep_bridge_{task_name}_{timestamp}.log"


def iso_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_iso_timestamp(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def resolve_job_state_dir(explicit_dir=None):
    raw_dir = explicit_dir or os.environ.get(DEFAULT_JOB_STATE_ENV)
    job_state_dir = Path(raw_dir) if raw_dir else DEFAULT_JOB_STATE_DIR
    job_state_dir.mkdir(parents=True, exist_ok=True)
    return job_state_dir


def build_job_state_path(job_state_dir, job_id):
    return Path(job_state_dir) / f"{job_id}.json"


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def load_job_state(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_job_state(path, job_state):
    job_state = dict(job_state)
    job_state["updated_at"] = iso_now()
    atomic_write_json(path, job_state)
    return job_state


def update_job_state(path, **updates):
    job_state = load_job_state(path) if path.exists() else {}
    job_state.update(updates)
    return save_job_state(path, job_state)


def resolve_latest_job_state_path(job_state_dir):
    candidates = sorted(Path(job_state_dir).glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


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


def extract_video_path_from_log(log_text):
    marker = "Video saved to "
    if not log_text or marker not in log_text:
        return ""
    line = log_text.split(marker)[-1].splitlines()[0].strip()
    return line


def parse_json_payload_from_text(text):
    trimmed = (text or "").strip()
    if not trimmed:
        return None
    try:
        parsed = json.loads(trimmed)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"({[\s\S]*}|\[[\s\S]*])\s*$", trimmed)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def is_pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_task(task_name, use_cached_query, instruction_override=None, scene_file_override=None, rekep_program_dir_override=None):
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {sorted(TASKS.keys())}")
    task = TASKS[task_name]
    return {
        "task": task_name,
        "instruction": instruction_override or task["instruction"],
        "scene_file": os.path.abspath(scene_file_override or task["scene_file"]),
        "rekep_program_dir": (
            os.path.abspath(rekep_program_dir_override)
            if rekep_program_dir_override
            else (os.path.abspath(task["rekep_program_dir"]) if use_cached_query else None)
        ),
    }


def find_latest_program_dir(started_at, existing_program_dirs):
    query_dir = REPO_DIR / "vlm_query"
    if not query_dir.exists():
        return None
    candidates = []
    for path in query_dir.iterdir():
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved in existing_program_dirs:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime + 1e-6 < started_at:
            continue
        candidates.append((stat.st_mtime, path))
    if not candidates:
        return None
    candidates.sort()
    return str(candidates[-1][1].resolve())


def collect_program_artifacts(program_dir):
    if not program_dir:
        return {}
    path = Path(program_dir)
    if not path.exists():
        return {}
    artifacts = {
        "rekep_program_dir": str(path.resolve()),
    }
    for key, name in (
        ("vlm_prompt_path", "prompt.txt"),
        ("vlm_raw_output_path", "output_raw.txt"),
        ("vlm_trace_path", "vlm_trace.json"),
        ("metadata_path", "metadata.json"),
    ):
        candidate = path / name
        if candidate.exists():
            artifacts[key] = str(candidate.resolve())
    return artifacts


def find_latest_video(started_at, existing_videos):
    videos_dir = REPO_DIR / "videos"
    if not videos_dir.exists():
        return None
    candidates = []
    for path in videos_dir.glob("*.mp4"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if str(path) in existing_videos:
            continue
        if stat.st_mtime + 1e-6 < started_at:
            continue
        candidates.append((stat.st_mtime, path))
    if not candidates:
        return None
    candidates.sort()
    return str(candidates[-1][1].resolve())


def run_task(args, run_log_path=None):
    preflight = build_preflight(args.use_cached_query, headless=args.headless)
    if preflight["status"] != "ready" and not args.force:
        return {
            "ok": False,
            "action": "run",
            "error": "Simulation preflight failed",
            "preflight": preflight,
        }, 2

    resolved_task = resolve_task(
        task_name=args.task,
        use_cached_query=args.use_cached_query,
        instruction_override=args.instruction,
        scene_file_override=args.scene_file,
        rekep_program_dir_override=args.rekep_program_dir,
    )
    run_log_path = Path(run_log_path) if run_log_path else build_run_log_path(args.task)
    started_at = dt.datetime.now().timestamp()
    existing_videos = {str(path.resolve()) for path in (REPO_DIR / "videos").glob("*.mp4")} if (REPO_DIR / "videos").exists() else set()
    existing_program_dirs = {str(path.resolve()) for path in (REPO_DIR / "vlm_query").glob("*") if path.is_dir()} if (REPO_DIR / "vlm_query").exists() else set()

    worker_argv = [
        sys.executable,
        "-u",
        str(REPO_DIR / "main.py"),
        "--task",
        args.task,
    ]
    if args.use_cached_query:
        worker_argv.append("--use_cached_query")
    if args.apply_disturbance:
        worker_argv.append("--apply_disturbance")
    if args.visualize:
        worker_argv.append("--visualize")
    worker_argv.append("--headless" if args.headless else "--no-headless")
    if args.hold_ui_seconds > 0:
        worker_argv.extend(["--hold_ui_seconds", str(args.hold_ui_seconds)])
    if args.instruction:
        worker_argv.extend(["--instruction", args.instruction])
    if args.scene_file:
        worker_argv.extend(["--scene_file", args.scene_file])
    if args.rekep_program_dir:
        worker_argv.extend(["--rekep_program_dir", args.rekep_program_dir])

    emit_progress(f"[bridge] run_log_path: {run_log_path}")
    emit_progress(f"[bridge] launch: {' '.join(worker_argv)}")
    emit_progress(
        f"[bridge] runtime: headless={args.headless} visualize={args.visualize} hold_ui_seconds={args.hold_ui_seconds}"
    )
    notes = preflight.get("notes", [])
    for note in notes:
        emit_progress(f"[bridge] note: {note}")

    with run_log_path.open("w", encoding="utf-8") as run_log:
        proc = subprocess.Popen(
            worker_argv,
            cwd=str(REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=build_runtime_env(args.headless),
        )
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    run_log.write(line)
                    run_log.flush()
                    emit_progress(line.rstrip("\n"))
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        returncode = proc.wait()

    if returncode != 0:
        return {
            "ok": False,
            "action": "run",
            "error": f"Simulation failed with exit code {returncode}",
            "preflight": preflight,
            "run_log_path": str(run_log_path),
        }, 1

    video_path = find_latest_video(started_at, existing_videos)
    if not video_path:
        return {
            "ok": False,
            "action": "run",
            "error": "Simulation finished but no new video artifact was found",
            "preflight": preflight,
            "run_log_path": str(run_log_path),
        }, 1

    return {
        "ok": True,
        "action": "run",
        "preflight": preflight,
        "result": {
            **resolved_task,
            "video_path": video_path,
            **collect_program_artifacts(
                resolved_task["rekep_program_dir"]
                if resolved_task["rekep_program_dir"]
                else find_latest_program_dir(started_at, existing_program_dirs)
            ),
        },
        "run_log_path": str(run_log_path),
    }, 0


def create_background_job_state(args, job_id, job_state_path, run_log_path, preflight, resolved_task):
    now = iso_now()
    return {
        "job_id": job_id,
        "kind": "run",
        "status": "queued",
        "task": args.task,
        "use_cached_query": bool(args.use_cached_query),
        "instruction": resolved_task["instruction"],
        "scene_file": resolved_task["scene_file"],
        "rekep_program_dir": resolved_task["rekep_program_dir"],
        "apply_disturbance": bool(args.apply_disturbance),
        "visualize": bool(args.visualize),
        "headless": bool(args.headless),
        "hold_ui_seconds": float(args.hold_ui_seconds),
        "force": bool(args.force),
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "process_group_id": None,
        "run_log_path": str(run_log_path),
        "status_path": str(job_state_path),
        "preflight": preflight,
        "result": None,
        "error": None,
        "exit_code": None,
    }


def build_job_worker_argv(args, job_id, job_state_dir, run_log_path):
    worker_argv = [
        sys.executable,
        "-u",
        str(REPO_DIR / "openclaw_bridge.py"),
        "job_worker",
        "--job_id",
        job_id,
        "--job_state_dir",
        str(job_state_dir),
        "--task",
        args.task,
        "--run_log_path",
        str(run_log_path),
    ]
    if args.use_cached_query:
        worker_argv.append("--use_cached_query")
    if args.apply_disturbance:
        worker_argv.append("--apply_disturbance")
    if args.visualize:
        worker_argv.append("--visualize")
    worker_argv.append("--headless" if args.headless else "--no-headless")
    if args.hold_ui_seconds > 0:
        worker_argv.extend(["--hold_ui_seconds", str(args.hold_ui_seconds)])
    if args.instruction:
        worker_argv.extend(["--instruction", args.instruction])
    if args.scene_file:
        worker_argv.extend(["--scene_file", args.scene_file])
    if args.rekep_program_dir:
        worker_argv.extend(["--rekep_program_dir", args.rekep_program_dir])
    if args.force:
        worker_argv.append("--force")
    return worker_argv


def run_background_task(args):
    preflight = build_preflight(args.use_cached_query, headless=args.headless)
    if preflight["status"] != "ready" and not args.force:
        return {
            "ok": False,
            "action": "run_background",
            "error": "Simulation preflight failed",
            "preflight": preflight,
        }, 2

    resolved_task = resolve_task(
        task_name=args.task,
        use_cached_query=args.use_cached_query,
        instruction_override=args.instruction,
        scene_file_override=args.scene_file,
        rekep_program_dir_override=args.rekep_program_dir,
    )
    job_state_dir = resolve_job_state_dir(args.job_state_dir)
    job_id = args.job_id or f"rekep-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_log_path = Path(args.run_log_path) if args.run_log_path else build_run_log_path(args.task)
    job_state_path = build_job_state_path(job_state_dir, job_id)
    job_state = create_background_job_state(args, job_id, job_state_path, run_log_path, preflight, resolved_task)
    save_job_state(job_state_path, job_state)

    worker_argv = build_job_worker_argv(args, job_id, job_state_dir, run_log_path)
    try:
        proc = subprocess.Popen(
            worker_argv,
            cwd=str(REPO_DIR),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={
                **build_runtime_env(args.headless),
                DEFAULT_JOB_STATE_ENV: str(job_state_dir),
            },
        )
    except Exception as exc:
        update_job_state(
            job_state_path,
            status="failed",
            finished_at=iso_now(),
            error=f"Failed to launch background worker: {exc}",
            exit_code=1,
        )
        return {
            "ok": False,
            "action": "run_background",
            "error": f"Failed to launch background worker: {exc}",
            "preflight": preflight,
            "job": load_job_state(job_state_path),
        }, 1

    job_state = update_job_state(
        job_state_path,
        status="running",
        started_at=iso_now(),
        pid=proc.pid,
        process_group_id=proc.pid,
    )
    return {
        "ok": True,
        "action": "run_background",
        "job": job_state,
    }, 0


def run_job_worker(args):
    job_state_dir = resolve_job_state_dir(args.job_state_dir)
    job_state_path = build_job_state_path(job_state_dir, args.job_id)
    current_state = load_job_state(job_state_path) if job_state_path.exists() else {}
    update_job_state(
        job_state_path,
        status="running",
        started_at=current_state.get("started_at") or iso_now(),
        pid=os.getpid(),
        process_group_id=os.getpgrp(),
        run_log_path=args.run_log_path or current_state.get("run_log_path"),
        error=None,
    )
    try:
        payload, exit_code = run_task(args, run_log_path=args.run_log_path)
    except Exception as exc:
        payload = {
            "ok": False,
            "action": "run",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_log_path": args.run_log_path,
        }
        exit_code = 1

    latest_state = load_job_state(job_state_path) if job_state_path.exists() else {}
    if latest_state.get("status") == "cancelled":
        return {
            "ok": False,
            "action": "job_worker",
            "error": "Job was cancelled",
            "job": latest_state,
        }, 1

    terminal_status = "succeeded" if exit_code == 0 and payload.get("ok") else "failed"
    update_job_state(
        job_state_path,
        status=terminal_status,
        finished_at=iso_now(),
        exit_code=exit_code,
        error=None if terminal_status == "succeeded" else payload.get("error", "unknown error"),
        result=payload.get("result"),
        preflight=payload.get("preflight", latest_state.get("preflight")),
        run_log_path=payload.get("run_log_path", args.run_log_path or latest_state.get("run_log_path")),
        payload=payload,
    )
    return payload, exit_code


def run_job_status(args):
    job_state_dir = resolve_job_state_dir(args.job_state_dir)
    job_state_path = (
        build_job_state_path(job_state_dir, args.job_id)
        if args.job_id
        else resolve_latest_job_state_path(job_state_dir)
    )
    if job_state_path is None or not job_state_path.exists():
        return {
            "ok": False,
            "action": "job_status",
            "error": "No ReKep background jobs were found" if not args.job_id else f"Unknown job id: {args.job_id}",
        }, 1

    job_state = load_job_state(job_state_path)
    log_tail = read_log_tail(job_state.get("run_log_path"), max_chars=64000)
    pid = job_state.get("pid")
    if job_state.get("status") == "running":
        worker_alive = is_pid_alive(pid) if isinstance(pid, int) else False
        job_state["worker_alive"] = worker_alive
        if not worker_alive:
            terminal_updates = {
                "finished_at": iso_now(),
            }
            recovered_video_path = extract_video_path_from_log(log_tail)
            if recovered_video_path and os.path.exists(recovered_video_path):
                result = job_state.get("result") if isinstance(job_state.get("result"), dict) else {}
                result = {
                    **result,
                    "task": result.get("task", job_state.get("task")),
                    "instruction": result.get("instruction", job_state.get("instruction")),
                    "scene_file": result.get("scene_file", job_state.get("scene_file")),
                    "video_path": os.path.abspath(recovered_video_path),
                }
                terminal_updates.update(
                    {
                        "status": "succeeded",
                        "exit_code": 0,
                        "error": None,
                        "result": result,
                    }
                )
            else:
                terminal_updates.update(
                    {
                        "status": "failed",
                        "exit_code": 1,
                        "error": "Background worker exited unexpectedly before reporting completion",
                    }
                )
            update_job_state(job_state_path, **terminal_updates)
            job_state = load_job_state(job_state_path)
            log_tail = read_log_tail(job_state.get("run_log_path"), max_chars=64000)

    program_dir = ""
    result = job_state.get("result")
    if isinstance(result, dict):
        program_dir = result.get("rekep_program_dir") or ""
    if not program_dir and job_state.get("use_cached_query"):
        program_dir = job_state.get("rekep_program_dir") or ""
    if not program_dir and not job_state.get("use_cached_query"):
        started_at = parse_iso_timestamp(job_state.get("started_at")) or parse_iso_timestamp(job_state.get("created_at")) or 0.0
        program_dir = find_latest_program_dir(started_at, set())
    artifacts = collect_program_artifacts(program_dir)
    return {
        "ok": True,
        "action": "job_status",
        "job": job_state,
        "artifacts": artifacts,
        "log_tail": log_tail[-12000:] if log_tail else "",
    }, 0


def run_job_cancel(args):
    job_state_dir = resolve_job_state_dir(args.job_state_dir)
    job_state_path = (
        build_job_state_path(job_state_dir, args.job_id)
        if args.job_id
        else resolve_latest_job_state_path(job_state_dir)
    )
    if job_state_path is None or not job_state_path.exists():
        return {
            "ok": False,
            "action": "job_cancel",
            "error": "No ReKep background jobs were found" if not args.job_id else f"Unknown job id: {args.job_id}",
        }, 1

    job_state = load_job_state(job_state_path)
    status = job_state.get("status")
    if status in {"succeeded", "failed", "cancelled"}:
        return {
            "ok": True,
            "action": "job_cancel",
            "job": job_state,
        }, 0

    pgid = job_state.get("process_group_id")
    pid = job_state.get("pid")
    try:
        if isinstance(pgid, int) and pgid > 0:
            os.killpg(pgid, signal.SIGKILL)
        elif isinstance(pid, int) and pid > 0:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    job_state = update_job_state(
        job_state_path,
        status="cancelled",
        finished_at=iso_now(),
        error="Cancelled by user",
        exit_code=130,
    )
    return {
        "ok": True,
        "action": "job_cancel",
        "job": job_state,
        "log_tail": read_log_tail(job_state.get("run_log_path")),
    }, 0


def run_scene_qa(args):
    preflight = build_preflight(use_cached_query=True, require_vlm=True, headless=args.headless)
    if preflight["status"] != "ready" and not args.force:
        return {
            "ok": False,
            "action": "scene_qa",
            "error": "Scene QA preflight failed",
            "preflight": preflight,
        }, 2

    if not args.question or not args.question.strip():
        return {
            "ok": False,
            "action": "scene_qa",
            "error": "A non-empty --question is required for scene_qa",
            "preflight": preflight,
        }, 2

    resolved_task = resolve_task(
        task_name=args.task,
        use_cached_query=True,
        instruction_override=args.instruction,
        scene_file_override=args.scene_file,
        rekep_program_dir_override=args.rekep_program_dir,
    )
    run_log_path = build_run_log_path(f"{args.task}_scene_qa")
    emit_progress(f"[bridge] run_log_path: {run_log_path}")
    emit_progress(
        f"[bridge] scene_qa task={args.task} camera_id={args.camera_id} question={args.question.strip()!r}"
    )
    for note in preflight.get("notes", []):
        emit_progress(f"[bridge] note: {note}")

    result_path = run_log_path.with_suffix(".result.json")
    worker_argv = [
        sys.executable,
        "-u",
        str(REPO_DIR / "scene_qa_worker.py"),
        "--scene_file",
        resolved_task["scene_file"],
        "--question",
        args.question.strip(),
        "--camera_id",
        str(args.camera_id),
        "--result_file",
        str(result_path),
    ]
    worker_argv.append("--headless" if args.headless else "--no-headless")
    emit_progress(f"[bridge] launch: {' '.join(worker_argv)}")
    emit_progress(f"[bridge] runtime: headless={args.headless}")

    with run_log_path.open("w", encoding="utf-8") as run_log:
        proc = subprocess.Popen(
            worker_argv,
            cwd=str(REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=build_runtime_env(args.headless),
        )
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    run_log.write(line)
                    run_log.flush()
                    emit_progress(line.rstrip("\n"))
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        returncode = proc.wait()

    if returncode != 0:
        return {
            "ok": False,
            "action": "scene_qa",
            "error": f"Scene QA failed with exit code {returncode}",
            "preflight": preflight,
            "run_log_path": str(run_log_path),
        }, 1

    if not result_path.exists():
        return {
            "ok": False,
            "action": "scene_qa",
            "error": "Scene QA finished but no result artifact was found",
            "preflight": preflight,
            "run_log_path": str(run_log_path),
        }, 1

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "action": "scene_qa",
            "error": f"Failed to parse scene QA result: {exc}",
            "preflight": preflight,
            "run_log_path": str(run_log_path),
        }, 1
    finally:
        try:
            result_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            emit_progress(f"[bridge] note: failed to delete scene QA result file {result_path}")

    if not isinstance(result, dict):
        return {
            "ok": False,
            "action": "scene_qa",
            "error": "Scene QA result payload is not a JSON object",
            "preflight": preflight,
            "run_log_path": str(run_log_path),
        }, 1

    return {
        "ok": True,
        "action": "scene_qa",
        "preflight": preflight,
        "result": {
            **resolved_task,
            **result,
        },
        "run_log_path": str(run_log_path),
    }, 0


def build_real_worker_argv(args):
    if args.action not in REAL_ACTION_MAP:
        raise ValueError(f"Unsupported real action: {args.action}")
    action = REAL_ACTION_MAP[args.action]
    worker_argv = [
        resolve_rekep_python(),
        "-u",
        str(REPO_DIR / "dobot_bridge.py"),
        action,
    ]
    state_dir = args.real_state_dir or args.job_state_dir
    if state_dir:
        worker_argv.extend(["--state_dir", state_dir])
    if args.job_id:
        worker_argv.extend(["--job_id", args.job_id])
    if args.task:
        worker_argv.extend(["--task", args.task])
    if args.question:
        worker_argv.extend(["--question", args.question])
    if args.instruction:
        worker_argv.extend(["--instruction", args.instruction])
    if args.command:
        worker_argv.extend(["--command", args.command])
    if args.command_text:
        worker_argv.extend(["--command_text", args.command_text])
    if args.camera_profile:
        worker_argv.extend(["--camera_profile", args.camera_profile])
    if args.camera_serial:
        worker_argv.extend(["--camera_serial", args.camera_serial])
    if args.dobot_settings_ini:
        worker_argv.extend(["--dobot_settings_ini", args.dobot_settings_ini])
    if args.camera_extrinsic_script:
        worker_argv.extend(["--camera_extrinsic_script", args.camera_extrinsic_script])
    if args.realsense_calib_dir:
        worker_argv.extend(["--realsense_calib_dir", args.realsense_calib_dir])
    if args.camera_source:
        worker_argv.extend(["--camera_source", args.camera_source])
    if args.camera_timeout_s is not None:
        worker_argv.extend(["--camera_timeout_s", str(args.camera_timeout_s)])
    if args.camera_warmup_frames is not None:
        worker_argv.extend(["--camera_warmup_frames", str(args.camera_warmup_frames)])
    if args.use_standby_frame:
        worker_argv.append("--use_standby_frame")
    if args.dobot_driver:
        worker_argv.extend(["--dobot_driver", args.dobot_driver])
    if args.dobot_host:
        worker_argv.extend(["--dobot_host", args.dobot_host])
    if args.dobot_port:
        worker_argv.extend(["--dobot_port", str(args.dobot_port)])
    if args.dobot_move_port:
        worker_argv.extend(["--dobot_move_port", str(args.dobot_move_port)])
    if args.xtrainer_sdk_dir:
        worker_argv.extend(["--xtrainer_sdk_dir", args.xtrainer_sdk_dir])
    if args.interval_s is not None:
        worker_argv.extend(["--interval_s", str(args.interval_s)])
    if args.standby_stale_timeout_s is not None:
        worker_argv.extend(["--standby_stale_timeout_s", str(args.standby_stale_timeout_s)])
    if args.execute_motion:
        worker_argv.append("--execute_motion")
    if args.model:
        worker_argv.extend(["--model", args.model])
    if args.temperature is not None:
        worker_argv.extend(["--temperature", str(args.temperature)])
    if args.max_tokens is not None:
        worker_argv.extend(["--max_tokens", str(args.max_tokens)])
    if args.max_runtime_minutes is not None:
        worker_argv.extend(["--max_runtime_minutes", str(args.max_runtime_minutes)])
    if args.monitor_interval_s is not None:
        worker_argv.extend(["--monitor_interval_s", str(args.monitor_interval_s)])
    if args.retry_limit is not None:
        worker_argv.extend(["--retry_limit", str(args.retry_limit)])
    if args.feishu_webhook:
        worker_argv.extend(["--feishu_webhook", args.feishu_webhook])
    if args.force:
        worker_argv.append("--force")
    return worker_argv, state_dir


def run_real_action(args):
    worker_argv, state_dir = build_real_worker_argv(args)
    emit_progress(f"[bridge] real launch: {' '.join(worker_argv)}")
    if state_dir:
        emit_progress(f"[bridge] real state_dir: {state_dir}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if state_dir:
        env["REKEP_REAL_STATE_DIR"] = os.path.abspath(state_dir)

    output_chunks = []
    proc = subprocess.Popen(
        worker_argv,
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                output_chunks.append(line)
                emit_progress(line.rstrip("\n"))
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    returncode = proc.wait()
    combined_output = "".join(output_chunks)
    payload = parse_json_payload_from_text(combined_output)
    if payload is None:
        payload = {
            "ok": False,
            "action": args.action,
            "error": "Real bridge returned invalid JSON output",
            "raw_output_tail": combined_output[-4000:],
        }
    payload["action"] = args.action

    job = payload.get("job")
    if isinstance(job, dict):
        status_path = job.get("status_path")
        state_path = job.get("state_path")
        if not status_path and state_path:
            job["status_path"] = state_path
        payload["job"] = job

    if returncode != 0:
        payload["ok"] = False
        payload.setdefault("error", f"Real bridge failed with exit code {returncode}")
        return payload, returncode
    if not payload.get("ok"):
        return payload, 1
    return payload, 0


def main():
    parser = argparse.ArgumentParser(description="OpenClaw bridge for ReKep simulation + real robot runtime")
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "list_tasks",
            "run",
            "run_background",
            "job_status",
            "job_cancel",
            "job_worker",
            "scene_qa",
            "real_preflight",
            "real_standby_start",
            "real_standby_status",
            "real_standby_stop",
            "real_scene_qa",
            "real_execute",
            "real_execute_background",
            "real_job_status",
            "real_job_cancel",
            "real_longrun_start",
            "real_longrun_status",
            "real_longrun_command",
            "real_longrun_stop",
        ],
    )
    parser.add_argument("--task", type=str, default="pen")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--command", type=str, default=None)
    parser.add_argument("--command_text", type=str, default=None)
    parser.add_argument("--camera_profile", type=str, default=None)
    parser.add_argument("--camera_serial", type=str, default=None)
    parser.add_argument("--dobot_settings_ini", type=str, default=None)
    parser.add_argument("--camera_extrinsic_script", type=str, default=None)
    parser.add_argument("--realsense_calib_dir", type=str, default=None)
    parser.add_argument("--scene_file", type=str, default=None)
    parser.add_argument("--rekep_program_dir", type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--job_state_dir", type=str, default=None)
    parser.add_argument("--real_state_dir", type=str, default=None)
    parser.add_argument("--run_log_path", type=str, default=None)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--camera_source", type=str, default=None)
    parser.add_argument("--camera_timeout_s", type=float, default=8.0)
    parser.add_argument("--camera_warmup_frames", type=int, default=6)
    parser.add_argument("--use_standby_frame", action="store_true")
    parser.add_argument("--dobot_driver", type=str, default=None)
    parser.add_argument("--dobot_host", type=str, default=None)
    parser.add_argument("--dobot_port", type=int, default=None)
    parser.add_argument("--dobot_move_port", type=int, default=None)
    parser.add_argument("--xtrainer_sdk_dir", type=str, default=None)
    parser.add_argument("--interval_s", type=float, default=0.2)
    parser.add_argument("--standby_stale_timeout_s", type=float, default=15.0)
    parser.add_argument("--execute_motion", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_runtime_minutes", type=float, default=30.0)
    parser.add_argument("--monitor_interval_s", type=float, default=2.0)
    parser.add_argument("--retry_limit", type=int, default=2)
    parser.add_argument("--feishu_webhook", type=str, default="")
    parser.add_argument("--use_cached_query", action="store_true")
    parser.add_argument("--apply_disturbance", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--hold_ui_seconds", type=float, default=0.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    args = normalize_runtime_args(args)

    try:
        if args.action == "preflight":
            payload = {
                "ok": True,
                "action": "preflight",
                "preflight": build_preflight(args.use_cached_query, headless=args.headless),
            }
            exit_code = 0
        elif args.action == "list_tasks":
            payload = {
                "ok": True,
                "action": "list_tasks",
                "tasks": sorted(TASKS.keys()),
                "preflight": build_preflight(args.use_cached_query, headless=args.headless),
            }
            exit_code = 0
        elif args.action == "run_background":
            payload, exit_code = run_background_task(args)
        elif args.action == "job_status":
            payload, exit_code = run_job_status(args)
        elif args.action == "job_cancel":
            payload, exit_code = run_job_cancel(args)
        elif args.action == "job_worker":
            payload, exit_code = run_job_worker(args)
        elif args.action == "scene_qa":
            payload, exit_code = run_scene_qa(args)
        elif is_real_action(args.action):
            payload, exit_code = run_real_action(args)
        else:
            payload, exit_code = run_task(args)
    except Exception as exc:
        payload = {
            "ok": False,
            "action": args.action,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    print_json(payload, args.pretty)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

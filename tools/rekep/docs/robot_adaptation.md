# ReKep Robot Adaptation (OpenEmbodiedAgent)

> Goal: integrate your own robot backend and keep the same runtime loop:
> `preflight -> scene understanding -> execution -> online monitoring -> long-horizon tasks`.

## 1) Unified Contract

Your adapter should follow `tools/rekep/robot_adapter.py`:

1. `connect()`
2. `close()`
3. `get_runtime_state()`
4. `execute_action(action, execute_motion=False)`

Required action protocol:

1. `movej`
2. `movel`
3. `open_gripper`
4. `close_gripper`
5. `wait`

## 2) Create Adapter File

Template file already included:

- `tools/rekep/cellbot_adapter.py`

Duplicate and replace SDK calls for your robot.

## 3) Register Adapter Family

`tools/rekep/robot_factory.py` now supports pluggable factory registration:

- Built-in families: `dobot`, `cellbot`
- Runtime registration API:
  - `register_adapter_factory(robot_family, factory, overwrite=False)`
  - `unregister_adapter_factory(robot_family)`
  - `list_adapter_families()`

For a custom family, register your factory before `create_robot_adapter(...)` is called.

## 4) Bridge Parameters

`tools/rekep/dobot_bridge.py` supports generic robot args (while keeping legacy `dobot_*`):

- `--robot_family`
- `--robot_driver`
- `--robot_host`
- `--robot_port`
- `--robot_move_port`

Environment variables are also supported:

- `REKEP_ROBOT_FAMILY`
- `REKEP_ROBOT_DRIVER`
- `REKEP_ROBOT_HOST`
- `REKEP_ROBOT_PORT`
- `REKEP_ROBOT_MOVE_PORT`

## 5) Validation Commands

Use `cellbot` as an example:

```bash
python tools/rekep/dobot_bridge.py preflight \
  --robot_family cellbot \
  --robot_driver cellbot_sdk \
  --robot_host 127.0.0.1 \
  --robot_port 9000 \
  --camera_source "0" \
  --pretty
```

```bash
python tools/rekep/dobot_bridge.py execute \
  --robot_family cellbot \
  --robot_driver cellbot_sdk \
  --robot_host 127.0.0.1 \
  --robot_port 9000 \
  --instruction "perform a small safe motion test" \
  --pretty
```

```bash
python tools/rekep/dobot_bridge.py execute \
  --robot_family cellbot \
  --robot_driver cellbot_sdk \
  --robot_host 127.0.0.1 \
  --robot_port 9000 \
  --instruction "perform a small safe real-robot motion test" \
  --execute_motion \
  --pretty
```

## 6) Camera Notes

If you are not using RealSense, extend:

- `tools/rekep/camera_factory.py`
- `tools/rekep/camera_adapter.py`

Make sure your `capture_rgbd()` output is normalized:

- `rgb: np.ndarray`
- `depth: np.ndarray`
- `capture_info: dict`

## 7) Calibration Notes

Reuse or mirror the structure under `tools/rekep/real_calibration/` for your camera profile and extrinsics.

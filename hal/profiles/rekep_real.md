# Robot Embodiment Declaration — ReKep Real Runtime (OpenClaw)

> Profile: rekep_real | Driver: ReKepRealDriver

## Identity

- **Name**: ReKep Real Manipulation Runtime
- **Type**: Fixed-base manipulator runtime (default Dobot XTrainer backend)
- **Execution Bridge**: `tools/rekep/dobot_bridge.py`
- **Primary Driver Modes**:
  - `xtrainer_zmq` (recommended): REQ/REP bridge to remote robot server
  - `xtrainer_sdk` (optional): local vendor SDK
  - `dashboard_tcp` / `mock` (debug only)

## Supported Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `real_preflight` / `preflight` | optional runtime params | Verify robot/camera/VLM/runtime dependencies |
| `real_standby_start` / `standby_start` | `camera_source`, `interval_s` | Start background standby video stream |
| `real_standby_status` / `standby_status` | — | Query standby worker heartbeat |
| `real_standby_stop` / `standby_stop` | — | Stop standby worker |
| `real_scene_qa` / `scene_qa` | `question`, optional frame/camera params | Ask VLM about the latest scene frame |
| `real_execute` / `execute` | `instruction`, `execute_motion` | Run one manipulation task |
| `real_execute_background` / `execute_background` | `instruction`, `execute_motion` | Launch manipulation task asynchronously |
| `real_job_status` / `job_status` | `job_id` | Query execute background job |
| `real_job_cancel` / `job_cancel` | `job_id` | Cancel execute background job |
| `real_longrun_start` / `longrun_start` | `instruction`, `execute_motion` | Start long-horizon execution loop |
| `real_longrun_status` / `longrun_status` | `job_id` | Query longrun state |
| `real_longrun_command` / `longrun_command` | `job_id`, `command`, `command_text` | Human intervention in longrun loop |
| `real_longrun_stop` / `longrun_stop` | `job_id` | Stop longrun loop |

## High-Level Action Compatibility

`ReKepRealDriver` also accepts these generic actions and maps them to `execute` with generated natural-language instructions:

- `move_to`
- `pick_up`
- `put_down`
- `push`
- `point_to`
- `open_gripper`
- `close_gripper`
- `rekep_instruction`

## Safety Defaults

- `execute_motion` is `False` unless explicitly provided.
- Default `rekep_execution_mode` is `vlm_stage` (solver mode requires extra dependencies).
- Runtime state files are isolated under `/tmp/rekep_real_state` by default.

## Required Files

- ReKep runtime code and configs in `tools/rekep/`
- Third-party runtime dependencies in `tools/rekep/third_party/`

# ReKep Real Runtime for OpenEmbodiedAgent

本目录提供 **OpenEmbodiedAgent** 下可直接使用的真机 ReKep 运行时（来源于 `~/model/clawkep/ReKep` 的迁移版），并将真机所需第三方代码统一收敛到 `tools/rekep/third_party/`，便于仓库独立发布。

## 1. 目录结构

- `dobot_bridge.py`：真机主入口（preflight / execute / longrun 等）
- `real_*`、`camera_*`、`robot_*`：ReKep 真机流程模块
- `real_calibration/`：默认相机与外参配置
- `third_party/dobot_xtrainer/`：Dobot 本地 SDK 相关代码（最小必要集）
- `third_party/dobot_xtrainer_remote/`：ZMQ 远端机器人/相机服务相关代码（最小必要集）
- `README_upstream_rekep.md`：上游 ReKep 原始说明（迁移前）

## 2. 在 OEA 框架中的集成点

已集成 `HAL` 驱动：`rekep_real`

- 驱动文件：`hal/drivers/rekep_real_driver.py`
- 档案文件：`hal/profiles/rekep_real.md`
- 注册入口：`hal/drivers/__init__.py`

## 2.1 新机器人适配入口

- 适配器模板：`tools/rekep/cellbot_adapter.py`
- 适配器工厂：`tools/rekep/robot_factory.py`（支持注册新 family）
- 适配文档（英文）：`tools/rekep/docs/robot_adaptation.md`
- 适配文档（中文）：`tools/rekep/docs/robot_adaptation_zh.md`

### 启动方式

```bash
python hal/hal_watchdog.py --driver rekep_real --workspace ~/.nanobot/workspace
```

## 3. 依赖安装（建议）

建议在独立 `conda` 环境安装（例如 `rekep`）：

```bash
pip install -r tools/rekep/requirements.rekep-extra.txt
# 可选：solver 模式所需
pip install -r tools/rekep/requirements.rekep-optional.txt
```

最小真机链路通常还需要：

- `opencv-python-headless`
- `pyzmq`
- `requests`
- `pyrealsense2`（若本机直接连 RealSense）

## 4. 推荐运行模式

使用 solver 模式，显式设置：

```bash
export REKEP_EXECUTION_MODE=solver
```

## 5. 关键环境变量

- `REKEP_TOOL_ROOT`：ReKep 根目录（默认即当前 `tools/rekep`）
- `REKEP_PYTHON`：调用 bridge 的 Python 路径
- `REKEP_REAL_STATE_DIR`：运行状态目录（默认 `/tmp/rekep_real_state`）
- `REKEP_DOBOT_DRIVER`：`xtrainer_zmq` / `xtrainer_sdk` / `dashboard_tcp` / `mock`
- `REKEP_DOBOT_HOST` / `REKEP_DOBOT_PORT` / `REKEP_DOBOT_MOVE_PORT`
- `REKEP_ROBOT_FAMILY` / `REKEP_ROBOT_DRIVER` / `REKEP_ROBOT_HOST` / `REKEP_ROBOT_PORT` / `REKEP_ROBOT_MOVE_PORT`（新机器人通用参数）
- `REKEP_CAMERA_SOURCE`：
  - 直连 RealSense：`realsense` 或 `realsense:<serial>`
  - 远端流：`realsense_zmq://<host>:7001/realsense`
- `REKEP_XTRAINER_SDK_DIR`：本地 SDK 路径（默认 `tools/rekep/third_party/dobot_xtrainer`）

## 6. 直接调用 bridge（调试）

### 6.0 启动远端服务（xtrainer_zmq 常用）

在机器人侧（或能连机器人与相机的主机）可使用迁移后的第三方脚本：

```bash
# 机器人 ZMQ 服务（默认 6001）
python tools/rekep/third_party/dobot_xtrainer_remote/experiments/launch_nodes.py \
  --hostname 0.0.0.0 --robot-port 6001

# RealSense ZMQ 流服务（默认 7001 / topic=realsense）
python tools/rekep/third_party/dobot_xtrainer_remote/experiments/launch_realsense_server.py \
  --host 0.0.0.0 --port 7001 --topic realsense --serial-number <YOUR_SERIAL>
```

### 6.1 预检

```bash
python tools/rekep/dobot_bridge.py preflight --pretty
```

指定新机器人 family 示例：

```bash
python tools/rekep/dobot_bridge.py preflight \
  --robot_family cellbot \
  --robot_driver cellbot_sdk \
  --robot_host 127.0.0.1 \
  --robot_port 9000 \
  --pretty
```

### 6.2 单次执行（默认 dry-run）

```bash
python tools/rekep/dobot_bridge.py execute \
  --instruction "pick up the red block and place it on the tray" \
  --pretty
```

### 6.3 真机执行（显式开运动）

```bash
python tools/rekep/dobot_bridge.py execute \
  --instruction "pick up the red block and place it on the tray" \
  --execute_motion \
  --pretty
```

## 7. 与 ACTION.md 对接（HAL）

`rekep_real` 驱动支持两类 action：

1. 原生真机 action（推荐）：`real_preflight` / `real_execute` / `real_longrun_start` 等
2. 通用操作 action（自动映射到 `execute`）：`move_to` / `pick_up` / `put_down` / `push` / `point_to`

示例（写入 `ACTION.md`）：

```json
{
  "action_type": "real_execute",
  "parameters": {
    "instruction": "pick up the chili pepper and place it in the plate",
    "execute_motion": true
  }
}
```

## 8. 第三方代码说明

本目录仅收录真机链路的最小必要子集，未包含训练模型/数据等大体积资产。

- 来自 Dobot XTrainer 生态代码的目录位于：
  - `third_party/dobot_xtrainer/`
  - `third_party/dobot_xtrainer_remote/`
- 对应许可证与第三方许可清单已随目录保留（`LICENSE` / `THIRD-PARTY-LICENSES`）

如需完整训练/数据采集栈，请在外部自行准备完整上游仓库。

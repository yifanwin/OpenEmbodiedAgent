# ReKep 新机器人适配指南（OpenEmbodiedAgent）

> 目标：接入你自己的机器人后端，同时复用同一条运行链路：
> `预检 -> 场景理解 -> 执行 -> 在线监控 -> 长程任务`。

## 1）统一接口契约

适配器需满足 `tools/rekep/robot_adapter.py`：

1. `connect()`
2. `close()`
3. `get_runtime_state()`
4. `execute_action(action, execute_motion=False)`

动作协议建议至少支持：

1. `movej`
2. `movel`
3. `open_gripper`
4. `close_gripper`
5. `wait`

## 2）新建适配器文件

仓库已内置模板：

- `tools/rekep/cellbot_adapter.py`

你可以复制该文件并替换其中 SDK / RPC 调用逻辑。

## 3）注册适配器家族

`tools/rekep/robot_factory.py` 已支持可扩展注册：

- 内置 family：`dobot`、`cellbot`
- 运行时注册 API：
  - `register_adapter_factory(robot_family, factory, overwrite=False)`
  - `unregister_adapter_factory(robot_family)`
  - `list_adapter_families()`

若你使用自定义 family，请在调用 `create_robot_adapter(...)` 前先注册 factory。

## 4）桥接参数

`tools/rekep/dobot_bridge.py` 现在支持通用参数（且兼容旧 `dobot_*` 参数）：

- `--robot_family`
- `--robot_driver`
- `--robot_host`
- `--robot_port`
- `--robot_move_port`

对应环境变量：

- `REKEP_ROBOT_FAMILY`
- `REKEP_ROBOT_DRIVER`
- `REKEP_ROBOT_HOST`
- `REKEP_ROBOT_PORT`
- `REKEP_ROBOT_MOVE_PORT`

## 5）验证命令

以 `cellbot` 为例：

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
  --instruction "执行一个小幅安全动作进行连通性验证" \
  --pretty
```

```bash
python tools/rekep/dobot_bridge.py execute \
  --robot_family cellbot \
  --robot_driver cellbot_sdk \
  --robot_host 127.0.0.1 \
  --robot_port 9000 \
  --instruction "执行一个小幅安全真机动作" \
  --execute_motion \
  --pretty
```

## 6）相机适配说明

若不是 RealSense，请扩展：

- `tools/rekep/camera_factory.py`
- `tools/rekep/camera_adapter.py`

并保证 `capture_rgbd()` 输出规范：

- `rgb: np.ndarray`
- `depth: np.ndarray`
- `capture_info: dict`

## 7）标定说明

建议复用或镜像 `tools/rekep/real_calibration/` 的目录结构来维护你自己的标定配置。

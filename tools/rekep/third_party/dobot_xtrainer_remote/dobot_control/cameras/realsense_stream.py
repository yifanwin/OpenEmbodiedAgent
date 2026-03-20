import io
import json
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq

DEFAULT_TOPIC = "realsense"


def list_connected_realsense_serials() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for dev in devices:
        serials.append(dev.get_info(rs.camera_info.serial_number))
    return serials


class RealSenseStreamServer:
    def __init__(
        self,
        serial_number: str,
        host: str = "0.0.0.0",
        port: int = 7001,
        topic: str = DEFAULT_TOPIC,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 85,
        flip: bool = False,
        show_ui: bool = False,
    ):
        self._serial_number = serial_number
        self._host = host
        self._port = port
        self._topic = topic.encode("utf-8")
        self._img_size: Optional[Tuple[int, int]] = (width, height)
        self._frame_interval = 0.0 if fps <= 0 else 1.0 / float(fps)
        self._encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(max(1, min(jpeg_quality, 100))),
        ]
        self._show_ui = show_ui

        from dobot_control.cameras.realsense_camera import RealSenseCamera

        self._camera = RealSenseCamera(
            device_id=serial_number,
            flip=flip,
            width=width,
            height=height,
            fps=fps,
        )

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(f"tcp://{host}:{port}")
        self._depth_scale = 0.001

    def serve(self) -> None:
        print(f"RealSense stream server started: tcp://{self._host}:{self._port}")
        print(f"Serial number: {self._serial_number}, topic: {self._topic.decode('utf-8')}")
        print("Press Ctrl+C to stop server.")

        frame_id = 0
        stats_tic = time.time()
        stats_frames = 0
        publish_fps = 0.0

        try:
            while True:
                tic = time.time()
                color_rgb, depth = self._camera.read(img_size=self._img_size)
                color_bgr = color_rgb[:, :, ::-1]
                depth_np = np.asarray(depth)
                if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                    depth_np = depth_np[:, :, 0]
                depth_np = depth_np.astype(np.uint16)

                ok, encoded = cv2.imencode(".jpg", color_bgr, self._encode_param)
                if not ok:
                    continue

                depth_buf = io.BytesIO()
                np.save(depth_buf, depth_np, allow_pickle=False)

                now = time.time()
                header = {
                    "frame_id": frame_id,
                    "timestamp": now,
                    "height": int(color_bgr.shape[0]),
                    "width": int(color_bgr.shape[1]),
                    "serial_number": self._serial_number,
                    "depth_scale": self._depth_scale,
                    "depth_encoding": "npy_uint16",
                    "color_encoding": "jpeg_bgr",
                }
                self._socket.send_multipart(
                    [
                        self._topic,
                        json.dumps(header).encode("utf-8"),
                        encoded.tobytes(),
                        depth_buf.getvalue(),
                    ]
                )

                stats_frames += 1
                dt = now - stats_tic
                if dt >= 1.0:
                    publish_fps = stats_frames / dt
                    stats_tic = now
                    stats_frames = 0

                if self._show_ui:
                    preview = color_bgr.copy()
                    cv2.putText(
                        preview,
                        f"PUB FPS: {publish_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("realsense_server_preview", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_id += 1

                if self._frame_interval > 0.0:
                    elapsed = time.time() - tic
                    if elapsed < self._frame_interval:
                        time.sleep(self._frame_interval - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._camera.close()
        self._socket.close(0)
        if self._show_ui:
            cv2.destroyWindow("realsense_server_preview")


class RealSenseStreamClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7001,
        topic: str = DEFAULT_TOPIC,
        timeout_ms: int = 1000,
    ):
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        # Keep multipart messages intact. ZMQ_CONFLATE can break multipart
        # receive patterns and trigger libzmq assertions.
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))

    def recv_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, object]]]:
        try:
            parts = self._socket.recv_multipart()
        except zmq.Again:
            return None
        if len(parts) < 3:
            return None
        _, header_raw, image_raw, *rest = parts

        header = json.loads(header_raw.decode("utf-8"))
        image_np = np.frombuffer(image_raw, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None

        depth = None
        if rest:
            try:
                depth = np.load(io.BytesIO(rest[0]), allow_pickle=False)
            except Exception:
                depth = None
        return image_bgr, depth, header

    def run_ui(self, window_name: str = "realsense_stream_client") -> None:
        print("Press q or ESC to close client UI.")
        try:
            while True:
                item = self.recv_frame()
                if item is None:
                    continue

                image_bgr, depth, header = item
                now = time.time()
                latency_ms = (now - float(header.get("timestamp", now))) * 1000.0
                cv2.putText(
                    image_bgr,
                    f"Latency: {latency_ms:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image_bgr,
                    f"Frame: {header.get('frame_id', '-')}",
                    (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(window_name, image_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            self.close(window_name=window_name)

    def close(self, window_name: Optional[str] = None) -> None:
        self._socket.close(0)
        if window_name:
            cv2.destroyWindow(window_name)

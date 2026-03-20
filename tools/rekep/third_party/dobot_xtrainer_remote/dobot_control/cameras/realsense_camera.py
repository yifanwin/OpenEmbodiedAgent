import time
from typing import List, Optional, Tuple
import numpy as np
from dobot_control.cameras.camera import CameraDriver
import cv2
import pyrealsense2 as rs

def get_device_ids() -> List[str]:

    ctx = rs.context()
    devices = ctx.query_devices()
    # print(devices)
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        import pyrealsense2 as rs
        print("init", device_id)
        self._device_id = device_id
        self._pipeline = rs.pipeline()
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        config = rs.config()
        config.enable_device(device_id)

        config.enable_stream(
            rs.stream.depth,
            self._width,
            self._height,
            rs.format.z16,
            self._fps,
        )
        config.enable_stream(
            rs.stream.color,
            self._width,
            self._height,
            rs.format.bgr8,
            self._fps,
        )
        try:
            self._pipeline.start(config)
        except RuntimeError as err:
            # Some RealSense models do not support all (resolution, fps) pairs.
            # Fallback to a widely supported profile before failing.
            if self._fps != 30:
                fallback_config = rs.config()
                fallback_config.enable_device(device_id)
                fallback_config.enable_stream(
                    rs.stream.depth, self._width, self._height, rs.format.z16, 30
                )
                fallback_config.enable_stream(
                    rs.stream.color, self._width, self._height, rs.format.bgr8, 30
                )
                self._pipeline.start(fallback_config)
                self._fps = 30
            else:
                raise err
        # Keep depth and color in the same pixel space for downstream keypoint->depth lookup.
        self._align = rs.align(rs.stream.color)
        self._flip = flip
        # print(device_id)
        for _ in range(50):
            self.read()

    def close(self) -> None:
        if getattr(self, "_pipeline", None) is not None:
            self._pipeline.stop()
            self._pipeline = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """

        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        return image, depth

def _debug_read(camera, save_datastream=False):

    cv2.namedWindow("image")
    # cv2.namedWindow("depth")
    counter = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while True:
        # time.sleep(0.1)
        tic = time.time()
        image, depth = camera.read()

        _, image_ = cv2.imencode('.jpg', image, encode_param)

        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        # cv2.imshow("depth", depth)
        toc = time.time()
        print(image_.shape, toc - tic)
        counter += 1


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)

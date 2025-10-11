# to get pyi from https://github.com/IntelRealSense/librealsense/blob/351d8813e4e9a8fdc83df93ff693da50bcdacbd1/wrappers/python/pyrealsense2/__init__.pyi
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from .base import DeviceBase

if TYPE_CHECKING:
    from typing import Any


def get_depth_value(face, depth, device_depth_scale):
    # TODO: device_depth_scale maybe remove?
    sum = 0
    effective_pixel = 0

    for i in range(36, 68):
        x = face.part(i).x
        y = face.part(i).y

        if x > 0 and x < 640 and y > 0 and y < 360:
            data = depth.get_distance(x, y)
            if data > 0:
                sum += data
                effective_pixel += 1

    if effective_pixel != 0:
        return sum / effective_pixel

    return 0


class RealsenseDevice(DeviceBase):
    def __init__(self):
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            logger.info(dev)
            dev.hardware_reset()

        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(
            self.device.get_info(rs.camera_info.product_line)
        )

        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        clipping_distance_in_meters = 1
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale
        self.align = rs.align(rs.stream.color)

    def on_start(self):
        pass

    def on_done(self):
        self.pipeline.stop()

    def read(self) -> tuple[bool, tuple[Any, Any]]:
        frames = self.pipeline.wait_for_frames()  # TODO: first calling is slow
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        aligned_depth_frame = aligned_frames.get_depth_frame()

        images = (color_image, aligned_depth_frame)
        return True, images

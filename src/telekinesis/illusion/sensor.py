"""Camera sensor and ISP effects applied to rendered 8-bit RGB images.

Blender renders an ideal, noise-free image. Real cameras add white-balance
drift, a tone curve, colour gain, lens shading, blur, sensor noise, in-camera
sharpening and compression. ``apply_sensor`` models these steps in camera
order on the display-referred image. Every effect is geometry-preserving, so
masks and boxes stay valid.

This module depends only on NumPy and OpenCV, so it can also be applied
offline to existing datasets.

Example:
    sensor_config = config.SensorConfig(read_noise=(1.0, 3.0))
    rng = np.random.default_rng(0)
    noisy = apply_sensor(image_rgb, sensor_config, rng)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from telekinesis.illusion import config


def sample_sensor_parameters(
    sensor_config: config.SensorConfig, rng: np.random.Generator
) -> dict[str, Any]:
    """Draw one value from every configured range.

    Args:
        sensor_config: Ranges to sample from. Unset effects are omitted.
        rng: Random generator.

    Returns:
        Effect names mapped to sampled values. White balance is a gain per
        RGB channel.
    """
    values: dict[str, Any] = {}
    for field in config.SensorConfig.RANGE_FIELDS:
        interval = getattr(sensor_config, field)
        if interval is not None:
            values[field] = float(rng.uniform(*interval))
    if sensor_config.white_balance is not None:
        values["white_balance"] = rng.uniform(
            *sensor_config.white_balance, size=3
        )
    if "motion_blur" in values:
        # Direction of the camera motion in the image plane; the length is
        # the sampled range field. Half a turn covers every line direction.
        values["motion_blur_angle"] = float(rng.uniform(0.0, 180.0))
    if sensor_config.jpeg_quality is not None:
        low, high = sensor_config.jpeg_quality
        values["jpeg_quality"] = int(rng.integers(low, high + 1))
    return values


def motion_blur_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Normalised line kernel of ``length`` pixels at ``angle_deg``.

    Models a camera that moves during the exposure, as a wrist camera does
    while the arm is in motion: every pixel is smeared along one direction.
    """
    size = int(np.ceil(length)) | 1  # odd, so the line is centred
    kernel = np.zeros((size, size), np.float32)
    centre = size // 2
    half = length / 2.0
    x0, x1 = centre - half, centre + half
    cv2.line(
        kernel,
        (int(round(x0)), centre),
        (int(round(x1)), centre),
        1.0,
        thickness=1,
    )
    rotation = cv2.getRotationMatrix2D((centre, centre), angle_deg, 1.0)
    kernel = cv2.warpAffine(
        kernel, rotation, (size, size), flags=cv2.INTER_LINEAR
    )
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def apply_sensor_parameters(
    image: np.ndarray,
    values: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply sampled sensor effects to one RGB image.

    Args:
        image: RGB or RGBA image, uint8, shape (H, W, 3|4). Alpha is kept.
        values: Output of :func:`sample_sensor_parameters`.
        rng: Random generator for the noise. None uses a fresh generator.

    Returns:
        The processed image with the input's shape and dtype.
    """
    rng = rng or np.random.default_rng()
    rgb = image[..., :3].astype(np.float32) / 255.0
    height, width = rgb.shape[:2]

    if "white_balance" in values:
        gains = np.asarray(values["white_balance"], np.float32)
        rgb *= gains / gains.mean()
    if "gain" in values:
        rgb *= values["gain"]
    if values.get("blur_sigma", 0) > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), values["blur_sigma"])
    if values.get("motion_blur", 0) >= 1.0:
        rgb = cv2.filter2D(
            rgb,
            -1,
            motion_blur_kernel(
                values["motion_blur"], values.get("motion_blur_angle", 0.0)
            ),
            borderType=cv2.BORDER_REPLICATE,
        )
    if "contrast" in values:
        # Contrast around mid-grey through a smooth S/anti-S curve.
        rgb = 0.5 + (rgb - 0.5) * values["contrast"]
    rgb = np.clip(rgb, 0.0, 1.0)
    if "gamma" in values:
        rgb = np.power(rgb, 1.0 / values["gamma"])
    if "saturation" in values:
        luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        rgb = luma[..., None] + (rgb - luma[..., None]) * values["saturation"]
    if values.get("vignetting", 0) > 0:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        radius2 = ((xx - width / 2) / (width / 2)) ** 2 + (
            (yy - height / 2) / (height / 2)
        ) ** 2
        rgb *= (1.0 - values["vignetting"] * radius2 / 2.0)[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)

    shot, read = values.get("shot_noise", 0.0), values.get("read_noise", 0.0)
    if shot > 0 or read > 0:
        # Noise in 8-bit units: variance = shot * intensity + read^2.
        sigma = np.sqrt(shot * rgb * 255.0 + read**2) / 255.0
        noise = rng.standard_normal(rgb.shape, dtype=np.float32)
        chroma = values.get("chroma_noise", 1.0)
        if chroma < 1.0:
            luma_noise = noise.mean(axis=2, keepdims=True)
            noise = luma_noise + (noise - luma_noise) * chroma
        rgb = rgb + sigma * noise
    if values.get("sharpen_amount", 0) > 0:
        sigma = values.get("sharpen_sigma", 1.0)
        blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
        rgb = rgb + values["sharpen_amount"] * (rgb - blurred)

    result = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    if "jpeg_quality" in values:
        ok, encoded = cv2.imencode(
            ".jpg",
            result[..., ::-1],
            [int(cv2.IMWRITE_JPEG_QUALITY), values["jpeg_quality"]],
        )
        if ok:
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
    if image.shape[-1] == 4:
        return np.concatenate([result, image[..., 3:]], axis=-1)
    return result


def apply_sensor(
    image: np.ndarray,
    sensor_config: config.SensorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample sensor parameters and apply them to one RGB image.

    Args:
        image: RGB or RGBA uint8 image.
        sensor_config: Effect ranges. ``probability`` decides whether the
            image is processed at all.
        rng: Random generator.

    Returns:
        The processed image, or the input when the image is skipped.
    """
    if rng.random() >= sensor_config.probability:
        return image
    return apply_sensor_parameters(
        image, sample_sensor_parameters(sensor_config, rng), rng
    )


def make_color_transform(
    sensor_config: config.SensorConfig | None, seed: int | None = None
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a per-image transform for writers, or None when disabled.

    Args:
        sensor_config: Effect ranges, or None for no sensor simulation.
        seed: Seed of the transform's random generator.

    Returns:
        A callable mapping an RGB image to its processed copy, or None.
    """
    if sensor_config is None:
        return None
    return SensorTransform(sensor_config, seed)


class SensorTransform:
    """Callable sensor simulation that records each image's sampled values.

    ``last`` holds the parameters of the most recent image (empty when the
    image was skipped), so writers can store per-image provenance.
    """

    def __init__(self, sensor_config: config.SensorConfig, seed: int | None = None):
        self.sensor_config = sensor_config
        self.rng = np.random.default_rng(seed)
        self.last: dict[str, Any] = {}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.sensor_config.probability:
            self.last = {}
            return image
        values = sample_sensor_parameters(self.sensor_config, self.rng)
        self.last = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in values.items()
        }
        return apply_sensor_parameters(image, values, self.rng)

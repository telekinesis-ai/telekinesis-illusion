"""Camera sensor and ISP effects applied to rendered 8-bit RGB images.

Blender renders an ideal, noise-free image. Real cameras add white-balance
drift, a tone curve, colour gain, lens shading, blur, sensor noise, in-camera
sharpening and compression. ``apply_sensor`` models these steps in camera
order, either on the display-referred image (``mode="display"``) or through a
raw pipeline: linear light, Bayer mosaic, photosite noise, demosaic, in-camera
denoise, tone curve (``mode="raw"``). Every effect is geometry-preserving, so
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


def encode_linear(lin: np.ndarray, values: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Linear radiance -> display-referred [0, 1]: exposure, white balance, gain, sensor clipping, sRGB curve.

    Returns the encoded image and ``values`` without the stages already applied (white balance, gain), so the
    display pipeline does not apply them twice.
    """
    lin = lin * np.float32(2.0 ** values.get("exposure_ev", 0.0))
    if "white_balance" in values:
        gains = np.asarray(values["white_balance"], np.float32)
        lin = lin * (gains / gains.mean())
    if "gain" in values:
        lin = lin * values["gain"]
    rest = {k: v for k, v in values.items() if k not in ("white_balance", "gain")}
    return linear_to_srgb(np.clip(lin, 0.0, 1.0)).astype(np.float32), rest


def noise_field(shape: tuple, values: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Unit-variance noise; with ``noise_blur`` / ``noise_blur_mix`` a share of its variance is spatially smooth.

    Camera processing (denoising, resampling, chroma subsampling) leaves real sensor noise in blotches a few
    pixels wide rather than independent per pixel; the RealSense D405 measured on a static scene correlates
    ~0.47 / 0.23 / 0.17 at 1 / 2 / 3 px (after removing frame-to-frame flicker). The smooth part is white noise
    blurred with sigma ``noise_blur`` and renormalised, mixed in at variance fraction ``noise_blur_mix``.
    """
    noise = rng.standard_normal(shape, dtype=np.float32)
    sigma, mix = values.get("noise_blur", 0.0), values.get("noise_blur_mix", 0.0)
    if sigma > 0 and mix > 0:
        smooth = cv2.GaussianBlur(rng.standard_normal(shape, dtype=np.float32), (0, 0), sigma)
        smooth /= max(float(smooth.std()), 1e-6)
        noise = np.sqrt(1.0 - mix) * noise + np.sqrt(mix) * smooth
    return noise


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# Channel (R=0, G=1, B=2) of each 2x2 Bayer cell position, and OpenCV's matching demosaic codes. OpenCV names a
# pattern by the second row's first two pixels, so an RGGB sensor is its "BayerBG".
_BAYER = {"RGGB": ((0, 1), (1, 2)), "BGGR": ((2, 1), (1, 0)), "GRBG": ((1, 0), (2, 1)), "GBRG": ((1, 2), (0, 1))}
_DEMOSAIC = {
    ("RGGB", "bilinear"): cv2.COLOR_BayerBG2RGB, ("RGGB", "vng"): cv2.COLOR_BayerBG2RGB_VNG,
    ("RGGB", "ea"): cv2.COLOR_BayerBG2RGB_EA,
    ("BGGR", "bilinear"): cv2.COLOR_BayerRG2RGB, ("BGGR", "vng"): cv2.COLOR_BayerRG2RGB_VNG,
    ("BGGR", "ea"): cv2.COLOR_BayerRG2RGB_EA,
    ("GRBG", "bilinear"): cv2.COLOR_BayerGB2RGB, ("GRBG", "vng"): cv2.COLOR_BayerGB2RGB_VNG,
    ("GRBG", "ea"): cv2.COLOR_BayerGB2RGB_EA,
    ("GBRG", "bilinear"): cv2.COLOR_BayerGR2RGB, ("GBRG", "vng"): cv2.COLOR_BayerGR2RGB_VNG,
    ("GBRG", "ea"): cv2.COLOR_BayerGR2RGB_EA,
}


def mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """One colour sample per pixel, as behind a Bayer colour-filter array: (H, W) from (H, W, 3)."""
    out = np.empty(rgb.shape[:2], rgb.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            out[dy::2, dx::2] = rgb[dy::2, dx::2, _BAYER[pattern][dy][dx]]
    return out


def demosaic(raw: np.ndarray, pattern: str, method: str) -> np.ndarray:
    """Reconstruct RGB from a Bayer mosaic in [0, 1] (16-bit through OpenCV; VNG needs 8-bit)."""
    code = _DEMOSAIC[pattern, method]
    if method == "vng":
        return cv2.cvtColor(np.clip(np.rint(raw * 255), 0, 255).astype(np.uint8), code).astype(np.float32) / 255.0
    return cv2.cvtColor(np.clip(np.rint(raw * 65535), 0, 65535).astype(np.uint16), code).astype(np.float32) / 65535.0


def _apply_raw(
    rgb: np.ndarray, values: dict[str, Any], rng: np.random.Generator, pattern: str, method: str
) -> np.ndarray:
    """The ``"raw"`` sensor mode on a display-referred RGB image in [0, 1]; see SensorConfig.mode."""
    height, width = rgb.shape[:2]
    lin = srgb_to_linear(rgb)
    if values.get("blur_sigma", 0) > 0:
        lin = cv2.GaussianBlur(lin, (0, 0), values["blur_sigma"])
    if values.get("motion_blur", 0) >= 1.0:
        lin = cv2.filter2D(
            lin, -1, motion_blur_kernel(values["motion_blur"], values.get("motion_blur_angle", 0.0)),
            borderType=cv2.BORDER_REPLICATE,
        )
    if "gain" in values:
        lin = lin * values["gain"]
    raw = np.clip(mosaic(lin, pattern), 0.0, 1.0)
    shot, read = values.get("shot_noise", 0.0), values.get("read_noise", 0.0)
    if shot > 0 or read > 0:
        # Linear 8-bit units: variance = shot * signal + read^2, per photosite.
        sigma = np.sqrt(shot * raw * 255.0 + read**2) / 255.0
        raw = np.clip(raw + sigma * noise_field(raw.shape, values, rng), 0.0, 1.0)
    lin = demosaic(raw, pattern, method)
    if values.get("isp_denoise", 0) > 0:
        h = float(values["isp_denoise"])
        enc = np.clip(np.rint(linear_to_srgb(lin) * 255), 0, 255).astype(np.uint8)
        enc = cv2.fastNlMeansDenoisingColored(enc, None, h, h, 5, 15)
        lin = srgb_to_linear(enc.astype(np.float32) / 255.0)
    if "white_balance" in values:
        gains = np.asarray(values["white_balance"], np.float32)
        lin = lin * (gains / gains.mean())
    out = linear_to_srgb(lin)
    if "contrast" in values:
        out = 0.5 + (out - 0.5) * values["contrast"]
    out = np.clip(out, 0.0, 1.0)
    if "gamma" in values:
        out = np.power(out, 1.0 / values["gamma"])
    if "saturation" in values:
        luma = out @ np.array([0.299, 0.587, 0.114], np.float32)
        out = luma[..., None] + (out - luma[..., None]) * values["saturation"]
    if values.get("vignetting", 0) > 0:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        radius2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
        out = out * (1.0 - values["vignetting"] * radius2 / 2.0)[..., None]
    if values.get("sharpen_amount", 0) > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), values.get("sharpen_sigma", 1.0))
        out = out + values["sharpen_amount"] * (out - blurred)
    return out


def apply_sensor_parameters(
    image: np.ndarray,
    values: dict[str, Any],
    rng: np.random.Generator | None = None,
    mode: str = "display",
    bayer_pattern: str = "RGGB",
    demosaic_method: str = "ea",
) -> np.ndarray:
    """Apply sampled sensor effects to one RGB image.

    Args:
        image: RGB or RGBA image, uint8, shape (H, W, 3|4). Alpha is kept.
        values: Output of :func:`sample_sensor_parameters`.
        rng: Random generator for the noise. None uses a fresh generator.
        mode: ``"display"`` or ``"raw"``, see :class:`config.SensorConfig`.
        bayer_pattern: Colour-filter layout for ``"raw"``.
        demosaic_method: Demosaicing for ``"raw"``.

    Returns:
        The processed image with the input's shape and dtype.
    """
    rng = rng or np.random.default_rng()
    if np.issubdtype(image.dtype, np.floating):
        # Scene-linear radiance (RenderConfig.color_output="linear"): exposure, white balance and gain act on
        # light, the sensor clips at saturation, then the camera's tone curve; the remaining stages run on
        # the encoded image without an 8-bit round trip, so shadows keep their precision.
        rgb, values = encode_linear(image[..., :3].astype(np.float32), values)
        alpha = None if image.shape[-1] != 4 else np.clip(np.rint(image[..., 3:] * 255.0), 0, 255).astype(np.uint8)
        image = np.zeros(image.shape[:2] + (3,), np.uint8) if alpha is None else np.concatenate(
            [np.zeros(image.shape[:2] + (3,), np.uint8), alpha], axis=-1)
    else:
        rgb = image[..., :3].astype(np.float32) / 255.0
    if mode == "raw":
        out = _apply_raw(rgb, values, rng, bayer_pattern, demosaic_method)
        return _finish(image, out, values)
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
        noise = noise_field(rgb.shape, values, rng)
        chroma = values.get("chroma_noise", 1.0)
        if chroma < 1.0:
            luma_noise = noise.mean(axis=2, keepdims=True)
            noise = luma_noise + (noise - luma_noise) * chroma
        rgb = rgb + sigma * noise
    if values.get("sharpen_amount", 0) > 0:
        sigma = values.get("sharpen_sigma", 1.0)
        blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
        rgb = rgb + values["sharpen_amount"] * (rgb - blurred)
    return _finish(image, rgb, values)


def _finish(image: np.ndarray, rgb: np.ndarray, values: dict[str, Any]) -> np.ndarray:
    """Quantise to 8 bits, apply the in-camera JPEG if sampled, and restore alpha."""
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
        image, sample_sensor_parameters(sensor_config, rng), rng,
        sensor_config.mode, sensor_config.bayer_pattern, sensor_config.demosaic,
    )


def make_color_transform(
    sensor_config: config.SensorConfig | None, seed: int | None = None, linear: bool = False
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a per-image transform for writers, or None when disabled.

    Args:
        sensor_config: Effect ranges, or None for no sensor simulation.
        seed: Seed of the transform's random generator.
        linear: The renders are scene-linear float images; without a sensor
            they are still clipped and sRGB-encoded to 8 bits.

    Returns:
        A callable mapping an RGB image to its processed copy, or None.
    """
    if sensor_config is None:
        return encode_srgb8 if linear else None
    return SensorTransform(sensor_config, seed)


def encode_srgb8(image: np.ndarray) -> np.ndarray:
    """Linear float RGB(A) -> 8-bit sRGB, with no other effect."""
    return apply_sensor_parameters(image, {})


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
            return encode_srgb8(image) if np.issubdtype(image.dtype, np.floating) else image
        values = sample_sensor_parameters(self.sensor_config, self.rng)
        self.last = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in values.items()
        }
        cfg = self.sensor_config
        return apply_sensor_parameters(image, values, self.rng, cfg.mode, cfg.bayer_pattern, cfg.demosaic)

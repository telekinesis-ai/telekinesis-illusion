"""Write rendered frames as a BOP scene (https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).

Each output directory (shard) becomes one BOP scene under ``<output_dir>/bop/``:

* ``rgb/<im>.jpg|png``: the colour image the COCO writer wrote (hard link, so the sensor
  simulation is applied once and both formats show the same pixels)
* ``depth/<im>.png``: uint16 depth, ``depth_mm = value * depth_scale`` (if depth was rendered)
* ``mask_visib/<im>_<k>.png``: visible mask of the k-th object of ``scene_gt[im]``
* ``scene_gt.json``: ``cam_R_m2c`` (row-major 3x3), ``cam_t_m2c`` (mm), ``obj_id``
* ``scene_camera.json``: ``cam_K``, ``depth_scale``, ``cam_R_w2c``, ``cam_t_w2c`` (mm)
* ``scene_gt_info.json``: ``bbox_visib`` and ``px_count_visib``. The amodal fields
  (``bbox_obj``, ``px_count_all``, ``visib_fract``) need the object models and are
  added offline by bop_toolkit's ``calc_gt_info.py`` (and ``calc_gt_masks.py`` for ``mask/``).

Poses are model-to-camera in the OpenCV camera frame (x right, y down, z forward).
``obj_id`` is the model's category id. ``cam_t_m2c`` is in mm, i.e. it is the pose of the
model file expressed in millimetres; for BOP models (mm files loaded at scale 0.001) it is
the standard BOP pose. Every active categorised model is listed, also fully occluded ones
(empty visible mask, ``px_count_visib`` 0, ``bbox_visib`` [-1, -1, -1, -1]).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from telekinesis.illusion.types.render_result import RenderResult

# Blender camera looks down -z with y up; OpenCV looks down +z with y down.
_BLENDER_TO_OPENCV_CAM = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_SCALE_MM = 0.1  # stored depth value = depth_mm / 0.1  (0.1 mm steps, max 6.55 m)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _dump(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rotation(m: np.ndarray) -> np.ndarray:
    """Closest rotation to a 3x3 block: drops any residual scale/shear (like
    BlenderProc's ``to_quaternion().to_matrix()``)."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def model_to_camera(camera_matrix_world: Any, matrix_world: Any) -> np.ndarray:
    """4x4 model-to-camera transform (metres) in the OpenCV camera frame.

    Same convention as BlenderProc's BopWriterUtility: the Blender camera frame
    is flipped to OpenCV with ["X", "-Y", "-Z"] and
    ``cam_H_m2c = inv(cam2world_opencv) @ local2world``.
    """
    cam2world = np.asarray(camera_matrix_world, float) @ _BLENDER_TO_OPENCV_CAM
    t = np.linalg.inv(cam2world) @ np.asarray(matrix_world, float)
    t[:3, :3] = _rotation(t[:3, :3])
    return t


def write_bop_frames(
    output_dir: str | Path,
    result: RenderResult,
    color_paths: list[str | Path],
    image_ids: list[int],
) -> None:
    """Append ``result``'s views to the BOP scene in ``<output_dir>/bop``.

    ``color_paths[i]`` is the already written colour file of view i and ``image_ids[i]``
    its BOP image id.
    """
    if not result.objects:
        raise ValueError(
            "BOP writing needs object poses in the render result (RenderResult.objects)."
        )
    root = Path(output_dir) / "bop"
    for sub in ("rgb", "depth", "mask_visib"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    scene_gt = _load(root / "scene_gt.json")
    scene_camera = _load(root / "scene_camera.json")
    scene_gt_info = _load(root / "scene_gt_info.json")
    data = result.data
    depths = data.get("depth")

    for view, (capture, color_path, im) in enumerate(zip(result.captures, color_paths, image_ids)):
        color_path = Path(color_path)
        _link_or_copy(color_path, root / "rgb" / f"{im:06d}{color_path.suffix}")

        cam_world = np.asarray(capture["camera_matrix_world"], float) @ _BLENDER_TO_OPENCV_CAM
        world2cam = np.linalg.inv(cam_world)
        world2cam[:3, :3] = _rotation(world2cam[:3, :3])
        camera_entry = {
            "cam_K": np.asarray(capture["camera_K"], float).reshape(-1).tolist(),
            "cam_R_w2c": world2cam[:3, :3].reshape(-1).tolist(),
            "cam_t_w2c": (world2cam[:3, 3] * 1000.0).tolist(),
        }
        if depths is not None:
            z = np.asarray(depths[view], np.float32)
            z[~np.isfinite(z)] = 0
            raw = np.round(z * 1000.0 / DEPTH_SCALE_MM)
            cv2.imwrite(str(root / "depth" / f"{im:06d}.png"), raw.clip(0, 65535).astype(np.uint16))
            camera_entry["depth_scale"] = DEPTH_SCALE_MM
        scene_camera[str(im)] = camera_entry

        segmap = np.asarray(data["instance_segmaps"][view])
        idx_by_name = {
            a.get("name"): int(a["idx"]) for a in data["instance_attribute_maps"][view] if "name" in a
        }
        gts, infos = [], []
        for k, obj in enumerate(result.objects):
            T = model_to_camera(capture["camera_matrix_world"], obj["matrix_world"])
            gts.append(
                {
                    "cam_R_m2c": T[:3, :3].reshape(-1).tolist(),
                    "cam_t_m2c": (T[:3, 3] * 1000.0).tolist(),
                    "obj_id": int(obj["category_id"]),
                }
            )
            idx = idx_by_name.get(obj["name"])
            mask = (segmap == idx) if idx is not None else np.zeros(segmap.shape, bool)
            cv2.imwrite(
                str(root / "mask_visib" / f"{im:06d}_{k:06d}.png"), mask.astype(np.uint8) * 255
            )
            n = int(mask.sum())
            if n:
                ys, xs = np.nonzero(mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            else:
                bbox = [-1, -1, -1, -1]
            infos.append({"bbox_visib": bbox, "px_count_visib": n})
        scene_gt[str(im)] = gts
        scene_gt_info[str(im)] = infos

    _dump(root / "scene_gt.json", scene_gt)
    _dump(root / "scene_camera.json", scene_camera)
    _dump(root / "scene_gt_info.json", scene_gt_info)

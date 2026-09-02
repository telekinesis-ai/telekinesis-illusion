"""
Split a merged COCO dataset into train/valid/test directories.

Produces either RF-DETR-compatible COCO output or YOLO-format output, written
to a user-specified destination directory. The split is stratified by each
image's dominant annotation category.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm

import cv2
import numpy as np
import yaml
from loguru import logger
from pycocotools import mask as mask_utils
from sklearn.model_selection import train_test_split

from telekinesis.illusion.dataset.dataset_generator import (
    MERGED_ANNOTATION_FILENAME,
)

SPLIT_NAMES = ("train", "valid", "test")
SUPPORTED_FORMATS = ("coco", "yolo")
COCO_ANNOTATION_FILENAME = "_annotations.coco.json"
NO_ANNOTATION_STRATUM = -1
# Minimum vertices a YOLO segmentation polygon must have (3 points = 6 numbers).
MIN_YOLO_SEG_POLYGON_POINTS = 3


def _copy_image(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


class DatasetConverter:
    """Convert a merged COCO dataset into a single train/valid/test layout.

    Reads ``<merged_dataset_dir>/<merged_filename>`` (default
    ``merged_coco_annotations.json``) and writes the split output to
    ``output_dir`` in the requested ``format`` ("coco" or "yolo").
    """

    def __init__(
        self,
        merged_dataset_dir: str | Path,
        output_dir: str | Path,
        dataset_format: str,
        ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
        stratify: bool = True,
        seed: int = 42,
        merged_filename: str = MERGED_ANNOTATION_FILENAME,
    ) -> None:
        self._dataset_dir = Path(merged_dataset_dir)
        if not self._dataset_dir.is_dir():
            raise NotADirectoryError(
                f"Merged dataset directory does not exist: {self._dataset_dir}"
            )

        self._merged_path = self._dataset_dir / merged_filename
        if not self._merged_path.is_file():
            raise FileNotFoundError(
                f"Merged annotation file not found: {self._merged_path}"
            )

        format_lower = dataset_format.lower()
        if format_lower not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported dataset_format '{dataset_format}'. Choose one of: {SUPPORTED_FORMATS}"
            )
        self._format = format_lower

        self._output_dir = Path(output_dir)

        if (
            len(ratios) != 3
            or abs(sum(ratios) - 1.0) > 1e-6
            or any(r < 0 for r in ratios)
        ):
            raise ValueError(
                f"ratios must be three non-negative values summing to 1.0, got {ratios}"
            )
        self._ratios = ratios
        self._stratify = stratify
        self._seed = seed

    def split(self) -> Path:
        """Run the split. Returns the output directory."""
        with open(self._merged_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        if not images:
            raise ValueError("Merged dataset contains no images")

        anns_by_image = self._group_annotations_by_image(annotations)
        assignments = self._compute_split_assignments(images, anns_by_image)

        self._log_split_sizes(assignments)

        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self._format == "coco":
            self._write_coco(self._output_dir, coco, anns_by_image, assignments)
        else:
            self._write_yolo(
                self._output_dir,
                images,
                anns_by_image,
                categories,
                assignments,
            )

        return self._output_dir

    # ------------------------------------------------------------------
    # Split computation
    # ------------------------------------------------------------------

    @staticmethod
    def _group_annotations_by_image(
        annotations: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for ann in annotations:
            grouped.setdefault(ann["image_id"], []).append(ann)
        return grouped

    def _compute_split_assignments(
        self,
        images: list[dict[str, Any]],
        anns_by_image: dict[int, list[dict[str, Any]]],
    ) -> dict[int, str]:
        img_ids = [img["id"] for img in images]
        strata_by_id = {
            img_id: self._dominant_category(anns_by_image.get(img_id, []))
            for img_id in img_ids
        }

        test_frac = self._ratios[2]
        train_frac = self._ratios[0]
        val_frac = self._ratios[1]

        # Step 1: peel off test set.
        full_strata = (
            [strata_by_id[i] for i in img_ids] if self._stratify else None
        )
        try:
            trainval_ids, test_ids = train_test_split(
                img_ids,
                test_size=test_frac,
                random_state=self._seed,
                stratify=full_strata,
            )
        except ValueError as e:
            logger.warning(
                "Stratified test split failed ({}). Falling back to random split.",
                e,
            )
            trainval_ids, test_ids = train_test_split(
                img_ids, test_size=test_frac, random_state=self._seed
            )

        # Step 2: split remaining into train and valid.
        rest_total = train_frac + val_frac
        val_frac_of_rest = val_frac / rest_total if rest_total > 0 else 0.0

        trainval_strata = (
            [strata_by_id[i] for i in trainval_ids] if self._stratify else None
        )
        try:
            train_ids, valid_ids = train_test_split(
                trainval_ids,
                test_size=val_frac_of_rest,
                random_state=self._seed,
                stratify=trainval_strata,
            )
        except ValueError as e:
            logger.warning(
                "Stratified train/valid split failed ({}). Falling back to random split.",
                e,
            )
            train_ids, valid_ids = train_test_split(
                trainval_ids,
                test_size=val_frac_of_rest,
                random_state=self._seed,
            )

        assignments: dict[int, str] = {}
        for iid in train_ids:
            assignments[iid] = "train"
        for iid in valid_ids:
            assignments[iid] = "valid"
        for iid in test_ids:
            assignments[iid] = "test"
        return assignments

    @staticmethod
    def _dominant_category(annotations: list[dict[str, Any]]) -> int:
        if not annotations:
            return NO_ANNOTATION_STRATUM
        counts = Counter(ann["category_id"] for ann in annotations)
        return counts.most_common(1)[0][0]

    def _log_split_sizes(self, assignments: dict[int, str]) -> None:
        counts = Counter(assignments.values())
        total = sum(counts.values())
        logger.info(
            "Split sizes: train={} ({:.1%}), valid={} ({:.1%}), test={} ({:.1%})",
            counts.get("train", 0),
            counts.get("train", 0) / total,
            counts.get("valid", 0),
            counts.get("valid", 0) / total,
            counts.get("test", 0),
            counts.get("test", 0) / total,
        )

    # ------------------------------------------------------------------
    # COCO writer
    # ------------------------------------------------------------------

    def _write_coco(
        self,
        coco_root: Path,
        merged: dict[str, Any],
        anns_by_image: dict[int, list[dict[str, Any]]],
        assignments: dict[int, str],
    ) -> Path:
        coco_root.mkdir(parents=True, exist_ok=True)
        images_by_id = {img["id"]: img for img in merged["images"]}

        for split_name in SPLIT_NAMES:
            split_dir = coco_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            split_images: list[dict[str, Any]] = []
            split_annotations: list[dict[str, Any]] = []

            for img_id, assigned in tqdm(assignments.items()):
                if assigned != split_name:
                    continue

                img = images_by_id[img_id]
                src = self._dataset_dir / img["file_name"]
                ext = Path(img["file_name"]).suffix
                new_basename = f"{img_id:06d}{ext}"
                dst = split_dir / new_basename
                _copy_image(src, dst)

                new_img = deepcopy(img)
                new_img["file_name"] = new_basename
                split_images.append(new_img)

                for ann in anns_by_image.get(img_id, []):
                    split_annotations.append(deepcopy(ann))

            split_coco = {
                "info": merged.get("info", {}),
                "licenses": merged.get("licenses", []),
                "categories": merged["categories"],
                "images": split_images,
                "annotations": split_annotations,
            }
            with open(
                split_dir / COCO_ANNOTATION_FILENAME, "w", encoding="utf-8"
            ) as f:
                json.dump(split_coco, f)

            logger.info(
                "Wrote COCO split '{}': {} images, {} annotations -> {}",
                split_name,
                len(split_images),
                len(split_annotations),
                split_dir,
            )

        return coco_root

    # ------------------------------------------------------------------
    # YOLO writer
    # ------------------------------------------------------------------

    def _write_yolo(
        self,
        yolo_root: Path,
        images: list[dict[str, Any]],
        anns_by_image: dict[int, list[dict[str, Any]]],
        categories: list[dict[str, Any]],
        assignments: dict[int, str],
    ) -> Path:
        yolo_root.mkdir(parents=True, exist_ok=True)
        sorted_cats = sorted(categories, key=lambda c: c["id"])
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted_cats)}
        class_names = [c["name"] for c in sorted_cats]
        images_by_id = {img["id"]: img for img in images}

        # YOLO convention uses "val" instead of "valid".
        _yolo_dir = {"valid": "val"}

        for split_name in SPLIT_NAMES:
            yolo_name = _yolo_dir.get(split_name, split_name)
            (yolo_root / "images" / yolo_name).mkdir(
                parents=True, exist_ok=True
            )
            (yolo_root / "labels" / yolo_name).mkdir(
                parents=True, exist_ok=True
            )

        per_split_counts: dict[str, int] = {s: 0 for s in SPLIT_NAMES}

        for img_id, split_name in tqdm(assignments.items()):
            img = images_by_id[img_id]
            src = self._dataset_dir / img["file_name"]
            ext = Path(img["file_name"]).suffix
            new_basename = f"{img_id:06d}{ext}"

            yolo_name = _yolo_dir.get(split_name, split_name)
            image_dst = yolo_root / "images" / yolo_name / new_basename
            label_dst = yolo_root / "labels" / yolo_name / f"{img_id:06d}.txt"

            _copy_image(src, image_dst)
            self._write_yolo_seg_label(
                label_dst,
                anns_by_image.get(img_id, []),
                img_width=img["width"],
                img_height=img["height"],
                cat_id_to_idx=cat_id_to_idx,
            )
            per_split_counts[split_name] += 1

        data_yaml = {
            "path": str(yolo_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": class_names,
        }
        with open(yolo_root / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)

        for split_name, count in per_split_counts.items():
            logger.info("Wrote YOLO split '{}': {} images", split_name, count)

        return yolo_root

    @staticmethod
    def _write_yolo_seg_label(
        label_path: Path,
        annotations: list[dict[str, Any]],
        img_width: int,
        img_height: int,
        cat_id_to_idx: dict[int, int],
    ) -> None:
        """Write Ultralytics YOLO-seg labels: `class x1 y1 x2 y2 ... xn yn`.

        Coordinates are normalized to [0, 1]. One polygon per instance; for
        multi-region annotations the largest contour is kept.
        """
        lines: list[str] = []
        for ann in annotations:
            polygon = DatasetConverter._annotation_to_polygon(
                ann, img_height, img_width
            )
            if polygon is None or len(polygon) < MIN_YOLO_SEG_POLYGON_POINTS:
                continue
            class_idx = cat_id_to_idx[ann["category_id"]]
            coords: list[str] = []
            for x, y in polygon:
                coords.append(f"{x / img_width:.6f}")
                coords.append(f"{y / img_height:.6f}")
            lines.append(f"{class_idx} {' '.join(coords)}")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _annotation_to_polygon(
        ann: dict[str, Any], img_height: int, img_width: int
    ) -> list[tuple[float, float]] | None:
        """Return the largest polygon for an annotation, or None if unusable."""
        seg = ann.get("segmentation")
        if not seg:
            return None

        if isinstance(seg, list):
            polys = [
                p for p in seg if len(p) >= 2 * MIN_YOLO_SEG_POLYGON_POINTS
            ]
            if not polys:
                return None
            poly = max(polys, key=len)
            return list(zip(poly[0::2], poly[1::2]))

        if isinstance(seg, dict):
            rle: Any = seg
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(
                    cast(Any, rle), img_height, img_width
                )
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.001 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            return [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]

        return None

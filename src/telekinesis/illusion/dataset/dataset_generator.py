"""
Code for generating full datasets from generated shards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


ANNOTATION_FILENAME = "coco_annotations.json"
MERGED_ANNOTATION_FILENAME = "merged_coco_annotations.json"


class CocoShardMerger:
    """Merges multiple COCO annotation shards into a single coherent dataset.

    Each shard is a sub-folder inside *shards_directory* that contains a
    ``coco_annotations.json`` and an ``images/`` directory.  The merger
    re-indexes image and annotation IDs so they are globally unique and
    rewrites ``file_name`` paths to be relative to *shards_directory*.

    Shards may contain different numbers of images and annotations — the
    merger keeps running ID counters across shards, so per-shard size is
    independent. Categories are also unioned across shards (see
    :meth:`merge` and :meth:`_merge_categories`), so shards do not need to
    declare the same category set either.
    """

    def __init__(self, shards_directory: str | Path) -> None:
        self._shards_dir = Path(shards_directory)
        if not self._shards_dir.is_dir():
            raise NotADirectoryError(
                f"Shards directory does not exist: {self._shards_dir}"
            )

    def merge(self) -> Path:
        """Merge all shards and write the combined annotation file.

        Categories from each shard are unioned into a single deterministic list
        (sorted by id). Mismatched id↔name pairings between shards raise
        ``ValueError``. See :meth:`_merge_categories` for the exact rule.

        Returns the path to the written ``merged_coco_annotations.json``.
        """
        shard_dirs = self._discover_shards()
        if not shard_dirs:
            raise FileNotFoundError(
                f"No shards found in {self._shards_dir} "
                f"(looked for sub-folders containing {ANNOTATION_FILENAME})"
            )

        logger.info("Found {} shard(s) to merge.", len(shard_dirs))

        merged_images: list[dict[str, Any]] = []
        merged_annotations: list[dict[str, Any]] = []
        reference_categories: list[dict[str, Any]] = []
        id_to_name: dict[int, str] = {}
        name_to_id: dict[str, int] = {}
        info: dict[str, Any] = {}
        licenses: list[dict[str, Any]] = []
        first_shard = True

        img_offset = 0
        ann_offset = 0

        for shard_dir in shard_dirs:
            shard = self._load_shard(shard_dir)

            if first_shard:
                info = shard.get("info", {})
                licenses = shard.get("licenses", [])
                first_shard = False

            self._merge_categories(
                shard["categories"],
                reference_categories,
                id_to_name,
                name_to_id,
                shard_dir.name,
            )

            images, annotations, img_offset, ann_offset = self._remap_shard(
                shard, shard_dir, img_offset, ann_offset
            )

            merged_images.extend(images)
            merged_annotations.extend(annotations)

            logger.info(
                "Merged shard '{}': {} images, {} annotations.",
                shard_dir.name,
                len(images),
                len(annotations),
            )

        reference_categories.sort(key=lambda c: c["id"])

        merged = {
            "info": info,
            "licenses": licenses,
            "categories": reference_categories,
            "images": merged_images,
            "annotations": merged_annotations,
        }

        output_path = self._shards_dir / MERGED_ANNOTATION_FILENAME
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f)

        logger.info(
            "Wrote merged annotations to {} ({} images, {} annotations).",
            output_path,
            len(merged_images),
            len(merged_annotations),
        )
        return output_path

    def preview(self) -> None:
        """Open the merged dataset in the COCO viewer.

        Must be called after :meth:`merge` so that the merged annotation file
        exists.
        """
        # Deferred: shard_viewer needs tkinter, which isn't available in
        # every Python environment this module gets imported into (e.g.
        # Blender's bundled Python) - only import it if preview() is
        # actually called.
        from telekinesis.illusion.viewer.shard_viewer import view_coco

        annotation_path = self._shards_dir / MERGED_ANNOTATION_FILENAME
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Merged annotation file not found: {annotation_path}. "
                "Call merge() first."
            )

        view_coco(
            self._shards_dir,
            images_subdir=None,
            annotations_filename=MERGED_ANNOTATION_FILENAME,
            title="Merged Dataset Preview",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_shards(self) -> list[Path]:
        """Return sorted list of sub-directories that contain a COCO file."""
        return sorted(
            d
            for d in self._shards_dir.iterdir()
            if d.is_dir() and (d / ANNOTATION_FILENAME).exists()
        )

    def _load_shard(self, shard_dir: Path) -> dict[str, Any]:
        annotation_path = shard_dir / ANNOTATION_FILENAME
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ("categories", "images", "annotations"):
            if key not in data:
                raise KeyError(
                    f"Shard '{shard_dir.name}' is missing required key '{key}'"
                )
        return data

    def _merge_categories(
        self,
        shard_categories: list[dict[str, Any]],
        reference_categories: list[dict[str, Any]],
        id_to_name: dict[int, str],
        name_to_id: dict[str, int],
        shard_name: str,
    ) -> None:
        """Validate and merge a shard's categories into the running reference.

        For each incoming category ``c`` with id ``cid`` and name ``cname``:

        - If both ``cid`` and ``cname`` are unknown: append a copy of ``c`` to
          ``reference_categories`` and register it in both lookup maps.
        - If ``cid`` and ``cname`` already resolve to the same registered
          category: no-op.
        - If ``cid`` is known but with a different name: raise ``ValueError``
          (id reused with a different name).
        - If ``cname`` is known but with a different id: raise ``ValueError``
          (name reused with a different id).

        ``supercategory`` and any other fields are taken from the first shard
        that introduces a category and are not part of the consistency check.

        ``reference_categories``, ``id_to_name``, and ``name_to_id`` are
        mutated in place. ``shard_name`` is used only for error messages.
        """
        for c in shard_categories:
            cid = c["id"]
            cname = c["name"]
            existing_name = id_to_name.get(cid)
            existing_id = name_to_id.get(cname)

            if existing_name is not None and existing_name != cname:
                raise ValueError(
                    f"Category id conflict in shard '{shard_name}': "
                    f"id={cid} already registered as {existing_name!r}, "
                    f"but shard reports it as {cname!r}."
                )
            if existing_id is not None and existing_id != cid:
                raise ValueError(
                    f"Category name conflict in shard '{shard_name}': "
                    f"name={cname!r} already registered with id={existing_id}, "
                    f"but shard reports it with id={cid}."
                )

            if existing_name is None:
                reference_categories.append(dict(c))
                id_to_name[cid] = cname
                name_to_id[cname] = cid

    def _remap_shard(
        self,
        shard: dict[str, Any],
        shard_dir: Path,
        img_offset: int,
        ann_offset: int,
    ) -> tuple[list[dict], list[dict], int, int]:
        """Re-index IDs and rewrite file paths for a single shard.

        Returns (images, annotations, new_img_offset, new_ann_offset).
        """
        id_map: dict[int, int] = {}

        images = shard["images"]
        for img in images:
            old_id = img["id"]
            new_id = img_offset
            img_offset += 1
            id_map[old_id] = new_id
            img["id"] = new_id
            img["file_name"] = str(Path(shard_dir.name) / img["file_name"])

        annotations = shard["annotations"]
        for ann in annotations:
            ann["id"] = ann_offset
            ann_offset += 1
            ann["image_id"] = id_map[ann["image_id"]]

        return images, annotations, img_offset, ann_offset

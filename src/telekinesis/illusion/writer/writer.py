"""
Defines the Writer class that is responsible for writing scenes into the desired
annotation format.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
import atexit

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported (blenderproc pulls it in). See blender_env.py.
isolate_user_extensions()

import blenderproc as bproc
import bpy
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping

from telekinesis.illusion.writer.coco_writer import write_coco_annotations


class Writer(ABC):
    """
    Abstract class for writers.
    """

    def __init__(
        self,
        create_output_dir_on_init: bool = True,
        output_dir: str = None,
    ) -> None:
        self._create_output_dir_on_init = create_output_dir_on_init
        self._output_dir = self._validate_or_create_output_dir(output_dir)

        # Track whether anything is written into the output directory
        self._has_written_output = False
        # atexit.register(self._cleanup_if_empty)

    def _validate_or_create_output_dir(
        self,
        output_dir: str = None,
    ) -> Path:
        """
        Validates the provided output directory or creates if none is provided.

        Args:
            output_dir: str
                Output directory for writing generated images and annotations.
                If none is provided, the output is written in the directory
                Path.cwd() / "output" / f"synthetic_data_{timestamp}".
        Returns:
            Returns the output directory as a pathlib.Path() object.

        Raises:
            FileNotFoundError:
                - If output_dir does not exists.
            NotADirectoryError:
                - If provided output_directory is not a directory.
        """
        if output_dir is None:
            if self._create_output_dir_on_init:
                # Create default output diretory below the current working
                # directory. Where output is written is deliberately unrelated
                # to where assets are read from.
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path.cwd() / "output" / f"synthetic_data_{timestamp}"

                if path.exists():
                    logger.info(f"Using existing output directory {path}.")
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created output directory {path}.")

                return path
            return None

        path = Path(output_dir)

        if path.exists():
            if path.is_dir():
                logger.info(
                    f"Set output directory to the exisiting path {path}"
                )
                return path
            raise NotADirectoryError(
                f"Provided path {path} is not a directory."
            )
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory at {path}")
        return path

    def _cleanup_if_empty(self) -> str:
        """
        Clean up function called automatically at exit, if nothing was wirtten
        to the output directory.
        """
        path = self._output_dir

        if not self._has_written_output:
            if path.exists() and path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                logger.info(f"Removed empty output directory '{path}'.")
            else:
                logger.info(f"Output directory not removed '{path}'.")

    def set_output_dir(self, output_dir: str | Path) -> None:
        """
        Set (or switch) the output directory for subsequent writes.

        Creates the directory if it does not already exist.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._output_dir = path
        self._has_written_output = False
        logger.info(f"Writer output directory set to {path}.")

    def get_output_dir(self) -> str:
        """
        Get the output directory of the writer.
        """
        return str(self._output_dir)

    def get_segmentation_output_config(self) -> Dict[str, Any]:
        """Describe the BlenderProc segmentation pass required by the writer."""
        return {
            "map_by": ["category_id", "instance", "name"],
            "default_values": {"category_id": 0},
        }

    @abstractmethod
    def write(
        self,
        data: Dict[str, Any],
        categories: Dict[int, str],
    ) -> int:
        pass


class CocoWriter(Writer):
    """
    Class for writing into the COCO format
    """

    def __init__(
        self,
        create_output_dir_on_init: bool = True,
        output_dir: str = None,
        info: Optional[Dict] = None,
        licenses: Optional[List] = None,
        supercategory_map: Optional[Dict] = None,
        compose_parent_masks: bool = False,
        include_camera_metadata: bool = False,
    ) -> None:
        """Add info, licenses and supercategory info from the specs if present"""
        super().__init__(create_output_dir_on_init, output_dir)
        if info:
            self._info = info
        else:
            self._info = {
                "description": "Synthetic instance-segmentation dataset created with Illusion.",
                "url": "https://github.com/telekinesis-ai",
                "version": "1.0.0",
                "year": 2026,
                "contributor": "Telekinesis AI",
            }

        if licenses:
            self._licenses = licenses
        else:
            self._licenses = [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike 4.0 International",
                    "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                }
            ]

        if supercategory_map:
            self._supercategory_map = supercategory_map
        else:
            self._supercategory_map = {}
        self._compose_parent_masks = compose_parent_masks
        self._include_camera_metadata = include_camera_metadata
        self._next_image_metadata: Dict[str, Any] = {}

    def set_image_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata copied onto each image in the next write call."""
        self._next_image_metadata = dict(metadata)

    def get_segmentation_output_config(self) -> Dict[str, Any]:
        config = super().get_segmentation_output_config()
        if self._compose_parent_masks:
            config["map_by"].append("annotation_parent")
            config["default_values"]["annotation_parent"] = ""
        return config

    def write(
        self,
        data: Dict[str, Any],
        categories: LabelIdMapping,
    ) -> int:
        required_keys = {
            "instance_segmaps",
            "instance_attribute_maps",
            "colors",
        }

        missing = required_keys - data.keys()
        if missing:
            raise KeyError(f"Missing required data keys: {missing}")

        image_metadata = []
        current_frame = bpy.context.scene.frame_current
        for frame in range(
            bpy.context.scene.frame_start, bpy.context.scene.frame_end
        ):
            metadata = dict(self._next_image_metadata)
            if self._include_camera_metadata:
                bpy.context.scene.frame_set(frame)
                camera = bpy.context.scene.camera
                metadata.update(
                    {
                        "camera_matrix_world": [
                            list(row) for row in camera.matrix_world
                        ],
                        "camera_lens_mm": camera.data.lens,
                    }
                )
            image_metadata.append(metadata)
        bpy.context.scene.frame_set(current_frame)

        write_coco_annotations(
            output_dir=str(self._output_dir),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="PNG",
            mask_encoding_format="rle",
            supercategory=None,
            supercategory_map=self._supercategory_map,
            info=self._info,
            licenses=self._licenses,
            append_to_existing_output=True,
            jpg_quality=95,
            label_mapping=categories,
            file_prefix="",
            indent=None,
            compose_parent_masks=self._compose_parent_masks,
            image_metadata=image_metadata,
        )
        self._next_image_metadata = {}

        self._has_written_output = True
        return len(data["colors"])

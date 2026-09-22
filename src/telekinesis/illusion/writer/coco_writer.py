"""Allows rendering the content of the scene in the coco file format."""

import datetime
import json
import os
from typing import Optional, Dict, Union, Tuple, List

from loguru import logger
import numpy as np
from skimage import measure
import cv2
from pycocotools import mask as mask_utils

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

import bpy

from blenderproc.python.utility.LabelIdMapping import LabelIdMapping


def write_coco_annotations(
    output_dir: str,
    instance_segmaps: List[np.ndarray],
    instance_attribute_maps: List[dict],
    colors: List[np.ndarray],
    color_file_format: str = "PNG",
    mask_encoding_format: str = "rle",
    supercategory: str = "coco_annotations",
    supercategory_map: Optional[Dict] = None,
    info: Optional[Dict] = None,
    licenses: Optional[List] = None,
    append_to_existing_output: bool = True,
    jpg_quality: int = 95,
    label_mapping: Optional[LabelIdMapping] = None,
    file_prefix: str = "",
    indent: Optional[Union[int, str]] = None,
    compose_parent_masks: bool = False,
    image_metadata: Optional[List[Dict]] = None,
):
    """Writes coco annotations in the following steps:
    1. Locate the seg images
    2. Locate the rgb maps
    3. Locate the seg mappings
    4. Read color mappings
    5. For each frame write the coco annotation

    :param output_dir: Output directory to write the coco annotations
    :param instance_segmaps: List of instance segmentation maps
    :param instance_attribute_maps: per-frame mappings with idx, class and optionally supercategory/bop_dataset_name
    :param colors: List of color images. Does not support stereo images, enter left and right inputs subsequently.
    :param color_file_format: Format to save color images in
    :param mask_encoding_format: Encoding format of the binary masks. Default: 'rle'. Available: 'rle', 'polygon'.
    :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset set
                          by 'bop_dataset_name' or any loaded object with specified 'cp_supercategory'
    :param append_to_existing_output: If true and if there is already a coco_annotations.json file in the output
                                      directory, the new coco annotations will be appended to the existing file.
                                      Also, the rgb images will be named such that there are no collisions.
    :param jpg_quality: The desired quality level of the jpg encoding
    :param label_mapping: The label mapping which should be used to label the categories based on their ids.
                          If None, is given then the `name` field in the csv files is used or - if not existing -
                          the category id itself is used.
    :param file_prefix: Optional prefix for image file names
    :param indent: If indent is a non-negative integer or string, then the annotation output
                   will be pretty-printed with that indent level. An indent level of 0, negative, or "" will
                   only insert newlines. None (the default) selects the most compact representation.
                   Using a positive integer indent indents that many spaces per level.
                   If indent is a string (such as "\t"), that string is used to indent each level.
    """

    if len(colors) > 0 and len(colors[0].shape) == 4:
        raise ValueError(
            "BlenderProc currently does not support writing coco annotations for stereo images. "
            "However, you can enter left and right images / segmaps separately."
        )

    # Create output directory
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    coco_annotations_path = os.path.join(output_dir, "coco_annotations.json")
    # Calculate image numbering offset, if append_to_existing_output is activated and coco data exists
    if append_to_existing_output and os.path.exists(coco_annotations_path):
        with open(coco_annotations_path, "r", encoding="utf-8") as fp:
            existing_coco_annotations = json.load(fp)
        image_offset = (
            max(image["id"] for image in existing_coco_annotations["images"])
            + 1
        )
    else:
        image_offset = 0
        existing_coco_annotations = None

    # collect all RGB paths
    new_coco_image_paths = []

    # for each rendered frame
    for frame in range(
        bpy.context.scene.frame_start, bpy.context.scene.frame_end
    ):
        color_rgb = colors[frame - bpy.context.scene.frame_start]

        # Reverse channel order for opencv
        color_bgr = color_rgb.copy()
        color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

        if color_file_format == "PNG":
            target_base_path = (
                f"images/{file_prefix}{frame + image_offset:06d}.png"
            )
            target_path = os.path.join(output_dir, target_base_path)
            cv2.imwrite(target_path, color_bgr)
        elif color_file_format == "JPEG":
            target_base_path = (
                f"images/{file_prefix}{frame + image_offset:06d}.jpg"
            )
            target_path = os.path.join(output_dir, target_base_path)
            cv2.imwrite(
                target_path,
                color_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality],
            )
        else:
            raise RuntimeError(
                f'Unknown color_file_format={color_file_format}. Try "PNG" or "JPEG"'
            )

        new_coco_image_paths.append(target_base_path)

    coco_output = _CocoWriterUtility.generate_coco_annotations(
        instance_segmaps,
        instance_attribute_maps,
        new_coco_image_paths,
        supercategory,
        supercategory_map,
        info,
        licenses,
        mask_encoding_format,
        existing_coco_annotations,
        label_mapping,
        compose_parent_masks,
        image_metadata,
    )

    print("Writing coco annotations to " + coco_annotations_path)
    with open(coco_annotations_path, "w", encoding="utf-8") as fp:
        json.dump(coco_output, fp, indent=indent)


def binary_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Union[str, List[int]]]:
    """Encode a binary mask as compressed, JSON-safe COCO RLE.

    ``pycocotools`` performs the pixel scan in compiled code. Its encoder
    returns the compressed counts as bytes, which are converted to ASCII for
    JSON serialization.
    """
    rle = mask_utils.encode(
        np.asfortranarray(binary_mask, dtype=np.uint8)
    )
    rle["counts"] = rle["counts"].decode("ascii")
    rle["size"] = [int(value) for value in rle["size"]]
    return rle


def rle_to_binary_mask(
    rle: Dict[str, Union[str, List[int]]],
) -> np.ndarray:
    """Decode compressed or legacy uncompressed COCO RLE."""
    encoded = rle
    if isinstance(rle.get("counts"), list):
        height, width = rle["size"]
        encoded = mask_utils.frPyObjects(rle, height, width)
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(bool)


class _CocoWriterUtility:
    @staticmethod
    def generate_coco_annotations(
        inst_segmaps,
        inst_attribute_maps,
        image_paths,
        supercategory,
        supercategory_map,
        info,
        licenses,
        mask_encoding_format,
        existing_coco_annotations=None,
        label_mapping: LabelIdMapping = None,
        compose_parent_masks: bool = False,
        image_metadata: Optional[List[Dict]] = None,
    ):
        """Generates coco annotations for images

        :param inst_segmaps: List of instance segmentation maps
        :param inst_attribute_maps: per-frame mappings with idx, class and optionally supercategory/bop_dataset_name
        :param image_paths: A list of paths which points to the rendered images.
        :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset
        :param mask_encoding_format: Encoding format of the binary mask. Type: string.
        :param existing_coco_annotations: If given, the new coco annotations will be appended to the given
                                          coco annotations dict.
        :param label_mapping: The label mapping which should be used to label the categories based on their ids.
                              If None, is given then the `name` field in the csv files is used or - if not existing -
                              the category id itself is used.
        :return: dict containing coco annotations
        """

        categories = []
        visited_categories = []
        instance_2_category_maps = []

        for inst_attribute_map in inst_attribute_maps:
            instance_2_category_map = {}
            for inst in inst_attribute_map:
                # skip background
                if int(inst["category_id"]) != 0:
                    # take all objects or objects from specified supercategory is defined
                    inst_supercategory = "coco_annotations"
                    if "bop_dataset_name" in inst:
                        inst_supercategory = inst["bop_dataset_name"]
                    elif "supercategory" in inst:
                        inst_supercategory = inst["supercategory"]

                    if supercategory in [
                        inst_supercategory,
                        "coco_annotations",
                    ]:
                        if int(inst["category_id"]) not in visited_categories:
                            cat_dict: Dict[str, Union[str, int]] = {
                                "id": int(inst["category_id"]),
                                "supercategory": inst_supercategory,
                            }
                            # Determine name of category based on label_mapping, name or category_id
                            if label_mapping is not None:
                                cat_dict["name"] = label_mapping.label_from_id(
                                    cat_dict["id"]
                                )
                            elif "name" in inst:
                                cat_dict["name"] = inst["name"]
                            else:
                                cat_dict["name"] = inst["category_id"]

                            categories.append(cat_dict)
                            visited_categories.append(cat_dict["id"])
                        instance_2_category_map[int(inst["idx"])] = int(
                            inst["category_id"]
                        )
                    else:
                        if int(inst["category_id"]) not in visited_categories:
                            if supercategory_map:
                                cat_dict: Dict[str, Union[str, int]] = {
                                    "id": int(inst["category_id"]),
                                    "supercategory": supercategory_map[
                                        int(inst["category_id"])
                                    ],
                                }
                            else:
                                cat_dict: Dict[str, Union[str, int]] = {
                                    "id": int(inst["category_id"]),
                                }
                            # Determine name of category based on label_mapping, name or category_id
                            if label_mapping is not None:
                                cat_dict["name"] = label_mapping.label_from_id(
                                    cat_dict["id"]
                                )
                            elif "name" in inst:
                                cat_dict["name"] = inst["name"]
                            else:
                                cat_dict["name"] = inst["category_id"]

                            categories.append(cat_dict)
                            visited_categories.append(cat_dict["id"])
                        instance_2_category_map[int(inst["idx"])] = int(
                            inst["category_id"]
                        )
            instance_2_category_maps.append(instance_2_category_map)

        images: List[Dict[str, Union[str, int]]] = []
        annotations: List[Dict[str, Union[str, int]]] = []

        for frame_index, (
            inst_segmap,
            image_path,
            instance_2_category_map,
            inst_attribute_map,
        ) in enumerate(zip(
            inst_segmaps,
            image_paths,
            instance_2_category_maps,
            inst_attribute_maps,
        )):
            # Add coco info for image
            image_id = len(images)
            image_info = _CocoWriterUtility.create_image_info(
                image_id, image_path, inst_segmap.shape
            )
            if image_metadata and frame_index < len(image_metadata):
                image_info.update(image_metadata[frame_index])
            images.append(image_info)

            # Go through all objects visible in this image.  A child may opt
            # into overlapping COCO annotations by naming an annotation parent;
            # its visible mask is then also included in that parent's mask.
            instances = np.unique(inst_segmap)
            # Remove background. Parent-mask composition also considers an
            # otherwise hidden parent when one of its children is visible.
            instances = set(
                np.delete(instances, np.where(instances == 0)).tolist()
            )
            visible_attributes = {
                int(item["idx"]): item for item in inst_attribute_map
            }
            name_to_instance = {
                item.get("name"): int(item["idx"])
                for item in inst_attribute_map
                if item.get("name")
            }
            children_by_parent = {}
            if compose_parent_masks:
                for child in inst_attribute_map:
                    parent_name = child.get("annotation_parent")
                    if parent_name:
                        children_by_parent.setdefault(parent_name, []).append(
                            int(child["idx"])
                        )
            if compose_parent_masks:
                for child in inst_attribute_map:
                    child_idx = int(child["idx"])
                    if (
                        child_idx in instances
                        and child.get("annotation_parent") in name_to_instance
                    ):
                        instances.add(
                            name_to_instance[child["annotation_parent"]]
                        )
            annotation_by_instance = {}
            mask_cache = {}

            def instance_mask(instance_id):
                instance_id = int(instance_id)
                if instance_id not in mask_cache:
                    mask_cache[instance_id] = inst_segmap == instance_id
                return mask_cache[instance_id]

            for inst in instances:
                if inst in instance_2_category_map:
                    inst_attributes = visible_attributes.get(int(inst))
                    if inst_attributes:
                        inst_attributes = {
                            k: v
                            for k, v in inst_attributes.items()
                            if k not in {"idx", "name"}
                        }
                        if len(inst_attributes) == 0:
                            inst_attributes = None
                    # Calc object mask
                    binary_inst_mask = instance_mask(inst)
                    if compose_parent_masks:
                        instance_name = visible_attributes.get(
                            int(inst), {}
                        ).get("name")
                        child_ids = children_by_parent.get(instance_name, [])
                        if child_ids:
                            binary_inst_mask = binary_inst_mask.copy()
                            for child_idx in child_ids:
                                binary_inst_mask |= instance_mask(child_idx)
                    # Add coco info for object in this image
                    annotation = _CocoWriterUtility.create_annotation_info(
                        len(annotations) + 1,
                        image_id,
                        instance_2_category_map[inst],
                        binary_inst_mask,
                        mask_encoding_format,
                        custom_props=inst_attributes,
                    )
                    if annotation is not None:
                        annotations.append(annotation)
                        annotation_by_instance[int(inst)] = annotation

            if compose_parent_masks:
                for child_idx, child in visible_attributes.items():
                    parent_name = child.get("annotation_parent")
                    if not parent_name or child_idx not in annotation_by_instance:
                        continue
                    parent_idx = name_to_instance.get(parent_name)
                    parent_annotation = annotation_by_instance.get(parent_idx)
                    if parent_annotation is not None:
                        annotation_by_instance[child_idx][
                            "parent_annotation_id"
                        ] = parent_annotation["id"]

        new_coco_annotations = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        if existing_coco_annotations is not None:
            new_coco_annotations = _CocoWriterUtility.merge_coco_annotations(
                existing_coco_annotations, new_coco_annotations
            )

        return new_coco_annotations

    @staticmethod
    def merge_coco_annotations(existing_coco_annotations, new_coco_annotations):
        """Merges the two given coco annotation dicts into one.

        Currently, this requires both coco annotations to have the exact same categories/objects.
        The "images" and "annotations" sections are concatenated and respective ids are adjusted.

        :param existing_coco_annotations: A dict describing the first coco annotations.
        :param new_coco_annotations: A dict describing the second coco annotations.
        :return: A dict containing the merged coco annotations.
        """

        # Concatenate category sections
        for cat_dict in new_coco_annotations["categories"]:
            if cat_dict not in existing_coco_annotations["categories"]:
                existing_coco_annotations["categories"].append(cat_dict)

        # Concatenate images sections
        image_id_offset = (
            max(image["id"] for image in existing_coco_annotations["images"])
            + 1
        )
        for image in new_coco_annotations["images"]:
            image["id"] += image_id_offset
        existing_coco_annotations["images"].extend(
            new_coco_annotations["images"]
        )

        # Concatenate annotations sections
        if len(existing_coco_annotations["annotations"]) > 0:
            annotation_id_offset = max(
                annotation["id"]
                for annotation in existing_coco_annotations["annotations"]
            )
        else:
            annotation_id_offset = 0
        for annotation in new_coco_annotations["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            if "parent_annotation_id" in annotation:
                annotation["parent_annotation_id"] += annotation_id_offset
        existing_coco_annotations["annotations"].extend(
            new_coco_annotations["annotations"]
        )

        return existing_coco_annotations

    @staticmethod
    def create_image_info(
        image_id: int, file_name: str, image_size: Tuple[int, int]
    ) -> Dict[str, Union[str, int]]:
        """Creates image info section of coco annotation

        :param image_id: integer to uniquly identify image
        :param file_name: filename for image
        :param image_size: The size of the image, given as [W, H]
        """
        image_info: Dict[str, Union[str, int]] = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": datetime.datetime.utcnow().isoformat(" "),
            "license": 1,
            "coco_url": "",
            "flickr_url": "",
        }

        return image_info

    @staticmethod
    def create_annotation_info(
        annotation_id: int,
        image_id: int,
        category_id: int,
        binary_mask: np.ndarray,
        mask_encoding_format: str,
        tolerance: int = 2,
        custom_props: Optional[Dict] = None,
    ) -> Optional[Dict[str, Union[str, int]]]:
        """Creates info section of coco annotation

        :param annotation_id: integer to uniquly identify the annotation
        :param image_id: integer to uniquly identify image
        :param category_id: Id of the category
        :param binary_mask: A binary image mask of the object with the shape [H, W].
        :param mask_encoding_format: Encoding format of the mask. Type: string.
        :param tolerance: The tolerance for fitting polygons to the objects mask.
        :param custom_props: A dictionary of custom properties to add to the annotation.
        """

        area = _CocoWriterUtility.calc_binary_mask_area(binary_mask)
        if area < 1:
            return None

        bounding_box = _CocoWriterUtility.bbox_from_binary_mask(binary_mask)

        if mask_encoding_format == "rle":
            segmentation = binary_mask_to_rle(binary_mask)
        elif mask_encoding_format == "polygon":
            segmentation = _CocoWriterUtility.binary_mask_to_polygon(
                binary_mask, tolerance
            )
            if not segmentation:
                return None
        else:
            raise RuntimeError(
                f"Unknown encoding format: {mask_encoding_format}"
            )

        annotation_info: Dict[str, Union[str, int]] = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": 0,
            "area": area,
            "bbox": bounding_box,
            "segmentation": segmentation,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }

        if custom_props:
            annotation_info.update(custom_props)

        return annotation_info

    @staticmethod
    def bbox_from_binary_mask(binary_mask: np.ndarray) -> List[int]:
        """Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

        :param binary_mask: A binary image mask with the shape [H, W].
        :return: The bounding box represented as [x, y, width, height]
        """
        # Find all columns and rows that contain 1s
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        # Find the min and max col/row index that contain 1s
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Calc height and width
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        return [int(cmin), int(rmin), int(w), int(h)]

    @staticmethod
    def calc_binary_mask_area(binary_mask: np.ndarray) -> int:
        """Returns the area of the given binary mask which is defined as the number of 1s in the mask.

        :param binary_mask: A binary image mask with the shape [H, W].
        :return: The computed area
        """
        return binary_mask.sum().tolist()

    @staticmethod
    def close_contour(contour: np.ndarray) -> np.ndarray:
        """Makes sure the given contour is closed.

        :param contour: The contour to close.
        :return: The closed contour.
        """
        # If first != last point => add first point to end of contour to close it
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    @staticmethod
    def binary_mask_to_polygon(
        binary_mask: np.ndarray, tolerance: int = 0
    ) -> List[np.ndarray]:
        """Converts a binary mask to COCO polygon representation

        :param binary_mask: a 2D binary numpy array where '1's represent the object
        :param tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If
                          tolerance is 0, the original coordinate array is returned.
        """
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(
            binary_mask, pad_width=1, mode="constant", constant_values=0
        )
        contours = np.array(measure.find_contours(padded_binary_mask, 0.5))
        # Reverse padding
        contours -= 1
        for contour in contours:
            # Make sure contour is closed
            contour = _CocoWriterUtility.close_contour(contour)
            # Approximate contour by polygon
            polygon = measure.approximate_polygon(contour, tolerance)
            # Skip invalid polygons
            if len(polygon) < 3:
                continue
            # Flip xy to yx point representation
            polygon = np.flip(polygon, axis=1)
            # Flatten
            polygon = polygon.ravel()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            polygon[polygon < 0] = 0
            polygons.append(polygon.tolist())

        return polygons

import numpy as np

from blenderproc.python.utility.LabelIdMapping import LabelIdMapping

from telekinesis.illusion.writer.coco_writer import (
    _CocoWriterUtility,
    binary_mask_to_rle,
    rle_to_binary_mask,
)
from telekinesis.illusion.writer.writer import CocoWriter


def test_parent_mask_contains_visible_child_and_links_annotation():
    segmap = np.array([[0, 1, 2], [0, 0, 2]])
    attributes = [
        {
            "idx": 1,
            "category_id": 2,
            "name": "Wire_001",
            "annotation_parent": "",
        },
        {
            "idx": 2,
            "category_id": 1,
            "name": "Metal_tip_001",
            "annotation_parent": "Wire_001",
        },
    ]
    labels = LabelIdMapping()
    labels.add("wire_tip", 1)
    labels.add("wire_blue", 2)

    coco = _CocoWriterUtility.generate_coco_annotations(
        [segmap],
        [attributes],
        ["images/000000.png"],
        "coco_annotations",
        {},
        None,
        None,
        "rle",
        label_mapping=labels,
        compose_parent_masks=True,
    )

    annotations = {
        annotation["category_id"]: annotation
        for annotation in coco["annotations"]
    }
    assert annotations[2]["area"] == 3
    assert annotations[1]["area"] == 2
    assert (
        annotations[1]["parent_annotation_id"] == annotations[2]["id"]
    )


def test_parent_mask_writer_requests_relationship_property(tmp_path):
    writer = CocoWriter(
        output_dir=str(tmp_path), compose_parent_masks=True
    )
    config = writer.get_segmentation_output_config()
    assert "annotation_parent" in config["map_by"]
    assert config["default_values"]["annotation_parent"] == ""


def test_compressed_rle_round_trip():
    mask = np.array([[0, 1, 1], [1, 0, 0]], dtype=bool)
    rle = binary_mask_to_rle(mask)
    assert isinstance(rle["counts"], str)
    np.testing.assert_array_equal(rle_to_binary_mask(rle), mask)


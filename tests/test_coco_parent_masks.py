import numpy as np
import pytest
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping

from telekinesis.illusion.writer import writer as writer_module
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


@pytest.mark.parametrize(
    "writer_kwargs,expected_quality",
    [({}, 90), ({"color_file_format": "jpg", "jpg_quality": 87}, 87)],
)
def test_coco_writer_forwards_jpeg_format_and_quality(
    tmp_path, monkeypatch, writer_kwargs, expected_quality
):
    captured = {}

    def capture_write(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        writer_module, "write_coco_annotations", capture_write
    )
    writer = CocoWriter(output_dir=str(tmp_path), **writer_kwargs)

    written = writer.write(
        {
            "instance_segmaps": [np.zeros((1, 1), dtype=np.uint8)],
            "instance_attribute_maps": [[]],
            "colors": [np.zeros((1, 1, 3), dtype=np.uint8)],
        },
        LabelIdMapping(),
    )

    assert written == 1
    assert captured["color_file_format"] == "JPEG"
    assert captured["jpg_quality"] == expected_quality


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"color_file_format": "WEBP"}, ValueError, "color_file_format"),
        ({"color_file_format": None}, TypeError, "color_file_format"),
        ({"jpg_quality": -1}, ValueError, "jpg_quality"),
        ({"jpg_quality": 101}, ValueError, "jpg_quality"),
        ({"jpg_quality": 95.0}, TypeError, "jpg_quality"),
    ],
)
def test_coco_writer_validates_image_encoding(
    tmp_path, kwargs, error, match
):
    with pytest.raises(error, match=match):
        CocoWriter(output_dir=str(tmp_path), **kwargs)


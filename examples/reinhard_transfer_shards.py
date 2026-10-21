"""Apply Reinhard-style Lab color transfer to COCO shards or exports.

The input layout is detected automatically from its COCO annotation filenames.
References are discovered recursively; one is sampled per image with --seed.
Uses CIE Lab, as in demo.py, with floating-point statistics and a flat-channel
guard. Method: Reinhard et al., Color Transfer between Images (2001),
https://users.cs.northwestern.edu/~bgooch/PDFs/ColorTransfer.pdf
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ANNOTATIONS = "coco_annotations.json"
EXPORTED_ANNOTATIONS = "_annotations.coco.json"


def read_image(path: Path) -> np.ndarray:
    """Read without EXIF rotation; support Unicode paths on Windows."""
    image = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected a grayscale, BGR or BGRA image: {path}")
    return image


def lab_statistics(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(
        image[..., :3].astype(np.float32) / 255, cv2.COLOR_BGR2LAB
    )
    pixels = (
        lab[image[..., 3] > 0] if image.shape[2] == 4 else lab.reshape(-1, 3)
    )
    if not len(pixels):
        raise ValueError(
            "Cannot transfer colors from an entirely transparent image."
        )
    return (
        lab,
        pixels.mean(axis=0, dtype=np.float64),
        pixels.std(axis=0, dtype=np.float64),
    )


def reinhard_transfer(
    image: np.ndarray, reference_mean: np.ndarray, reference_std: np.ndarray
) -> np.ndarray:
    """Match Lab means/stds while retaining image dimensions and alpha."""
    lab, mean, std = lab_statistics(image)
    scale = np.divide(reference_std, std, out=np.ones(3), where=std > 1e-6)
    adjusted = ((lab - mean) * scale + reference_mean).astype(np.float32)
    bgr = cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
    result = image.copy()
    result[..., :3] = np.rint(np.clip(bgr, 0, 1) * 255).astype(np.uint8)
    return result


def discover_annotations(root: Path) -> list[Path]:
    """Recognize native/exported COCO at the root or one directory below it."""
    directories = [root, *sorted(p for p in root.iterdir() if p.is_dir())]
    annotations = [
        directory / name
        for directory in directories
        for name in (ANNOTATIONS, EXPORTED_ANNOTATIONS)
        if (directory / name).is_file()
    ]
    if not annotations:
        raise ValueError(
            f"No COCO dataset found in {root}. Expected {ANNOTATIONS} or "
            f"{EXPORTED_ANNOTATIONS} directly or in immediate subdirectories."
        )
    if len({path.name for path in annotations}) > 1 or (
        annotations[0].parent == root and len(annotations) > 1
    ):
        raise ValueError(
            "Ambiguous COCO layout: provide one native shard dataset, one "
            "exported dataset, or one individual shard/split directory."
        )
    return annotations


def coco_images(annotation_path: Path) -> dict[Path, dict]:
    """Validate COCO paths before any output is created."""
    directory = annotation_path.parent
    with annotation_path.open(encoding="utf-8") as stream:
        coco = json.load(stream)
    if not isinstance(coco, dict) or not isinstance(coco.get("images"), list):
        raise TypeError(f"Missing COCO images list: {annotation_path}")
    images = {}
    for entry in coco["images"]:
        relative = Path(entry["file_name"])
        path = (directory / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(directory):
            raise ValueError(
                f"COCO image must be relative to its annotation directory: {relative}"
            )
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported COCO image extension: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {path}")
        if path in images:
            raise ValueError(f"Duplicate COCO image path: {path}")
        images[path] = entry
    return images


def transform_shards(
    reference_dir: Path, synthetic_dir: Path, output_dir: Path, seed: int = 42
) -> tuple[int, int]:
    """Auto-detect native or exported COCO and create a new dataset."""
    reference_dir = reference_dir.resolve()
    synthetic_dir = synthetic_dir.resolve()
    output_dir = output_dir.resolve()
    for directory in (reference_dir, synthetic_dir):
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Input directory does not exist: {directory}"
            )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if any(
        output_dir.is_relative_to(d) for d in (reference_dir, synthetic_dir)
    ):
        raise ValueError(
            "Output directory must be outside both input directories."
        )

    references = sorted(
        p
        for p in reference_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not references:
        raise ValueError(
            f"No PNG/JPG/JPEG reference images found in {reference_dir}"
        )
    annotation_files = discover_annotations(synthetic_dir)
    datasets = {path.parent: coco_images(path) for path in annotation_files}
    total = sum(len(images) for images in datasets.values())
    if not total:
        raise ValueError(f"No COCO images found in {synthetic_dir}")
    layout = (
        "native COCO"
        if annotation_files[0].name == ANNOTATIONS
        else "exported COCO"
    )
    print(f"Detected {layout}: {len(datasets)} annotation file(s).")

    reference_stats = []
    for path in tqdm(references, desc="Reading references", unit="image"):
        _, mean, std = lab_statistics(read_image(path))
        reference_stats.append((mean, std))

    rng = random.Random(seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete dataset; failures clean up our temporary copy.
    with (
        tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-", dir=output_dir.parent
        ) as temporary,
        tqdm(total=total, desc="Transferring colors", unit="image") as progress,
    ):
        staging = Path(temporary)
        for directory, images in datasets.items():
            targets = {
                path: rng.choice(reference_stats) for path in sorted(images)
            }

            def copy_file(source, destination, images=images, targets=targets):
                path = Path(source).resolve()
                if path not in images:
                    return shutil.copy2(source, destination)
                image = read_image(path)
                entry = images[path]
                if image.shape[:2] != (entry["height"], entry["width"]):
                    raise ValueError(
                        f"Image dimensions disagree with COCO: {path}"
                    )
                result = reinhard_transfer(image, *targets[path])
                params = (
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                    if path.suffix.lower() in {".jpg", ".jpeg"}
                    else []
                )
                ok, encoded = cv2.imencode(path.suffix.lower(), result, params)
                if not ok:
                    raise OSError(f"Could not encode image: {path}")
                encoded.tofile(destination)
                progress.update()
                return str(destination)

            shutil.copytree(
                directory,
                staging / directory.relative_to(synthetic_dir),
                copy_function=copy_file,
                dirs_exist_ok=True,
            )
        merged = synthetic_dir / "merged_coco_annotations.json"
        if synthetic_dir not in datasets and merged.is_file():
            shutil.copy2(merged, staging / merged.name)
        staging.rename(output_dir)
    return len(datasets), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Folder of real PNG/JPG/JPEG images (searched recursively).",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        required=True,
        help="Native shards, an exported COCO dataset, or one shard/split (auto-detected).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New destination folder; must not already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reference sampling seed (default: 42).",
    )
    args = parser.parse_args()
    try:
        directories, images = transform_shards(
            args.reference_dir, args.synthetic_dir, args.output_dir, args.seed
        )
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(
        f"Wrote {images} images in {directories} COCO directories to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

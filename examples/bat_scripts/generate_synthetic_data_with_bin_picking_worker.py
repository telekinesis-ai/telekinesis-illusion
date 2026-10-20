"""Generate a bin-picking dataset with CLI overrides for batch runs."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main(argv: list[str] | None = None) -> None:
    """Generate and merge a dataset using command-line overrides.

    Args:
        argv: Arguments to parse, or None to read the process arguments.
    """
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    configs_dir = Path(__file__).resolve().parents[2] / "configs"

    parser = argparse.ArgumentParser(
        description=(
            "Generate a bin-picking dataset using BinPickingWorker, then merge "
            "the per-shard COCO annotations into a single file."
        )
    )

    parser.add_argument(
        "--spec-file",
        type=Path,
        default=configs_dir / "example_bin_picking_gearwheel_2.yaml",
        help=(
            "Path to the bin-picking spec YAML. Bare filenames are resolved "
            "against the repo's 'configs' directory (e.g. 'example_bin_picking_gearwheel_2.yaml')."
        ),
    )

    parser.add_argument(
        "--dataset-name", help="Dataset subdirectory name (overrides YAML)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base output directory; writes OUTPUT_DIR/DATASET_NAME.",
    )
    parser.add_argument(
        "--num-images", type=int, help="Override metadata.num_images."
    )
    parser.add_argument("--shard-size", type=int, help="Override shard.size.")
    parser.add_argument("--dataset-format", choices=("coco", "yolo"))
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the merged dataset in the FiftyOne viewer after merging.",
    )

    parser.add_argument(
        "--resolution", type=int, help="Override both image dimensions."
    )
    parser.add_argument(
        "--samples", type=int, help="Override Cycles sample count."
    )
    parser.add_argument(
        "--seed", type=int, help="Set Python and NumPy random seeds."
    )
    args = parser.parse_args(argv)
    if args.seed is not None:
        import random
        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)

    for name in ("num_images", "shard_size", "resolution", "samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.dataset_name is not None and (
        not args.dataset_name.strip()
        or args.dataset_name in (".", "..")
        or Path(args.dataset_name).name != args.dataset_name
    ):
        parser.error("--dataset-name must be a directory name, not a path")

    spec_path: Path = args.spec_file
    if not spec_path.is_absolute() and not spec_path.exists():
        spec_path = configs_dir / spec_path
    if not spec_path.is_file():
        parser.error(f"Spec file does not exist: {spec_path}")

    import yaml

    with spec_path.open(encoding="utf-8") as file:
        specs = yaml.safe_load(file)
    if args.resolution is not None:
        specs.setdefault("camera", {}).update(
            image_width=args.resolution, image_height=args.resolution
        )
    if args.samples is not None:
        specs.setdefault("renderer", {})["samples"] = args.samples
    metadata = specs["metadata"]
    if args.dataset_name is not None:
        metadata["dataset_name"] = args.dataset_name
    if args.output_dir is not None:
        metadata["base_output_directory"] = str(args.output_dir.resolve())
    if args.num_images is not None:
        metadata["num_images"] = args.num_images
    if args.shard_size is not None:
        specs["shard"]["size"] = args.shard_size
    if args.dataset_format is not None:
        specs.setdefault("output", {})["dataset_format"] = args.dataset_format

    # Keep asset paths anchored to the original YAML, not the temporary copy.
    if metadata.get("asset_directory") is not None:
        metadata["asset_directory"] = str(
            (spec_path.resolve().parent / metadata["asset_directory"]).resolve()
        )

    dataset_dir = (
        Path(metadata.get("base_output_directory") or "output")
        / metadata["dataset_name"]
    ).resolve()
    dataset_format = specs.get("output", {}).get("dataset_format")
    export_dir = dataset_dir / f"{dataset_format}_format"
    if dataset_format and export_dir.exists():
        parser.error(
            f"Export already exists: {export_dir}. Use a new --dataset-name "
            "or --output-dir to avoid overwriting it."
        )

    from telekinesis.illusion.workers.bin_picking_worker import BinPickingWorker

    with TemporaryDirectory(prefix="illusion-spec-") as temporary_dir:
        effective_spec = Path(temporary_dir) / spec_path.name
        effective_spec.write_text(yaml.safe_dump(specs), encoding="utf-8")
        worker = BinPickingWorker(effective_spec)
        worker.generate()
        worker.merge_shards(preview=args.preview)
    print(f"Dataset ready: {export_dir if dataset_format else dataset_dir}")


if __name__ == "__main__":
    from _run import finish_batch_run

    main()
    finish_batch_run()

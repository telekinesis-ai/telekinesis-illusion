"""Generate a bin-picking dataset with BinPickingWorker."""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
from pathlib import Path

from telekinesis.illusion.workers.bin_picking_worker import BinPickingWorker


def main() -> None:

    configs_dir = Path(__file__).resolve().parent.parent / "configs"

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
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the merged dataset in the FiftyOne viewer after merging.",
    )

    args = parser.parse_args()

    spec_path: Path = args.spec_file
    if not spec_path.is_absolute() and not spec_path.exists():
        spec_path = configs_dir / spec_path

    worker = BinPickingWorker(spec_path)
    worker.generate()
    worker.merge_shards(preview=args.preview)


if __name__ == "__main__":
    main()

"""Generate a bin-picking dataset using a YAML configuration."""

import os
from pathlib import Path


def main() -> None:
    """Generate and merge the dataset described by the example YAML."""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    from telekinesis.illusion.workers.bin_picking_worker import BinPickingWorker

    # Choose a YAML file and edit its dataset settings in configs/.
    spec_file = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "bin_picking_epson.yaml"
    )
    worker = BinPickingWorker(spec_file)
    worker.generate()
    worker.merge_shards(preview=False)


if __name__ == "__main__":
    main()

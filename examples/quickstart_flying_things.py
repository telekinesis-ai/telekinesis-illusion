"""Generate a "flying things"-type dataset."""

import numpy as np

from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.core.context import Context
from telekinesis.illusion.types.object import Object
from telekinesis.illusion.sampler.camera_pose_sampler import shell_sampler
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    ObjectPoseRandomizer,
    ObjectInstanceRandomizer,
    BackgroundRandomizer,
    MaterialRandomizer,
    CameraPoseRandomizer,
)
from telekinesis.illusion.writer.writer import CocoWriter
from telekinesis.illusion.viewer.shard_viewer import view_coco
from telekinesis.illusion.utils.assets import resolve_asset_dir


def main():
    # Create the context
    context = Context()

    assets_dir = resolve_asset_dir()

    # Add models to the context
    model_1_path = str(
        assets_dir / "models" / "mechanical_parts" / "gearwheel_1.glb"
    )
    context.add_model(
        model_1_path,
        object_name="part_1",
        min_number_instances=1,
        max_number_instances=3,
    )

    model_2_path = str(
        assets_dir / "models" / "mechanical_parts" / "pipe_1.glb"
    )
    context.add_model(
        model_2_path,
        object_name="part_2",
        min_number_instances=1,
        max_number_instances=1,
    )

    # Create the randomizer
    randomizer = Randomizer()

    # Add obect instance randomizer
    object_instance_randomizer = ObjectInstanceRandomizer(
        target_objects=["part_1", "part_2"],
        min_num_total_objects=2,
        max_num_total_objects=4,
    )

    randomizer.add_randomizer(
        randomizer_node=object_instance_randomizer,
        node_name="instance_randomizer_objects",
    )

    # Add object pose randomizer
    def sample_pose(obj: Object):
        """
        Randomly samples and applies a 6-DoF pose to an object.

        The object's location is sampled uniformly within an axis-aligned box
        centered around the origin.
        """
        obj.set_location(np.random.uniform((-0.1, -0.1, 0.1), (0.1, 0.1, 0.1)))
        obj.set_rotation(np.random.uniform((-180, -180, -180), (180, 180, 180)))

    object_pose_randomizer = ObjectPoseRandomizer(
        pose_sampling_function=sample_pose, target_objects=["part_1", "part_2"]
    )

    randomizer.add_randomizer(
        randomizer_node=object_pose_randomizer, node_name="pose_randomizer"
    )

    # Add matrial randomizer
    material_randomizer = MaterialRandomizer(
        target_objects=["part_1", "part_2"], types=["metal"], context=context
    )

    randomizer.add_randomizer(
        randomizer_node=material_randomizer, node_name="material_randomizer"
    )

    # Add background randomizer
    background_randomizer = BackgroundRandomizer(
        categories=["indoor/industrial", "indoor/studio"]
    )

    randomizer.add_randomizer(
        randomizer_node=background_randomizer, node_name="background_randomizer"
    )

    # Add camera pose randomizer
    camera_pose_randomizer = CameraPoseRandomizer(
        pose_sampling_function=shell_sampler,
        number_of_views=2,
        radius_min=0.5,
        radius_max=0.7,
    )

    randomizer.add_randomizer(
        randomizer_node=camera_pose_randomizer,
        node_name="camera_pose_randomizer",
    )

    # Create the writer
    writer = CocoWriter()

    # Create the data generator with context, randomizer and writer
    data_generator = SyntheticDataGenerator(
        context=context, randomizer=randomizer, writer=writer
    )

    # Generate data
    data_generator.generate(num_images=5, save_blender_scene=False)

    # View data
    view_coco(writer.get_output_dir())


if __name__ == "__main__":
    main()

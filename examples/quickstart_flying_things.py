"""Generate a "flying things"-type dataset."""

import argparse

import numpy as np

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    BackgroundRandomizer,
    CameraPoseRandomizer,
    LightPoseRandomizer,
    LightRandomizer,
    MaterialRandomizer,
    ObjectInstanceRandomizer,
    ObjectPoseRandomizer,
)
from telekinesis.illusion.sampler.camera_pose_sampler import shell_sampler
from telekinesis.illusion.types.distribution import uniform
from telekinesis.illusion.types.object import Object
from telekinesis.illusion.utils.assets import resolve_asset_dir
from telekinesis.illusion.viewer.shard_viewer import view_coco
from telekinesis.illusion.writer.writer import CocoWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the generated dataset in the interactive viewer.",
    )
    args = parser.parse_args()

    # Create the context
    context = Context()

    assets_dir = resolve_asset_dir()

    context.add_light("key", "POINT", power=40.0, radius=0.05)
    context.add_light("sun", "SUN", power=0.2)

    # Add models to the context
    model_1_path = str(
        assets_dir / "models" / "mechanical_parts" / "gearwheel_1.glb"
    )
    context.add_model(
        model_1_path,
        object_name="part_1",
        min_number_instances=1,
        max_number_instances=3,
        uv_mapping="cube",
    )

    model_2_path = str(
        assets_dir / "models" / "mechanical_parts" / "pipe_1.glb"
    )
    context.add_model(
        model_2_path,
        object_name="part_2",
        min_number_instances=1,
        max_number_instances=1,
        uv_mapping="cube",
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

    # Randomize point-light power/radius and sun strength/angular diameter.
    randomizer.add_randomizer(
        LightRandomizer(
            ["key"],
            color=uniform((0.8, 0.85, 0.9), (1.0, 1.0, 1.0)),
            power=uniform(20.0, 60.0),
            radius=uniform(0.02, 0.08),
        ),
        node_name="point_light_properties",
    )
    randomizer.add_randomizer(
        LightRandomizer(
            ["sun"], power=uniform(0.1, 0.4), angle=uniform(0.01, 0.1)
        ),
        node_name="sun_properties",
    )
    # Sample overhead poses aimed at the parts, keeping the emitter clear.
    randomizer.add_randomizer(
        LightPoseRandomizer(
            shell_sampler,
            target_lights=["key", "sun"],
            min_distance=0.3,
            radius_min=1.0,
            radius_max=1.5,
            elevation_min=35.0,
            elevation_max=75.0,
        ),
        node_name="light_poses",
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

    if args.preview:
        view_coco(writer.get_output_dir())


if __name__ == "__main__":
    main()

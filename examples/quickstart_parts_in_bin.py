"""Generate a bin-picking dataset."""

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
from telekinesis.illusion.sampler.camera_pose_sampler import (
    shell_sampler,
    volume_sampler,
)
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

    context.add_light("softbox", "AREA", power=80.0, size=0.4)
    context.add_light("spot", "SPOT", power=20.0, radius=0.03)

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
        active_in_simulation=True,
    )

    model_2_path = str(
        assets_dir / "models" / "mechanical_parts" / "pipe_fixture_1.glb"
    )
    context.add_model(
        model_2_path,
        object_name="part_2",
        min_number_instances=0,
        max_number_instances=1,
        uv_mapping="cube",
        shading="SMOOTH",
        active_in_simulation=True,
    )

    model_3_path = str(assets_dir / "models" / "bins" / "plastic_bin_2.glb")
    context.add_model(
        model_3_path,
        object_name="crate_2",
        min_number_instances=1,
        max_number_instances=1,
        uv_mapping="cube",
        collision_shape="MESH",
    )

    context.get_objects_by_name("crate_2").set_location(
        np.array([0.0, 0.0, -0.05])
    )

    # Create the randomizer
    randomizer = Randomizer()

    # Add object instance ranodmizer
    object_instance_randomizer = ObjectInstanceRandomizer(
        target_objects=["part_1", "part_2"],
        min_num_total_objects=1,
        max_num_total_objects=4,
    )

    randomizer.add_randomizer(
        randomizer_node=object_instance_randomizer,
        node_name="instance_randomizer_objects",
    )

    # Add container instance ranodmizer
    container_instance_randomizer = ObjectInstanceRandomizer(
        target_objects=["crate_2"],
        min_num_total_objects=1,
        max_num_total_objects=1,
    )

    randomizer.add_randomizer(
        randomizer_node=container_instance_randomizer,
        node_name="instance_randomizer_containers",
    )

    # Add object pose randomizer
    def sample_pose(obj: Object):
        """
        Randomly samples and applies a 6-DoF pose to an object.

        The object's location is sampled uniformly within an axis-aligned box
        centered around the origin.
        """
        obj.set_location(
            np.random.uniform((-0.15, -0.15, 0.15), (0.15, 0.15, 0.15))
        )
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
        randomizer_node=material_randomizer,
        node_name="material_randomizer_parts",
    )

    material_randomizer = MaterialRandomizer(
        target_objects=["crate_2"], types=["plastic"], context=context
    )

    randomizer.add_randomizer(
        randomizer_node=material_randomizer,
        node_name="material_randomizer_crate",
    )

    # Add background randomizer
    background_randomizer = BackgroundRandomizer(
        categories=["indoor/industrial", "indoor/studio"]
    )

    randomizer.add_randomizer(
        randomizer_node=background_randomizer, node_name="background_randomizer"
    )

    # Sample a softbox shape/size and a separate spotlight cone.
    randomizer.add_randomizer(
        LightRandomizer(
            ["softbox"],
            color=uniform((0.85, 0.85, 0.85), (1.0, 1.0, 1.0)),
            power=uniform(40.0, 100.0),
            shape=["SQUARE", "RECTANGLE", "DISK", "ELLIPSE"],
            size=uniform(0.3, 0.6),
            size_y=uniform(0.2, 0.4),
        ),
        node_name="area_light_properties",
    )
    randomizer.add_randomizer(
        LightRandomizer(
            ["spot"],
            power=uniform(10.0, 30.0),
            radius=uniform(0.02, 0.05),
            spot_size=uniform(np.deg2rad(45), np.deg2rad(75)),
            spot_blend=uniform(0.3, 0.7),
        ),
        node_name="spot_light_properties",
    )
    # Keep both emitters above and clear of the bin and parts.
    randomizer.add_randomizer(
        LightPoseRandomizer(
            shell_sampler,
            target_lights=["softbox", "spot"],
            center=["crate_2"],
            min_distance=0.3,
            radius_min=1.2,
            radius_max=1.6,
            elevation_min=50.0,
            elevation_max=80.0,
        ),
        node_name="light_poses",
    )

    # Add camera pose randomizer
    camera_pose_randomizer = CameraPoseRandomizer(
        pose_sampling_function=volume_sampler, number_of_views=2
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
    data_generator.generate(
        num_images=5, simulate_physics=True, save_blender_scene=False
    )

    if args.preview:
        view_coco(writer.get_output_dir())


if __name__ == "__main__":
    main()

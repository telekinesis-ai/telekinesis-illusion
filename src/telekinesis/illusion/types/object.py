from loguru import logger
import numpy as np
from typing import List, Union

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

import bpy
from mathutils import Vector, Matrix
import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject

from telekinesis.illusion.loader.object_loader import load_obj


class Object:
    """
    Every instance of this class is a mesh which can be rendered in the scene.
    It can have multiple materials and different configurations of vertices with
    faces and edges. This class is a wrapper around BlenderProc's MeshObject
    class.
    """

    def __init__(
        self,
        object_name: str,
        category_name: str,
        category_id: int | None = 0,
        model_path: str | None = None,
        object: MeshObject | None = None,
        active_in_simulation: bool = False,
        collision_shape: str = "CONVEX_HULL",
        scale: float | np.ndarray = 1.0,
        preprocess_model: bool = True,
        min_number_instances: int = 1,
        max_number_instances: int = 1,
    ):
        """
        Initialize the internal parameters.

        Args:
            object_name: str
                Unique name for the object within the scene.
            category_name: str or None
                Name of the category for annotation. Required for the
                LabelIdMapping in the context. Not specifcally used in this
                class, rather an attribute.
            category_id: int or None
                Optional explicit category ID. 'None' indicates a distractor.
            model_path: str
                Path to the 3D model file on disk. If 'None', a MeshObject needs
                to be provided
            object: MeshObject
                MeshObject to be assigned to the Object. If None, it attempts
            active_in_simulation: bool
                Whether this object activly participates in the physics
                simulation.
            collision_shape: str
                Collision shape of the object for collision checks and physics
                simulation. Defaults to "CONVEX_HULL". Available collision
                shapes: "BOX", "SPHERE", "CAPSULE", "CYLINDER", "CONE",
                "CONVEX_HULL", "MESH"
            scale: float or np.ndarray
                Scale factor when adding the model. If a float is provided, the
                object is scaled uniformaly along its x, y and z axis with the
                provided value. If a np.ndarray is provided, the object is
                scaled with (scale_x, scale_y, scale_z).
            preprocess_model: bool
                Whether to preprocess the model upon loading.
                NOTE: All the preprocessed objects are made to rigid bodies.
            min_number_instances: int
                Minimum number of instances for objects of this type.
            max_number_instances: int
                Maximum number of instances for objects of this type.
        """
        self._object = None
        self._category_name = category_name
        self._model_path = model_path  # Store original model path for reference
        self._original_blender_name = None
        self._object_name = None
        self._preprocess_model = preprocess_model
        self._instance_number = 0
        self._active_in_simulation = active_in_simulation
        self._collision_shape = collision_shape
        self._scale = scale
        self._min_number_instances = min_number_instances
        self._max_number_instances = max_number_instances

        # If model_path is provided, load the model. Else assign the provided
        # MeshObject
        if model_path:
            self.load_model(model_path, object_name, preprocess_model)
            # Scaling is only applied when the model is loaded
            if np.any(np.asarray(scale) != 1.0):
                if isinstance(scale, float):
                    self._scale_vector = Vector((scale, scale, scale))
                elif isinstance(scale, np.ndarray):
                    self._scale_vector = Vector(scale)
                else:
                    raise ValueError(
                        "Provided scale needs to be either a float or np.ndarray."
                    )
                self._object.blender_obj.scale = self._scale_vector

            # Apply transform so the scale is at 1.0
            obj = self._object.blender_obj
            # Bake scale into mesh vertices directly
            scale_matrix = Matrix.Diagonal(obj.scale).to_4x4()
            obj.data.transform(scale_matrix)
            obj.scale = (1.0, 1.0, 1.0)

            # Clear custom split normals and shade flat
            if self._preprocess_model:
                # Deselct everything
                bpy.ops.object.select_all(action="DESELECT")
                # Select object
                bpy.data.objects[self._object_name].select_set(True)
                # Remove custom normals
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
                # Shade flat
                for poly in bpy.context.object.data.polygons:
                    poly.use_smooth = False
                # Deselct everything
                bpy.ops.object.select_all(action="DESELECT")

        elif object:
            # This method is only used for creating linked duplicates
            self._object = object
            self.set_name(object_name=object_name)
            if self._preprocess_model:
                logger.info("Preprocessing model...")
                # self._object.add_uv_mapping('smart', overwrite=True)
                self.hide(True)
                # self.enable_rigid_body()
        else:
            raise ValueError(
                "Provide either a path to the model or a MeshObject."
            )

        self._category_id = category_id
        if category_id:
            self.set_category_id(category_id)

    def __repr__(self):
        representation = (
            f"<{self.__class__} '{self.get_name()}' at {hex(id(self))}>"
        )
        return representation

    @property
    def min_number_instances(self) -> int:
        return self._min_number_instances

    @property
    def max_number_instances(self) -> int:
        return self._max_number_instances

    def get_object(self) -> MeshObject:
        return self._object

    def get_name(self) -> str:
        return self._object_name

    def get_scale(self) -> float | np.ndarray:
        """
        The scale this object's mesh was baked at when it was loaded.

        __init__ applies the scale to the mesh vertices and resets the object
        transform to 1, so this is the only record of it - a caller wanting to
        re-scale later needs it to work out the relative factor.
        """
        return self._scale

    def set_scale(self, scale: float | np.ndarray) -> None:
        """
        Record a new baked scale.

        Bookkeeping only - this does NOT transform the mesh. Callers that
        re-scale the mesh data themselves use this to keep the baseline in
        step (see BinPickingWorker._apply_model_scales).

        Args:
            scale: float or np.ndarray
                The scale the mesh is now baked at.
        """
        self._scale = scale

    def set_name(self, object_name: str) -> None:
        self._object.set_name(object_name)
        self._object_name = object_name

    def get_category_name(self) -> str:
        """Returns the category name (semantic name for annotations)."""
        return self._object.get_name()

    def get_model_path(self) -> str:
        """Returns the original model file path."""
        return self._model_path

    def set_category_name(self, category_name: str | None = None) -> None:
        """Set the category name and update both the Blender object name and internal tracking."""
        self._object.set_name(category_name)

    def set_category_id(self, category_id: int) -> None:
        """
        Set category ID for annotation

        Args:
        category_id: int
            Category ID for the object.

        Returns:
            None

        Raises:
            None
        """
        self._object.set_cp("category_id", category_id)
        self._category_id = category_id

    def get_category_id(self) -> int:
        """
        Return the set category id.

        Returns:
            Either the provided category id or 'None', which indicates a
            distractor.
        """
        if self._category_id:
            return self._object.get_cp("category_id")
        return self._category_id

    def get_location(self) -> np.ndarray:
        """
        Returns location of the object.

        Returns:
            Returns the location of the object as a np.ndarray.
        """
        return np.array(self._object.blender_obj.location)

    def set_location(self, location: np.ndarray) -> None:
        """
        Sets location of the object.

        Args:
            location: np.ndarray
                The location to set as a 3D XYZ-array.
        """

        self._object.blender_obj.location = Vector(location)

    def get_rotation(self) -> np.ndarray:
        """
        Returns the rotation of the obect as XYZ euler angles.
        TODO: Extend this to also optionally return the rotation matrix.

        Returns:
            A np.ndarray as the rotation of the object in XYZ euler angles.
        """
        self._object.blender_obj.rotation_mode = "XYZ"
        return np.array(self._object.blender_obj.rotation_euler)

    def set_rotation(self, rotation: np.ndarray) -> None:
        """
        Sets the rotation of the object in XYZ euler angles.
        TODO: Extend this to also accepts rotation matrices.

        Args:
            rotation: np.ndarray
                Rotation in XYS euler angles.
        """
        self._object.blender_obj.rotation_mode = "XYZ"
        self._object.blender_obj.rotation_euler = rotation

    def load_model(
        self, model_path: str, object_name: str, preprocess: bool = True
    ) -> None:
        """
        Loads the model into the context from the provided 'model_path' and ads
        it to the objects dictionary with the 'object_name' as the prefix.
        All the models are added with the name
        'object_name'_INSTANCE_'self._instance_number', where
        'self._instance_number' starts from 0 and is incremented up to the
        number of instances.
        The optional preprocessing includes the autmoatic smart UV-map addings
        and enables rigid_body physics for collision checking with a
        'CONVEX_HULL' collision shape.

        Args:
            model_path: str
                Path to the .glb model.
            object_name: str
                Prefix for the object_name.
            preprocess: bool
                Wheter to preprocess the added object.
        """
        self._object = load_obj(model_path)[0]

        # Store original Blender object name for reference (before category name is set)
        self._original_blender_name = self._object.blender_obj.name_full

        # Update the name
        object_name_suffix = "_INSTANCE_" + str(
            self._instance_number
        )  # Add always the suffix
        self._instance_number += 1
        new_object_name = object_name + object_name_suffix
        self.set_name(object_name=new_object_name)

        # Preprocessing for data generation
        if preprocess:
            logger.info("Preprocessing model...")
            uv_layer_count = len(self._object.blender_obj.data.uv_layers)
            if uv_layer_count <= 1:
                self._object.add_uv_mapping("smart", overwrite=True)
            else:
                logger.info(
                    f"Model already has {uv_layer_count} UV layers — "
                    "skipping smart projection and using existing UVs."
                )
            self.hide(True)
            # self.enable_rigid_body()
            # Remove all materials and add a dummy material
            logger.info("Clearing all material slots...")
            self._object.blender_obj.data.materials.clear()

            logger.info("Adding dummy material...")
            dummy_mat = bpy.data.materials.new(name="DummyMaterial")
            self._object.blender_obj.data.materials.append(dummy_mat)

    def enable_rigid_body(self) -> None:
        """
        Enable rigid body properties of the object to participate in collision
        checking and physics simulation.
        """
        self._object.enable_rigidbody(
            active=self._active_in_simulation,
            collision_shape=self._collision_shape,
        )

    def disable_rigid_body(self) -> None:
        """
        Disable rigid body properties of the object to not participate in
        collision checking and physics simulation.
        """
        if self._object.has_rigidbody_enabled():
            self._object.disable_rigidbody()

    def get_bound_box(self, local_coords: bool = False) -> np.ndarray:
        """
        Returns a 8x3 array describing the object aligned bounding box
        coordinates in world coordinates
        """
        return self._object.get_bound_box(local_coords)

    def get_local2world_mat(self) -> np.ndarray:
        """
        Returns the pose of the object in the form of a 4x4 local2world matrix.
        """
        return self._object.get_local2world_mat()

    def set_local2world_mat(self, matrix_world: Union[np.ndarray, Matrix]):
        """Sets the pose of the object in the form of a local2world matrix.

        :param matrix_world: A 4x4 matrix.
        """
        return self._object.set_local2world_mat(matrix_world)

    def hide(self, hide_object: bool = True):
        """
        Sets the visibility of the object.

        Args:
            hide_object: bool
                Determines whether the object should be hidden in rendering.
        """
        self._object.hide(hide_object)

    def is_hidden(self) -> bool:
        """
        Returns whether the object is hidden in rendering.

        Return:
            True, if it is hidden.
        """
        return self._object.is_hidden()

    def replace_material(self, material: bpy.types.Material) -> None:
        """
        Replaces all materials of the object with the given new material.

        Args:
            material: bpy.types.Material
                A material that should exclusively be used as new material for
                the object.
        """

        self._object.replace_materials(material)

    def create_linked_duplicate(
        self,
    ) -> "Object":
        """
        Creates a linked duplicate of this Object instance. The geometry data is
        shared, but the material and transformation can be modified independently.

        Returns:
            The newly created Object instance.
        """
        new_obj = self._object.blender_obj.copy()
        bpy.context.collection.objects.link(new_obj)

        bpy.context.view_layer.objects.active = new_obj
        bpy.context.object.material_slots[0].link = "OBJECT"
        bpy.context.view_layer.objects.active = self._object.blender_obj

        # Convert to bproc subclass
        new_entity = MeshObject(new_obj)

        # Convert to illusion subclass
        base, _ = self._object_name.rsplit("_", 1)
        new_obj_name = f"{base}_{self._instance_number}"
        self._instance_number += 1

        new_illusion_obj = Object(
            object_name=new_obj_name,
            category_name=self.get_category_name(),
            category_id=self.get_category_id(),
            active_in_simulation=self._active_in_simulation,
            preprocess_model=self._preprocess_model,
            object=new_entity,
            collision_shape=self._collision_shape,
            scale=self.get_scale(),
            min_number_instances=self._min_number_instances,
            max_number_instances=self._max_number_instances,
        )

        return new_illusion_obj


def compute_poi(objects: List[Object]) -> np.ndarray:
    """
    Computes a point of interest in the scene. Point is defined as a location of
    the one of the selected objects that is the closest one to the mean location
    of the bboxes of the selected objects.

    Args:
        objects: List[Object]
            The list of mesh objects that should be considered.

    Returns:
        A np.ndarray as the point of interest in the scene.
    """
    # Init matrix for all points of all bounding boxes
    mean_bb_points = []

    for obj in objects:
        # Get MeshObject
        mesh_obj = obj.get_object()
        # Get bounding box corners
        bb_points = mesh_obj.get_bound_box()
        # Compute mean coords of bounding box
        mean_bb_points.append(np.mean(bb_points, axis=0))
    # Query point - mean of means
    mean_bb_point = np.mean(mean_bb_points, axis=0)
    # Closest point (from means) to query point (mean of means)
    poi = mean_bb_points[
        np.argmin(np.linalg.norm(mean_bb_points - mean_bb_point, axis=1))
    ]

    return poi

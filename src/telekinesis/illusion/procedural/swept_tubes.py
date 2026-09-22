"""Reusable geometry for circular tubes swept along Bezier paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
from mathutils import Vector
from blenderproc.python.types.MeshObjectUtility import MeshObject

from telekinesis.illusion.core.context import Context


@dataclass(frozen=True)
class SweptTubeSettings:
    """Mesh resolution shared by a group of swept tubes."""

    curve_resolution_u: int = 4
    bevel_resolution: int = 2


class SweptTubeBuilder:
    """Build, materialize and optionally annotate swept Bezier tubes."""

    def __init__(
        self,
        context: Context,
        collection_name: str,
        settings: SweptTubeSettings | None = None,
    ) -> None:
        self.context = context
        self.settings = settings or SweptTubeSettings()
        self.collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(self.collection)

    @staticmethod
    def material(
        name: str,
        color: tuple[float, float, float],
        metallic: float = 0.0,
        roughness: float = 0.3,
    ) -> bpy.types.Material:
        """Create or update a Principled material."""
        material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        material.diffuse_color = (*color, 1.0)
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = (*color, 1.0)
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Roughness"].default_value = roughness
        return material

    @staticmethod
    def _as_mesh(obj: bpy.types.Object) -> bpy.types.Object:
        if obj.type == "MESH":
            return obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target="MESH")
        return bpy.context.object

    def add_tube(
        self,
        name: str,
        points_mm: Sequence[Sequence[float]],
        radius_mm: float,
        material: bpy.types.Material,
        *,
        cyclic: bool = False,
        category_name: str | None = None,
        category_id: int | None = None,
    ) -> bpy.types.Object:
        """Sweep a circular profile along a Bezier path and create a mesh."""
        if len(points_mm) < 2:
            raise ValueError("A swept tube requires at least two path points.")

        data = bpy.data.curves.new(name + "_curve", "CURVE")
        data.dimensions = "3D"
        data.resolution_u = self.settings.curve_resolution_u
        data.bevel_depth = radius_mm * 0.001
        data.bevel_resolution = self.settings.bevel_resolution
        data.use_fill_caps = True
        spline = data.splines.new("BEZIER")
        spline.bezier_points.add(len(points_mm) - 1)
        for point, xyz in zip(spline.bezier_points, points_mm):
            point.co = Vector(xyz) * 0.001
            point.handle_left_type = "AUTO"
            point.handle_right_type = "AUTO"
        spline.use_cyclic_u = cyclic

        obj = bpy.data.objects.new(name, data)
        self.collection.objects.link(obj)
        data.materials.append(material)
        obj = self._as_mesh(obj)

        if category_name is not None or category_id is not None:
            self.context.add_mesh_object(
                MeshObject(obj),
                object_name=name,
                category_name=category_name,
                category_id=category_id,
            )
        return obj

    def add_cylinder_endpoint(
        self,
        name: str,
        endpoint_mm: Sequence[float],
        direction: Sequence[float],
        radius_mm: float,
        length_mm: float,
        material: bpy.types.Material,
        *,
        category_name: str | None = None,
        category_id: int | None = None,
        annotation_parent: str | None = None,
    ) -> bpy.types.Object:
        """Place a cylindrical mesh along the tangent of a path endpoint."""
        tangent = Vector(direction).normalized()
        endpoint = Vector(endpoint_mm) * 0.001
        length = length_mm * 0.001
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16,
            radius=radius_mm * 0.001,
            depth=length,
            location=endpoint + tangent * (length / 2.0 - 0.0002),
        )
        obj = bpy.context.object
        obj.name = name
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        self.collection.objects.link(obj)
        obj.rotation_euler = tangent.to_track_quat("Z", "Y").to_euler()
        obj.data.materials.append(material)
        bevel = obj.modifiers.new("Soft endpoint edges", "BEVEL")
        bevel.width = 0.00010
        bevel.segments = 2
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

        if annotation_parent:
            obj["annotation_parent"] = annotation_parent
        if category_name is not None or category_id is not None:
            self.context.add_mesh_object(
                MeshObject(obj),
                object_name=name,
                category_name=category_name,
                category_id=category_id,
            )
        return obj

    def add_split_tube_with_endpoint(
        self,
        name: str,
        points_mm: Sequence[Sequence[float]],
        annotated_start_index: int,
        radius_mm: float,
        material: bpy.types.Material,
        category_name: str,
        category_id: int,
        *,
        endpoint_name: str,
        endpoint_radius_mm: float,
        endpoint_length_mm: float,
        endpoint_material: bpy.types.Material,
        endpoint_category_name: str,
        endpoint_category_id: int,
        unannotated_prefix: str = "Unannotated_",
    ) -> tuple[bpy.types.Object, bpy.types.Object]:
        """Split a path into visible context and an annotated terminal section."""
        split_index = annotated_start_index - 1
        if annotated_start_index < 2 or annotated_start_index >= len(points_mm):
            raise ValueError("annotated_start_index must split the tube path.")

        self.add_tube(
            unannotated_prefix + name,
            points_mm[: split_index + 1],
            radius_mm,
            material,
        )
        tube = self.add_tube(
            name,
            points_mm[split_index:],
            radius_mm,
            material,
            category_name=category_name,
            category_id=category_id,
        )
        direction = Vector(points_mm[-1]) - Vector(points_mm[-2])
        endpoint = self.add_cylinder_endpoint(
            endpoint_name,
            points_mm[-1],
            direction,
            endpoint_radius_mm,
            endpoint_length_mm,
            endpoint_material,
            category_name=endpoint_category_name,
            category_id=endpoint_category_id,
            annotation_parent=tube.name,
        )
        return tube, endpoint


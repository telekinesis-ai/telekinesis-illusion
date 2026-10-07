"""Material targeting, source adapters and assignment for Blender scenes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
from blenderproc.python.material.MaterialLoaderUtility import (
    create,
)
from blenderproc.python.types.MaterialUtility import Material

from telekinesis.illusion.randomizer.surface_imperfections import (
    apply_imperfections,
    clean_imperfection_images,
    discover_imperfections,
)
from telekinesis.illusion.types.material import (
    PARAMETERS,
    PrincipledMaterial,
)

if TYPE_CHECKING:
    from telekinesis.illusion.core.context import Context, MaterialManager
    from telekinesis.illusion.types.object import Object


def _names(values, label: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of names.")
    result = tuple(values)
    if any(not isinstance(n, str) or not n for n in result):
        raise ValueError(f"{label} must contain non-empty names.")
    return result


@dataclass(frozen=True)
class MaterialTarget:
    """Exact registered object names; None selects all registered objects."""

    objects: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.objects is not None:
            names = _names(self.objects, "objects")
            if not names:
                raise ValueError("objects cannot be empty; use None for all.")
            if len(set(names)) != len(names):
                raise ValueError("objects contains duplicate names.")
            object.__setattr__(self, "objects", names)

    def resolve(self, context: "Context") -> list["Object"]:
        available = context.get_objects()
        names = sorted(available) if self.objects is None else self.objects
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"Missing material target objects: {missing}.")
        return [available[name] for name in names]


class SlotSelector(Protocol):
    """Implement resolve() to add further slot-selection strategies."""

    def resolve(self, obj: "Object") -> tuple[int, ...]: ...


@dataclass(frozen=True)
class AllSlots:
    def resolve(self, obj: "Object") -> tuple[int, ...]:
        return tuple(range(len(obj.get_material_slot_names())))


@dataclass(frozen=True)
class NamedSlots:
    """Resolve original names or zero-based indices; ambiguity is an error."""

    slots: tuple[str | int, ...]

    def __post_init__(self):
        if isinstance(self.slots, (str, bytes)) or not self.slots:
            raise ValueError("slots must be a non-empty sequence.")
        slots = tuple(self.slots)
        if any(
            isinstance(s, bool)
            or not isinstance(s, (str, int))
            or (isinstance(s, int) and s < 0)
            or (isinstance(s, str) and not s)
            for s in slots
        ):
            raise ValueError("Slots must be non-empty names or indices >= 0.")
        object.__setattr__(self, "slots", slots)

    def resolve(self, obj: "Object") -> tuple[int, ...]:
        names = obj.get_material_slot_names()
        result = []
        for slot in self.slots:
            matches = (
                [slot]
                if isinstance(slot, int) and slot < len(names)
                else [i for i, name in enumerate(names) if name == slot]
            )
            if len(matches) != 1:
                reason = "missing" if not matches else "ambiguous"
                raise ValueError(
                    f"Material slot {slot!r} is {reason} on {obj.get_name()!r}; "
                    f"original slots: {names}. Use an index for duplicates."
                )
            result.extend(matches)
        if len(set(result)) != len(result):
            raise ValueError(
                "Multiple slot selectors resolve to the same slot."
            )
        return tuple(result)


class MaterialSource(Protocol):
    """A source returns a BlenderProc material without assigning it."""

    def prepare(self, manager: "MaterialManager") -> None: ...

    def generate(self, manager: "MaterialManager", rng) -> Material: ...


@dataclass(frozen=True)
class PBRMaterial:
    """Choose a texture material from the existing MaterialManager catalog."""

    types: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.types is not None:
            object.__setattr__(self, "types", _names(self.types, "PBR types"))

    def prepare(self, manager: "MaterialManager") -> None:
        manager.update_materials(self.types)

    def generate(self, manager: "MaterialManager", rng) -> Material:
        return manager.get_random_material(self.types, rng=rng)


@dataclass(frozen=True)
class _PrincipledSource:
    recipe: PrincipledMaterial
    _imperfection_maps: tuple = field(default=(), init=False, repr=False)

    def prepare(self, manager):
        settings = self.recipe.surface_imperfections
        if settings.enabled:
            directory = manager.get_asset_dir() / settings.directory
            object.__setattr__(
                self,
                "_imperfection_maps",
                discover_imperfections(directory.resolve()),
            )

    def generate(self, manager, rng) -> Material:
        values = self.recipe.sample(rng)
        material = create(f"Illusion_{self.recipe.preset.name}")
        try:
            shader = material.get_the_one_node_with_type("BsdfPrincipled")
            for name, value in values.items():
                shader.inputs[PARAMETERS[name][0]].default_value = value
            if self.recipe.surface_imperfections.enabled:
                apply_imperfections(
                    material.blender_obj,
                    shader,
                    self.recipe.surface_imperfections,
                    self._imperfection_maps,
                    rng,
                )
            # Only materials owned by this generator may be reclaimed.
            material.blender_obj["illusion_generated_material"] = True
            return material
        except Exception:
            bpy.data.materials.remove(material.blender_obj)
            clean_imperfection_images()
            raise


@dataclass(frozen=True)
class _ExistingSource:
    material: Material

    def prepare(self, manager):
        pass

    def generate(self, manager, rng) -> Material:
        return self.material


MaterialChoice: TypeAlias = str | PrincipledMaterial | Material | MaterialSource
MaterialSlots: TypeAlias = (
    Mapping[str | int, Sequence[MaterialChoice]] | Literal["all"]
)
MaterialMode: TypeAlias = Literal["active", "object", "slots"]


def material_choices_from_config(choices):
    """Parse YAML/JSON choices without changing string PBR-tag semantics."""
    if not isinstance(choices, (tuple, list)):
        raise TypeError("Material choices must be a list.")
    result = []
    for choice in choices:
        if isinstance(choice, Mapping):
            unknown = set(choice) - {
                "preset",
                "parameters",
                "surface_imperfections",
            }
            if unknown:
                raise ValueError(
                    f"Unknown Principled configuration keys: {unknown}."
                )
            parameters = choice.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise TypeError("Principled parameters must be a mapping.")
            choice = PrincipledMaterial(
                choice.get("preset"),
                surface_imperfections=choice.get(
                    "surface_imperfections", False
                ),
                **parameters,
            )
        result.append(choice)
    return result


class MaterialPool:
    """Select a source, then generate or fetch a material through one path."""

    def __init__(self, choices: Sequence[MaterialChoice]):
        if isinstance(choices, (str, bytes)) or not isinstance(
            choices, Sequence
        ):
            raise TypeError("materials must be a non-empty sequence.")
        if not choices:
            raise ValueError("materials must be a non-empty sequence.")
        self.sources = tuple(self._source(choice) for choice in choices)

    @staticmethod
    def _source(choice) -> MaterialSource:
        if isinstance(choice, str):
            return PBRMaterial((choice,))
        if isinstance(choice, PrincipledMaterial):
            return _PrincipledSource(choice)
        if isinstance(choice, Material):
            return _ExistingSource(choice)
        if callable(getattr(choice, "prepare", None)) and callable(
            getattr(choice, "generate", None)
        ):
            return choice
        raise ValueError(f"Unsupported material source: {choice!r}.")

    def prepare(self, manager):
        for source in self.sources:
            source.prepare(manager)

    def generate(self, manager, rng) -> Material:
        # A singleton consumes no extra random draw, preserving legacy PBR RNG.
        source = (
            self.sources[0]
            if len(self.sources) == 1
            else rng.choice(self.sources)
        )
        return source.generate(manager, rng)


def slot_pools(material_slots) -> list[tuple[SlotSelector, MaterialPool]]:
    if not isinstance(material_slots, Mapping) or not material_slots:
        raise ValueError("material_slots must be 'all' or a non-empty mapping.")
    return [
        (NamedSlots((slot,)), MaterialPool(choices))
        for slot, choices in material_slots.items()
    ]


def assign_material(
    obj: "Object", indices: tuple[int, ...], material: Material
):
    """Override object slots without editing shared mesh data or shaders."""
    if material.blender_obj.get("illusion_imperfection_map"):
        obj.ensure_uv_mapping()
    blender_obj = obj.get_object().blender_obj
    if not blender_obj.material_slots:
        # Adding a slot changes mesh data, so make only this empty mesh private.
        if blender_obj.data.users > 1:
            blender_obj.data = blender_obj.data.copy()
        # Leave the mesh's data slot empty; keeping a generated material here
        # would pin the first sample even after its object override changed.
        blender_obj.data.materials.append(None)
        obj.refresh_material_slot_names()
    for index in indices:
        slot = blender_obj.material_slots[index]
        slot.link = "OBJECT"
        slot.material = material.blender_obj


def clean_generated_materials():
    """Reclaim only unused procedural materials, never imported/PBR materials."""
    for material in list(bpy.data.materials):
        if material.get("illusion_generated_material") and material.users == 0:
            bpy.data.materials.remove(material)
    clean_imperfection_images()

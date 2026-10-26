from telekinesis.illusion.core.model_manager import ModelManager
from telekinesis.illusion.types.model_asset import ModelAsset


manager = ModelManager()
asset = manager.add(ModelAsset(
    "assets/models/mechanical_parts/gearwheel_1.glb", "gear")
)
print()

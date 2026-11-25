from telekinesis.illusion import config
from telekinesis.illusion.core.model_manager import ModelManager
from telekinesis.illusion.utils import assets


manager = ModelManager()
asset = manager.add(config.ModelConfig(
    assets.resolve_asset_dir()
    / "models/parts/mechanical_parts/gearwheels/gearwheel_1.glb", "gear")
)
print()

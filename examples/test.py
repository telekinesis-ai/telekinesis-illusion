from telekinesis.illusion import config
from telekinesis.illusion.core.model_manager import ModelManager


manager = ModelManager()
asset = manager.add(config.ModelConfig(
    "assets/models/mechanical_parts/gearwheel_1.glb", "gear")
)
print()

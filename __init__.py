# Import specific classes from the Qwen Tuner module
from .qwen_te_arthemy_tuner import (
    ArthemyQwenTunerSimple,
    ArthemyQwenTunerLab,
    ArthemyQwenSaver
)

# Import specific classes from the Z-Image Tuner module
from .z_image_arthemy_tuner import (
    ArthemyTunerLoader,
    ArthemyZImage_Tuner_Simple,
    ArthemyZImage_Tuner_Lab,
    ArthemyZImage_Saver
)

# Main dictionary mapping internal node names to their Python classes
NODE_CLASS_MAPPINGS = {
    "ArthemyQwenTunerSimple": ArthemyQwenTunerSimple,
    "ArthemyQwenTunerLab": ArthemyQwenTunerLab,
    "ArthemyQwenSaver": ArthemyQwenSaver,
    "ArthemyTunerLoader": ArthemyTunerLoader,
    "ArthemyZImage_Tuner_Simple": ArthemyZImage_Tuner_Simple,
    "ArthemyZImage_Tuner_Lab": ArthemyZImage_Tuner_Lab,
    "ArthemyZImage_Saver": ArthemyZImage_Saver
}

# Optional dictionary for custom display names in the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArthemyQwenTunerSimple": "Arthemy Qwen Tuner (Simple)",
    "ArthemyQwenTunerLab": "Arthemy Qwen Tuner (LAB)",
    "ArthemyQwenSaver": "Arthemy Qwen Saver",
    "ArthemyTunerLoader": "Arthemy Tuner Model Loader",
    "ArthemyZImage_Tuner_Simple": "Arthemy Z-Image Tuner (Simple)",
    "ArthemyZImage_Tuner_Lab": "Arthemy Z-Image Tuner (LAB)",
    "ArthemyZImage_Saver": "Arthemy Z-Image Saver"
}

# Expose the mappings to ComfyUI's main loading mechanism
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

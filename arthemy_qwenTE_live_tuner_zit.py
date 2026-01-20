import torch
import re

class ArthemyQwenLiveTunerZImage:
    """
    Arthemy Qwen Tuner (Z-Image)
    
    This node modulates the weights of the Qwen3-4B Text Encoder used by Z-Image.
    It divides the 36 transformer layers into four semantic bands, enabling precise
    control over how the LLM processes syntax, literal meaning, context, and 
    abstract concepts before passing embeddings to the diffusion model.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",), # Z-Image Qwen is loaded within the standard CLIP wrapper
                "mode": (["Soft Value", "Real Value"],),
                "base_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Global multiplier applied to all 36 layers."}),
                
                # Band 1: Layers 0-8
                "LLM_Syntax_Parsing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls grammar strictness and tokenization (Layers 0-8)."}),
                
                # Band 2: Layers 9-17
                "LLM_Literal_Meaning": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls direct object definitions (Layers 9-17)."}),
                
                # Band 3: Layers 18-26
                "LLM_Contextual_Web": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls relationships between subjects and environment (Layers 18-26)."}),
                
                # Band 4: Layers 27-35
                "LLM_Abstract_Concept": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls high-level conceptual interpretation (Layers 27-35)."}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING", )
    RETURN_NAMES = ("CLIP", "info", )
    FUNCTION = "tune_qwen"
    CATEGORY = "Arthemy/Z-Image"

    def tune_qwen(self, clip, mode, base_scale, **kwargs):
        
        def get_target_weight(w):
            if mode == "Real Value": return w
            # Soft Value: Conservative scaling range for LLMs (0.8 to 1.2)
            # to prevent hidden state collapse.
            return 0.8 + (0.2 * w) 

        # Qwen3-4B typically consists of 36 transformer layers.
        TOTAL_LAYERS_QWEN = 36
        
        # Map slider names to layer ranges
        bands = [
            ("LLM_Syntax_Parsing", range(0, 9)),
            ("LLM_Literal_Meaning", range(9, 18)),
            ("LLM_Contextual_Web", range(18, 27)),
            ("LLM_Abstract_Concept", range(27, 36))
        ]
        
        # Build the weight map
        layer_map = {}
        w_base = get_target_weight(base_scale)
        
        for name, rng in bands:
            val = get_target_weight(kwargs.get(name, 1.0))
            for i in rng:
                layer_map[i] = val * w_base

        clip_out = clip.clone()
        
        # Retrieve internal model patches/keys.
        kp = clip_out.get_key_patches()
        
        active_patches = 0

        for key in kp:
            # Exclude normalization layers and biases to maintain stability
            if "norm" in key or "bias" in key:
                continue
                
            # Regex to identify layer index. 
            # Matches standard patterns like "model.layers.5..." or "h.5..."
            match = re.search(r"\.layers\.(\d+)\.", key) or re.search(r"\.h\.(\d+)\.", key)
            
            if match:
                idx = int(match.group(1))
                if idx in layer_map:
                    target_scale = layer_map[idx]
                    strength = target_scale - 1.0
                    
                    if strength != 0:
                        clip_out.add_patches({key: kp[key]}, strength, 1.0)
                        active_patches += 1

        info = f"Qwen Tuned | Active Patches: {active_patches} | Mode: {mode}"
        return (clip_out, info, )

NODE_CLASS_MAPPINGS = {"ArthemyQwenLiveTunerZImage": ArthemyQwenLiveTunerZImage}
NODE_DISPLAY_NAME_MAPPINGS = {"ArthemyQwenLiveTunerZImage": "✨ Arthemy Qwen Tuner (Z-Image)"}
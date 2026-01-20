import torch
import re

class ArthemyQwenLiveTunerZImageLab:
    """
    ⚗️ Arthemy Qwen Tuner (Z-Image Lab Version)
    
    Architecture: Qwen3-4B Transformer (36 Layers)
    Technique: Non-destructive Patching via comfy.sd.
    
    LAB VERSION:
    This node exposes all 36 layers of the Qwen Text Encoder individually.
    It retains the semantic naming convention (Syntax, Literal, Context, Abstract) 
    to map the LLM's processing depth to specific transformer blocks for factual verification.
    """
    
    # Layer Grouping Logic (36 Layers Total)
    # Syntax: 0-8
    # Literal: 9-17
    # Context: 18-26
    # Abstract: 27-35
    
    LAYER_CONFIG = []
    
    # Generate ordered configuration for UI generation
    for i in range(36):
        if i < 9:
            prefix = "LLM_Syntax_Parsing"
        elif i < 18:
            prefix = "LLM_Literal_Meaning"
        elif i < 27:
            prefix = "LLM_Contextual_Web"
        else:
            prefix = "LLM_Abstract_Concept"
            
        key = f"{prefix}_L{i:02d}"
        LAYER_CONFIG.append({"index": i, "name": key})

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP",), # Z-Image Qwen is loaded within the standard CLIP wrapper
                "mode": (["Soft Value", "Real Value"],),
                "base_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Global multiplier applied to all layers."}),
            },
            "optional": {}
        }
        
        # Dynamically add all 36 sliders
        for layer in s.LAYER_CONFIG:
            inputs["optional"][layer["name"]] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
            
        return inputs

    RETURN_TYPES = ("CLIP", "STRING", )
    RETURN_NAMES = ("CLIP", "info", )
    FUNCTION = "tune_qwen_lab"
    CATEGORY = "Arthemy/Z-Image"

    def tune_qwen_lab(self, clip, mode, base_scale, **kwargs):
        
        def get_target_weight(w):
            if mode == "Real Value": return w
            # Soft Value: Conservative scaling range for LLMs (0.8 to 1.2)
            # LLMs are very sensitive to weight variance.
            return 0.8 + (0.2 * w)

        w_base = get_target_weight(base_scale)
        
        # Create fast lookup map for the sliders provided in kwargs
        layer_scales = {}
        for layer in self.LAYER_CONFIG:
            idx = layer["index"]
            name = layer["name"]
            val = kwargs.get(name, 1.0)
            layer_scales[idx] = get_target_weight(val)

        clip_out = clip.clone()
        
        # Retrieve internal model patches/keys.
        kp = clip_out.get_key_patches()
        
        active_patches = 0
        modulations = 0

        for key in kp:
            # Exclude normalization layers and biases to maintain stability
            # Modifying these in LLMs often leads to immediate token collapse (gibberish).
            if "norm" in key or "bias" in key:
                continue

            target_scale = w_base
                
            # Regex to identify layer index. 
            # Matches standard patterns like "model.layers.5..." or "h.5..." depending on library version
            match = re.search(r"\.layers\.(\d+)\.", key) or re.search(r"\.h\.(\d+)\.", key)
            
            if match:
                idx = int(match.group(1))
                if idx in layer_scales:
                    # Apply specific layer scale * base scale
                    target_scale = layer_scales[idx] * w_base
                    modulations += 1

            # Calculate strength for patcher
            # Final Weight = Original Weight + (Original Weight * strength)
            strength = target_scale - 1.0
            
            if strength != 0:
                clip_out.add_patches({key: kp[key]}, strength, 1.0)
                active_patches += 1

        info = f"Qwen Lab Active | Mode: {mode} | Patches: {active_patches} | Layers Mod: {modulations // 3}"
        # Divided by approx 3 (Q, K, V, MLP, etc) to estimate blocks
        return (clip_out, info, )

NODE_CLASS_MAPPINGS = {"ArthemyQwenLiveTunerZImageLab": ArthemyQwenLiveTunerZImageLab}

NODE_DISPLAY_NAME_MAPPINGS = {"ArthemyQwenLiveTunerZImageLab": "✨ Arthemy Qwen Tuner (Z-Image Lab)"}

import torch
import folder_paths
import os
import comfy.sd
import comfy.utils

# ==============================================================================
# 1. ARTHEMY TUNER LOADER
# Description: Handles loading of Unet models with forced refresh capabilities.
# ==============================================================================
class ArthemyTunerLoader:
    """
    Loads a Unet model from the defined 'unet' folder path.
    
    Key Features:
    - Implements a caching bypass via the 'mode_lock' input and IS_CHANGED method.
    - Allows for immediate casting of model weights to FP8 formats (e4m3fn, e5m2)
      during the load process to optimize memory usage.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Retrieve the list of available files in the 'unet' directory
        unet_files = folder_paths.get_filename_list("unet")
        return {
            "required": {
                "unet_name": (unet_files, ),
                # The 'mode_lock' input acts as a trigger. It has a single value
                # to prevent user error, but its presence is used to log actions.
                "mode_lock": (["🔒 REFRESH_SAME_MODEL"],),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_and_refresh"
    CATEGORY = "Arthemy/Loaders"

    # Returning NaN ensures ComfyUI considers this node 'changed' on every execution cycle.
    # This forces the load_and_refresh function to run even if inputs haven't changed,
    # which is necessary for workflows requiring iterative model reloading.
    @classmethod
    def IS_CHANGED(s, unet_name, mode_lock, weight_dtype):
        return float("NaN")

    def load_and_refresh(self, unet_name, mode_lock, weight_dtype):
        model_path = folder_paths.get_full_path("unet", unet_name)
        
        # Console logging for debugging and verification of reload events
        print(f"\n--- ARTHEMY TUNER: Force Reloading Target ---")
        print(f"--- Target: {unet_name}")
        print(f"--- Action: {mode_lock} ---\n")

        # Standard ComfyUI Unet loader
        model = comfy.sd.load_unet(model_path)
        
        # Cast model weights if a specific floating point precision is requested
        if weight_dtype == "fp8_e4m3fn":
            model.model.to(dtype=torch.float8_e4m3fn)
        elif weight_dtype == "fp8_e5m2":
            model.model.to(dtype=torch.float8_e5m2)

        return (model,)


# ==============================================================================
# 2. ARTHEMY Z-IMAGE TUNER (SIMPLE)
# Description: Applies scaling factors to model weights using semantic blocks.
# ==============================================================================
class ArthemyZImage_Tuner_Simple:
    """
    Modifies the weights of a Diffusion Model by grouping layers into 6 semantic blocks.
    
    Technical Logic:
    - Iterates through the model's state_dict.
    - Identifies layers by index (0-29).
    - Multiplies weight tensors by a scalar value derived from user inputs.
    - Provides a 'Soft Value' mode which maps a 0.0-2.0 input range to a 
      conservative deviation from 1.0 (identity).
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["Soft Value", "Real Value"],),
                "base_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                # --- Block Definitions (Grouping 5 layers per block) ---
                "block_1_start_00_04": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "block_2_early_05_09": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "block_3_mid_10_14":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "block_4_core_15_19":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "block_5_late_20_24":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "block_6_end_25_29":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),

                # --- Component Modifiers ---
                "global_attention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "global_mlp":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                
                # --- Auxiliary Models ---
                "embedders_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "refiners_strength":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                
                # --- Normalization Handling ---
                "unsafe_tune_normalization": ("BOOLEAN", {"default": False, "label_on": "Tune Norms (Unstable)", "label_off": "Lock Norms (Stable)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "tune"
    CATEGORY = "Arthemy/Z-Image/Tuning"

    def tune(self, model, mode, base_strength, block_1_start_00_04, block_2_early_05_09, block_3_mid_10_14, 
             block_4_core_15_19, block_5_late_20_24, block_6_end_25_29,
             global_attention, global_mlp, embedders_strength, refiners_strength, unsafe_tune_normalization):
        
        print(f"--- Arthemy Simple Tuner Executing (Mode: {mode}) ---")
        
        # Helper: Calculates the actual multiplier based on mode selection.
        def get_val(v):
            if mode == "Real Value": 
                return v
            # Soft Value Mode:
            # Compresses the dynamic range to prevent model collapse.
            # Logic: 1.0 + (Input - 1.0) * 0.2
            return 1.0 + ((v - 1.0) * 0.2)

        # Helper: Applies base_strength logic.
        def calc_final(val):
            v = get_val(val)
            if mode == "Real Value": 
                v = v * base_strength
            return v

        # Map inputs to a dictionary for easier access during iteration
        params = {
            "b1": calc_final(block_1_start_00_04),
            "b2": calc_final(block_2_early_05_09),
            "b3": calc_final(block_3_mid_10_14),
            "b4": calc_final(block_4_core_15_19),
            "b5": calc_final(block_5_late_20_24),
            "b6": calc_final(block_6_end_25_29),
            "g_att": calc_final(global_attention),
            "g_mlp": calc_final(global_mlp),
            "emb": calc_final(embedders_strength),
            "ref": calc_final(refiners_strength)
        }

        # Clone model to prevent modifying the original cached object in memory
        model_clone = model.clone()
        internal_model = model_clone.model.diffusion_model
        state_dict = internal_model.state_dict()
        
        count = 0
        skipped_norm = 0
        
        for key, tensor in state_dict.items():
            if "weight" not in key and "bias" not in key: 
                continue
            
            # Normalization Layer Protection
            # Modifying norms often leads to image artifacts. We skip them unless explicitly overridden.
            is_norm_layer = ("norm" in key) or ("adaLN" in key)
            if is_norm_layer and not unsafe_tune_normalization:
                skipped_norm += 1
                continue

            scale = 1.0
            is_target = False
            
            # Logic 1: Handle Main Transformer/Unet Layers (Indices 0-29)
            if key.startswith("layers."):
                try:
                    parts = key.split(".")
                    idx = int(parts[1])
                    is_target = True
                    
                    # Assign Block Multiplier
                    if   0 <= idx <= 4:  scale = params["b1"]
                    elif 5 <= idx <= 9:  scale = params["b2"]
                    elif 10 <= idx <= 14: scale = params["b3"]
                    elif 15 <= idx <= 19: scale = params["b4"]
                    elif 20 <= idx <= 24: scale = params["b5"]
                    elif 25 <= idx <= 29: scale = params["b6"]
                    
                    # Apply Global Component Multipliers (Attention vs MLP)
                    if "attention" in key: 
                        scale *= params["g_att"]
                    elif "feed_forward" in key: 
                        scale *= params["g_mlp"]
                except: 
                    pass

            # Logic 2: Handle Embedders
            elif "embedder" in key:
                scale = params["emb"]
                is_target = True

            # Logic 3: Handle Refiners
            elif "refiner" in key:
                scale = params["ref"]
                is_target = True
            
            # Logic 4: Handle Final Output Layer
            elif "final_layer" in key:
                scale = params["b6"]
                is_target = True

            # Apply In-Place Multiplication
            # We use a threshold (1e-4) to avoid unnecessary operations for identity values (1.0).
            if is_target and abs(scale - 1.0) > 1e-4:
                tensor.mul_(scale)
                count += 1

        print(f"Update Complete: Modified {count} tensors. Skipped {skipped_norm} normalization layers.")
        return (model_clone,)


# ==============================================================================
# 3. ARTHEMY Z-IMAGE TUNER (LAB)
# Description: Provides granular access to individual layers (0-29).
# ==============================================================================
class ArthemyZImage_Tuner_Lab:
    """
    Advanced Tuner allowing per-layer weight scaling.
    Dynamically generates input fields for all 30 standard layers found in SD/Flux architectures.
    """
    
    # Configuration for dynamic input generation
    LAYER_CONFIG = []
    for i in range(30):
        LAYER_CONFIG.append({"index": i, "name": f"Layer_{i:02d}"})

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "mode": (["Soft Value", "Real Value"],),
                "base_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "unsafe_tune_normalization": ("BOOLEAN", {"default": False, "label_on": "Tune Norms", "label_off": "Lock Norms"}),
                "Noise_Refiners": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "Context_Refiners": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "Embedders_Global": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
        
        # Programmatically add inputs for Layer 00 through Layer 29
        for layer in s.LAYER_CONFIG:
            inputs["optional"][layer["name"]] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "tune_lab"
    CATEGORY = "Arthemy/Z-Image/Tuning"

    def tune_lab(self, model, mode, base_strength, unsafe_tune_normalization, **kwargs):
        print(f"--- Arthemy Lab Tuner Executing (Mode: {mode}) ---")
        
        def get_val(v):
            if mode == "Real Value": 
                return v * base_strength
            return 1.0 + ((v - 1.0) * 0.2)

        # 1. Parse Dynamic Inputs
        layer_scales = {}
        for layer in self.LAYER_CONFIG:
            user_val = kwargs.get(layer["name"], 1.0)
            layer_scales[layer["index"]] = get_val(user_val)

        val_noise = get_val(kwargs.get("Noise_Refiners", 1.0))
        val_context = get_val(kwargs.get("Context_Refiners", 1.0))
        val_embed = get_val(kwargs.get("Embedders_Global", 1.0))

        # 2. Clone and Iterate Model State
        model_clone = model.clone()
        internal_model = model_clone.model.diffusion_model
        state_dict = internal_model.state_dict()
        
        count = 0
        skipped_norm = 0
        
        for key, tensor in state_dict.items():
            if "weight" not in key and "bias" not in key: 
                continue
            
            # Check Normalization Constraints
            is_norm_layer = ("norm" in key) or ("adaLN" in key)
            if is_norm_layer and not unsafe_tune_normalization:
                skipped_norm += 1
                continue

            scale = 1.0
            is_target = False
            
            # Apply Per-Layer logic
            if key.startswith("layers."):
                try:
                    parts = key.split(".")
                    idx = int(parts[1])
                    if idx in layer_scales:
                        scale = layer_scales[idx]
                        is_target = True
                except: 
                    pass
            
            # Apply Specific Component logic
            elif "noise_refiner" in key:
                scale = val_noise
                is_target = True
                
            elif "context_refiner" in key:
                scale = val_context
                is_target = True
                
            elif "embedder" in key:
                scale = val_embed
                is_target = True

            elif "final_layer" in key:
                # The final layer typically correlates with the exit flow of the last main layer (29)
                scale = layer_scales.get(29, 1.0)
                is_target = True

            # Execute Modification
            if is_target and abs(scale - 1.0) > 1e-4:
                tensor.mul_(scale)
                count += 1
                
        print(f"Lab Update Complete: Modified {count} tensors. Skipped {skipped_norm} normalization layers.")
        return (model_clone,)


# ==============================================================================
# 4. ARTHEMY Z-IMAGE SAVER
# Description: Serializes the modified model state to disk as a Safetensors file.
# ==============================================================================
class ArthemyZImage_Saver:
    def __init__(self): 
        # CHANGE: Set base output directory to ComfyUI/output/diffusion_models
        base_output = folder_paths.get_output_directory()
        self.output_dir = os.path.join(base_output, "diffusion_models")
        
        # Ensure the directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                # CHANGE: Default filename updated
                "filename_prefix": ("STRING", {"default": "z-image_arthemy_tuned"}),
            },
            "optional": {
                "save_precision": (["fp16", "bf16", "float32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Arthemy/Z-Image/IO"

    def save(self, model, filename_prefix, save_precision):
        # CHANGE: Use the pre-calculated directory from __init__
        filename = f"{filename_prefix}.safetensors"
        full_path = os.path.join(self.output_dir, filename)
        
        print(f"--- Arthemy Saver: Exporting {save_precision} to {filename}... ---")
        try:
            internal_model = model.model.diffusion_model
            state_dict = internal_model.state_dict()
            
            # Prepare state dict for saving (CPU transfer and casting)
            clean_dict = {}
            for k, v in state_dict.items():
                if v is not None:
                    t = v.to("cpu").contiguous()
                    if save_precision == "fp16": 
                        t = t.half()
                    elif save_precision == "float32": 
                        t = t.float()
                    elif save_precision == "bf16": 
                        t = t.bfloat16()
                    clean_dict[k] = t
            
            import safetensors.torch
            safetensors.torch.save_file(clean_dict, full_path)
            print(f"Export successful: {full_path}")
        except Exception as e:
            print(f"Export failed: {e}")
        return ()


# ==============================================================================
# NODE MAPPINGS
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "ArthemyTunerLoader": ArthemyTunerLoader,
    "ArthemyZImage_Tuner_Simple": ArthemyZImage_Tuner_Simple,
    "ArthemyZImage_Tuner_Lab": ArthemyZImage_Tuner_Lab,
    "ArthemyZImage_Saver": ArthemyZImage_Saver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArthemyTunerLoader": "Arthemy Tuner Model Loader",
    "ArthemyZImage_Tuner_Simple": "Arthemy Z-Image Tuner (Simple)",
    "ArthemyZImage_Tuner_Lab": "Arthemy Z-Image Tuner (LAB)",
    "ArthemyZImage_Saver": "Arthemy Z-Image Saver"
}
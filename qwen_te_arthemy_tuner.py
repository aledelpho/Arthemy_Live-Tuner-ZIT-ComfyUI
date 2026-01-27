import torch
import re
import os
import folder_paths
from safetensors.torch import save_file
import safetensors

# ==============================================================================
# 1. ARTHEMY QWEN TUNER (SIMPLE)
# Description: Semantic block-based control for the Qwen 3.4B Text Encoder.
# ==============================================================================
class ArthemyQwenTunerSimple:
    """
    Modifies the weights of the Qwen Text Encoder by grouping its 36 layers
    into 6 semantic zones for intuitive control.

    Technical Logic:
    - Uses ComfyUI's native patching system (lazy evaluation).
    - Maps 6 sliders to specific layer ranges (6 layers per zone).
    - Includes 'Soft Value' mapping to prevent model collapse during tuning.
    """
    
    # Semantic Grouping Logic (36 Layers / 6 Groups = 6 Layers per Group)
    # Zone 1 (00-05): Tokenization & Embedding
    # Zone 2 (06-11): Low-Level Syntax
    # Zone 3 (12-17): High-Level Syntax
    # Zone 4 (18-23): Semantic Bridge
    # Zone 5 (24-29): Contextual Logic
    # Zone 6 (30-35): Abstract Reasoning
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",), 
                "mode": (["Soft Value", "Real Value"],),
                "base_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "Zone_1_Embedding_00_05": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Zone_2_Syntax_Low_06_11": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Zone_3_Syntax_High_12_17": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Zone_4_Semantics_18_23": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Zone_5_Context_24_29": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Zone_6_Abstract_30_35": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING", "DICT", )
    RETURN_NAMES = ("CLIP", "info", "debug_data", )
    FUNCTION = "tune_qwen_simple"
    CATEGORY = "Arthemy/Qwen-TE/Tuning"

    def tune_qwen_simple(self, clip, mode, base_strength, **kwargs):
        print(f"--- Arthemy Qwen Simple Tuner Executing (Mode: {mode}) ---")

        # Helper: Calculates the actual multiplier based on mode selection.
        def get_val(v):
            if mode == "Real Value": 
                return v
            # Soft Value Mode: Maps 0.0-2.0 to a safe range (e.g., 0.8-1.2)
            return 0.8 + (0.2 * v)

        w_base = get_val(base_strength)
        
        # Define Layer Ranges
        group_inputs = {
            "Zone_1_Embedding_00_05": range(0, 6),
            "Zone_2_Syntax_Low_06_11": range(6, 12),
            "Zone_3_Syntax_High_12_17": range(12, 18),
            "Zone_4_Semantics_18_23": range(18, 24),
            "Zone_5_Context_24_29": range(24, 30),
            "Zone_6_Abstract_30_35": range(30, 36),
        }

        layer_scales = {}
        debug_map = {"base_strength": base_strength, "mode": mode, "layers": {}}

        # 1. Map Groups to Individual Indices
        for group_name, layer_range in group_inputs.items():
            user_val = kwargs.get(group_name, 1.0)
            final_val = get_val(user_val)
            
            for idx in layer_range:
                layer_scales[idx] = final_val
                # Store Expected Strength (Multiplier - 1.0) for validation
                debug_map["layers"][idx] = (final_val * w_base) - 1.0

        # 2. Clone and Patch
        clip_out = clip.clone()
        model_obj = clip_out.patcher.model
        if hasattr(model_obj, "model"): 
            model_obj = model_obj.model
        
        current_keys = model_obj.state_dict().keys()
        active_patches = 0

        # 3. Iterate and Apply Patches
        for key in current_keys:
            if "norm" in key or "bias" in key: 
                continue
            
            # Regex to find layer index in standard Qwen structures
            match = re.search(r"\.layers\.(\d+)\.", key) or re.search(r"\.h\.(\d+)\.", key)
            if match:
                idx = int(match.group(1))
                if idx in layer_scales:
                    target_scale = layer_scales[idx] * w_base
                    strength = target_scale - 1.0
                    
                    if strength != 0:
                        original_weight = model_obj.state_dict()[key]
                        clip_out.add_patches({key: (original_weight,)}, strength, 1.0)
                        active_patches += 1

        info = f"Simple Tuner (6-Zones) Active | Patches: {active_patches}"
        print(f"Update Complete: {info}")
        return (clip_out, info, debug_map, )


# ==============================================================================
# 2. ARTHEMY QWEN TUNER (LAB)
# Description: Surgical control over all 36 individual layers.
# ==============================================================================
class ArthemyQwenTunerLab:
    """
    Advanced Qwen Tuner.
    Provides individual control sliders for all 36 layers of the model.
    Dynamically generates input fields for surgical precision.
    """
    
    # Generate Configuration for 36 Layers
    LAYER_CONFIG = []
    for i in range(36):
        if i < 9: prefix = "LLM_Syntax"
        elif i < 18: prefix = "LLM_Semantics"
        elif i < 27: prefix = "LLM_Context"
        else: prefix = "LLM_Abstract"
        key = f"{prefix}_L{i:02d}"
        LAYER_CONFIG.append({"index": i, "name": key})

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP",), 
                "mode": (["Soft Value", "Real Value"],),
                "base_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {}
        }
        # Dynamic inputs generation
        for layer in s.LAYER_CONFIG:
            inputs["optional"][layer["name"]] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("CLIP", "STRING", "DICT", )
    RETURN_NAMES = ("CLIP", "info", "debug_data", )
    FUNCTION = "tune_qwen_lab"
    CATEGORY = "Arthemy/Qwen-TE/Tuning"

    def tune_qwen_lab(self, clip, mode, base_strength, **kwargs):
        print(f"--- Arthemy Qwen Lab Tuner Executing (Mode: {mode}) ---")

        # Value mapping helper
        def get_val(v):
            if mode == "Real Value": 
                return v
            # Soft mapping: maps 0.0-2.0 to a safe range (e.g. 0.8-1.2)
            return 0.8 + (0.2 * v)

        w_base = get_val(base_strength)
        layer_scales = {}
        debug_map = {"base_strength": base_strength, "mode": mode, "layers": {}}
        
        # 1. Map User Inputs to Layer Indices
        for layer in self.LAYER_CONFIG:
            val = kwargs.get(layer["name"], 1.0)
            final_val = get_val(val)
            layer_scales[layer["index"]] = final_val
            
            debug_map["layers"][layer["index"]] = (final_val * w_base) - 1.0

        # 2. Clone CLIP (Non-destructive operation)
        clip_out = clip.clone()
        
        # 3. Access Internal Model Structure
        model_obj = clip_out.patcher.model
        if hasattr(model_obj, "model"): 
            model_obj = model_obj.model
        
        current_keys = model_obj.state_dict().keys()
        active_patches = 0

        # 4. Iterate and Patch
        for key in current_keys:
            if "norm" in key or "bias" in key: 
                continue
            
            # Regex to find layer index
            match = re.search(r"\.layers\.(\d+)\.", key) or re.search(r"\.h\.(\d+)\.", key)
            if match:
                idx = int(match.group(1))
                if idx in layer_scales:
                    target_scale = layer_scales[idx] * w_base
                    strength = target_scale - 1.0
                    
                    if strength != 0:
                        # Lazy Patching: We refer to the original weight in RAM.
                        # New Weight = Old Weight + (Old Weight * Strength)
                        original_weight = model_obj.state_dict()[key]
                        clip_out.add_patches({key: (original_weight,)}, strength, 1.0)
                        active_patches += 1

        info = f"Lab Tuner Active | Patches: {active_patches}"
        print(f"Lab Update Complete: {info}")
        return (clip_out, info, debug_map, )


# ==============================================================================
# 3. ARTHEMY QWEN SAVER
# Description: Serializes patches into a .safetensors file.
# ==============================================================================
class ArthemyQwenSaver:
    """
    Production-ready Saver for Qwen CLIP models.
    Supports complex nested patch structures via recursive unpacking.
    """
    def __init__(self):
        # CHANGE: Set base output directory to ComfyUI/output/text_encoders
        base_output = folder_paths.get_output_directory()
        self.output_dir = os.path.join(base_output, "text_encoders")
        
        # Ensure the directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @classmethod
    def INPUT_TYPES(s):
        # 1. Dynamically fetch available models
        # We check both folders because users might store Qwen in either
        clips = folder_paths.get_filename_list("clip")
        text_encoders = folder_paths.get_filename_list("text_encoders")
        
        # 2. Merge, deduplicate, and sort the list
        all_files = sorted(list(set(clips + text_encoders)))

        return {
            "required": {
                "tuned_clip": ("CLIP",),
                # Dropdown menu for original filename
                "original_filename": (all_files, ), 
                # CHANGE: Default filename set to "qwen_arthemy_tuned"
                "filename_prefix": ("STRING", {"default": "qwen_arthemy_tuned"}),
                "save_precision": (["fp16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_qwen"
    OUTPUT_NODE = True
    CATEGORY = "Arthemy/Qwen-TE/IO"

    def save_qwen(self, tuned_clip, original_filename, filename_prefix, save_precision):
        print(f"\n--- Arthemy Saver: Exporting {save_precision} to {filename_prefix}... ---")
        
        # --- RECURSIVE UNPACKERS ---
        # Essential for traversing ComfyUI's nested list/tuple structures for patches
        def find_tensor(data):
            if isinstance(data, torch.Tensor): return data
            if isinstance(data, (list, tuple)):
                for x in data:
                    res = find_tensor(x)
                    if res is not None: return res
            return None

        def find_strength(data):
            if isinstance(data, (float, int)): return float(data)
            if isinstance(data, (list, tuple)):
                for x in data:
                    if isinstance(x, (float, int)): return float(x)
                    res = find_strength(x)
                    if res is not None: return res
            return None
        # ---------------------------

        # 1. Locate Template File
        # We try 'clip' first, then 'text_encoders'
        template_path = folder_paths.get_full_path("clip", original_filename) or \
                        folder_paths.get_full_path("text_encoders", original_filename)
        
        if not template_path:
            print(f"Error: Template file '{original_filename}' not found on disk.")
            return ()

        try:
            patcher = tuned_clip.patcher
            internal_model = patcher.model
            ram_sd = internal_model.state_dict() if not hasattr(internal_model, "model") else internal_model.model.state_dict()

            # 2. Suffix Map: Correlate Disk Keys to RAM Keys
            suffix_map = {}
            for k in ram_sd.keys():
                clean_k = k.replace("qwen3_4b.transformer.", "").replace("model.", "")
                suffix_map[clean_k] = k 

            final_sd = {}
            count_patched = 0
            dtype = torch.float32 if save_precision == "fp32" else torch.float16

            # 3. Read Template and Apply Patches
            with safetensors.safe_open(template_path, framework="pt", device="cpu") as f:
                original_keys = f.keys()
                metadata = f.metadata() if f.metadata() else {}
                metadata["tuned_by"] = "Arthemy_Unified"

                for orig_key in original_keys:
                    clean_orig = orig_key.replace("qwen3_4b.transformer.", "").replace("model.", "")
                    
                    if clean_orig in suffix_map:
                        ram_key = suffix_map[clean_orig]
                        # Load base weight
                        weight = ram_sd[ram_key].to(device="cpu", dtype=torch.float32)
                        
                        # Find patches (checking multiple naming conventions)
                        patches = []
                        if ram_key in patcher.patches: 
                            patches = patcher.patches[ram_key]
                        elif clean_orig in patcher.patches: 
                            patches = patcher.patches[clean_orig]
                        
                        if patches:
                            for p_data in patches:
                                # Recursively find the tensor and strength value
                                p_tensor = find_tensor(p_data)
                                p_strength = find_strength(p_data)
                                
                                if p_tensor is not None and p_strength is not None:
                                    p_tensor = p_tensor.to(device="cpu", dtype=torch.float32)
                                    # Handle shape broadcasting
                                    if p_tensor.shape != weight.shape:
                                        try: 
                                            p_tensor = p_tensor.view(weight.shape)
                                        except: 
                                            pass 
                                    
                                    # Apply Math: Final = Base + (Base * Strength)
                                    weight = weight + (p_tensor * p_strength)
                                    count_patched += 1
                        
                        final_sd[orig_key] = weight.to(dtype)

            # 4. Write File
            # Uses the directory set in __init__ (output/text_encoders)
            out_path = os.path.join(self.output_dir, f"{filename_prefix}.safetensors")
            save_file(final_sd, out_path, metadata=metadata)
            
            print(f"Export successful: {out_path}")
            print(f"Total Patches Applied: {count_patched}")

        except Exception as e:
            print(f"Export failed: {e}")
            import traceback
            traceback.print_exc()

        return ()


# ==============================================================================
# NODE MAPPINGS
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "ArthemyQwenTunerSimple": ArthemyQwenTunerSimple,
    "ArthemyQwenTunerLab": ArthemyQwenTunerLab,
    "ArthemyQwenSaver": ArthemyQwenSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArthemyQwenTunerSimple": "Arthemy Qwen Tuner (Simple)",
    "ArthemyQwenTunerLab": "Arthemy Qwen Tuner (LAB)",
    "ArthemyQwenSaver": "Arthemy Qwen Saver"
}
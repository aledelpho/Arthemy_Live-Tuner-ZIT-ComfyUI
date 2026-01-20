import torch
import torch.nn as nn

class ArthemyLiveModelTunerZImageLab:
    """
    ✨ Arthemy Model Tuner (Z-Image Lab Version)
    
    Architecture: S3-DiT (Scalable Single-Stream Diffusion Transformer)
    Technique: Per-Layer Weight Injection with Auto-Restore safeguards.
    
    LAB VERSION:
    This node creates a granular control interface, exposing every single transformer block
    individually (Layers 0-29). It retains the semantic naming convention of the 
    standard version to help map specific layers to their theoretical effects.
    """
    
    # Original Taxonomy for reference and naming
    GROUP_MAP = {
        "STAGE_1_Semantic_Seeding": range(0, 6),
        "STAGE_2_Spatial_Layout": range(6, 12),
        "STAGE_3_Morphological_Form": range(12, 18),
        "STAGE_4_Volumetric_Lighting": range(18, 24),
        "STAGE_5_Surface_Refinement": range(24, 30),
    }
    
    # Generate ordered keys for all 30 layers with their parent stage name
    # Format: STAGE_X_Name_Lxx
    LAB_KEYS = []
    LAYER_TO_NAME_MAP = {}

    # We iterate strictly in order to generate the UI sliders correctly
    _ordered_groups = [
        "STAGE_1_Semantic_Seeding", 
        "STAGE_2_Spatial_Layout", 
        "STAGE_3_Morphological_Form", 
        "STAGE_4_Volumetric_Lighting", 
        "STAGE_5_Surface_Refinement"
    ]

    for group in _ordered_groups:
        rng = GROUP_MAP[group]
        for i in rng:
            key_name = f"{group}_L{i:02d}" # e.g. STAGE_1_Semantic_Seeding_L05
            LAB_KEYS.append(key_name)
            LAYER_TO_NAME_MAP[i] = key_name

    # Global Cache for Model Safety (Backups)
    # Shared with the main class logic concept, but stored separately to avoid conflicts
    BACKUP_CACHE = {}

    @classmethod
    def INPUT_TYPES(s):
        # Base inputs
        inputs = {
            "required": {
                "model": ("MODEL",),
                "mode": (["Soft Value", "Real Value"],),
                "base_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Global multiplier."}),
            },
            "optional": {}
        }
        
        # dynamically add a slider for every single layer (0-29)
        for key in s.LAB_KEYS:
            inputs["optional"][key] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
            
        return inputs

    RETURN_TYPES = ("MODEL", "STRING", )
    RETURN_NAMES = ("MODEL", "info", )
    FUNCTION = "tune_zimage_lab"
    CATEGORY = "Arthemy/Z-Image"

    def tune_zimage_lab(self, model, mode, base_scale, **kwargs):
        
        # --- 1. ACCESS PHYSICAL MODEL ---
        try:
            backbone = model.model.diffusion_model
            # Auto-detect layer container name
            if hasattr(backbone, "layers"): layer_container = backbone.layers
            elif hasattr(backbone, "joint_blocks"): layer_container = backbone.joint_blocks
            elif hasattr(backbone, "blocks"): layer_container = backbone.blocks
            else: return (model, "Error: Unknown Model Structure",)
        except AttributeError: return (model, "Error: Model Access Failed",)

        model_id = id(backbone)

        # --- 2. SAFETY BACKUP (Runs once per session) ---
        if model_id not in self.BACKUP_CACHE:
            print(f"[Arthemy Lab] 🛡️ Initializing Safety Backup for Model {model_id}...")
            self.BACKUP_CACHE[model_id] = {}
            
            total_params_backed = 0
            
            for i, block in enumerate(layer_container):
                layer_backup = {}
                # Recursively capture ALL Linear weights (Q, K, V, MLP, Proj)
                for name, module in block.named_modules():
                    if isinstance(module, nn.Linear):
                        # Offload to CPU RAM to preserve VRAM
                        layer_backup[name] = module.weight.detach().cpu().clone()
                        total_params_backed += 1
                
                self.BACKUP_CACHE[model_id][i] = layer_backup
            
            print(f"[Arthemy Lab] ✅ Backup Complete. {total_params_backed} sub-modules secured.")

        # --- 3. RESTORE ORIGINAL STATE (Reset to 1.0) ---
        # Crucial step: wipes previous modifications before applying new ones
        restored_count = 0
        if model_id in self.BACKUP_CACHE:
            for i, block in enumerate(layer_container):
                if i in self.BACKUP_CACHE[model_id]:
                    backup_dict = self.BACKUP_CACHE[model_id][i]
                    for name, module in block.named_modules():
                        if name in backup_dict and isinstance(module, nn.Linear):
                            original_weight = backup_dict[name]
                            # Copy back to GPU
                            module.weight.data.copy_(original_weight.to(module.weight.device))
                    restored_count += 1

        # --- 4. CALCULATE NEW WEIGHTS ---
        def get_target_weight(w):
            if mode == "Real Value": return w
            # Soft Value: Quadratic curve for organic feel
            if w <= 1.0: return max(0.0, -1.02 * (w**2) + 2.02 * w)
            return 1.0 + (w - 1.0) * 0.133

        w_base = get_target_weight(base_scale)
        
        # --- 5. INJECTION EXECUTION (Per-Layer) ---
        injected_layers = 0
        injected_submodules = 0
        
        for i, block in enumerate(layer_container):
            # Identify the specific slider name for this layer index
            # If the index exceeds our map (e.g. layer 30+ in future models), default to 1.0
            slider_key = self.LAYER_TO_NAME_MAP.get(i, None)
            
            if slider_key:
                slider_val = kwargs.get(slider_key, 1.0)
            else:
                slider_val = 1.0
                
            target_val = get_target_weight(slider_val)
            final_scale = target_val * w_base
            
            if final_scale == 1.0: 
                continue

            # Scale ALL Linear layers in the block (Attention + MLP)
            layer_hits = 0
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    with torch.no_grad():
                        module.weight.data.mul_(final_scale)
                    layer_hits += 1
            
            if layer_hits > 0:
                injected_layers += 1
                injected_submodules += layer_hits

        info = f"Z-Image Lab | Mode: {mode} | Layers Modulated: {injected_layers}"
        return (model, info, )

NODE_CLASS_MAPPINGS = {"ArthemyLiveModelTunerZImageLab": ArthemyLiveModelTunerZImageLab}
NODE_DISPLAY_NAME_MAPPINGS = {"ArthemyLiveModelTunerZImageLab": "✨ Arthemy Model Tuner (Z-Image Lab)"}
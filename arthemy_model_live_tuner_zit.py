import torch
import torch.nn as nn

class ArthemyLiveModelTunerZImage:
    """
    ✨ Arthemy Model Tuner (Z-Image / Universal DiT)
    
    Architecture: S3-DiT (Scalable Single-Stream Diffusion Transformer)
    Technique: Recursive Weight Injection with Auto-Restore safeguards.
    
    This node provides deep control over the Z-Image generation process by 
    modulating the physical weights of the transformer blocks in real-time.
    It includes a safety system that backs up the original model state to RAM
    and restores it before every generation, ensuring non-destructive experimentation.
    """
    
    # 5-Stage Deep Semantic Taxonomy
    GROUP_MAP = {
        # Layers 0-5: Defines global composition and prompt adherence
        "STAGE_1_Semantic_Seeding": range(0, 6),
        
        # Layers 6-11: Defines spatial masses and object positioning
        "STAGE_2_Spatial_Layout": range(6, 12),
        
        # Layers 12-17: Defines shapes, limbs, and boundaries
        "STAGE_3_Morphological_Form": range(12, 18),
        
        # Layers 18-23: Defines volumetric lighting and depth
        "STAGE_4_Volumetric_Lighting": range(18, 24),
        
        # Layers 24-29: Defines surface texture and micro-details
        "STAGE_5_Surface_Refinement": range(24, 30),
    }
    
    ORDERED_KEYS = [
        "STAGE_1_Semantic_Seeding", 
        "STAGE_2_Spatial_Layout", 
        "STAGE_3_Morphological_Form", 
        "STAGE_4_Volumetric_Lighting", 
        "STAGE_5_Surface_Refinement"
    ]

    # Global Cache for Model Safety (Backups)
    BACKUP_CACHE = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["Soft Value", "Real Value"],),
                "base_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Global multiplier."}),
            },
            "optional": {k: ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}) for k in s.ORDERED_KEYS}
        }

    RETURN_TYPES = ("MODEL", "STRING", )
    RETURN_NAMES = ("MODEL", "info", )
    FUNCTION = "tune_zimage"
    CATEGORY = "Arthemy/Z-Image"

    def tune_zimage(self, model, mode, base_scale, **kwargs):
        
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
            print(f"[Arthemy] 🛡️ Initializing Safety Backup for Model {model_id}...")
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
            
            print(f"[Arthemy] ✅ Backup Complete. {total_params_backed} sub-modules secured.")

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
        layer_weights = {}
        
        for group_name, layer_indices in self.GROUP_MAP.items():
            slider_val = kwargs.get(group_name, 1.0)
            target_val = get_target_weight(slider_val)
            final_val = target_val * w_base
            for i in layer_indices:
                layer_weights[i] = final_val

        # --- 5. INJECTION EXECUTION (Total War) ---
        injected_layers = 0
        injected_submodules = 0
        
        for i, block in enumerate(layer_container):
            scale = layer_weights.get(i, 1.0)
            
            if scale == 1.0: 
                continue

            # Scale ALL Linear layers in the block (Attention + MLP)
            layer_hits = 0
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    with torch.no_grad():
                        module.weight.data.mul_(scale)
                    layer_hits += 1
            
            if layer_hits > 0:
                injected_layers += 1
                injected_submodules += layer_hits

        info = f"Z-Image Active | Mode: {mode} | Layers Modulated: {injected_layers}"
        return (model, info, )

NODE_CLASS_MAPPINGS = {"ArthemyLiveModelTunerZImage": ArthemyLiveModelTunerZImage}
NODE_DISPLAY_NAME_MAPPINGS = {"ArthemyLiveModelTunerZImage": "✨ Arthemy Model Tuner (Z-Image)"}
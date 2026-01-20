# Arthemy Live Tuner Z-Image • ComfyUI nodes

Advanced real-time control for **Z-Image** (S3-DiT) models and **Qwen3-4B** Text Encoders. These nodes perform "live surgery" on your model's weights to boost or lower specific generative phases without any training.

---

# ✨ Arthemy Live Tuner - Z-Image (standard version)

Hi everyone!

I developed these nodes to slice the monolithic **S3-DiT** of **Z-Image** to give you more control over different slices of both the Model and Text Encoder.
This transform your **Z-Image** checkpoint into a set of "Sliders," allowing you to tweak composition, lighting, or micro-details in real-time.

We can finally **play with the models**!

*Work in Progress: Please keep in mind that the boundaries for these functional blocks are experimental. Since Z-Image is a single-stream architecture... it's quite chaotic.
If you find that a specific slider affects something different than described, let me know! Your feedback helps refine this mapping for everyone.*

---

## Model Tuner: The 5 Generative Stages

Z-Image doesn't have "Input" or "Output" blocks. Instead, it uses a flat stack of **30 identical layers**. I’ve organized these into **5 Functional Stages**:

### STAGE 1: Semantic Seeding (Layers 0-5)

* **The Foundation:** Where random noise first meets your prompt.
- Crank this up to force the model to stick rigidly to your prompt's layout. Lower it for more creative drift.

### STAGE 2: Spatial Layout (Layers 6-11)

* **The Blueprint:** Defines masses, object positioning, and depth.
- Adjust where subjects are placed and their overall "weight" in the scene.

### STAGE 3: Morphological Form (Layers 12-17)

* **The Skeleton:** Resolves object boundaries and limb coherence.
- High values separate objects clearly; low values create "dream-like" blending.

### STAGE 4: Volumetric Lighting (Layers 18-23)

* **The Atmosphere:** Handles 3D depth, shading, and lighting mood.
- Influence the contrast and "feel" of the environment.

### STAGE 5: Surface Refinement (Layers 24-29)

* **The Detailer:** "Hallucinates" micro-details like skin pores and fabric weave.
- Crank it for ultra-sharp textures; lower it for a soft, painterly look.

---

## Qwen Tuner: Language Abstraction

Z-Image uses **Qwen3-4B**, a 36-layer LLM, as its text encoder. This node lets you adjust how the model "thinks" about your prompt:

* **Syntax Parsing**: Controls how strictly the AI follows your grammar and word order.
* **Literal Meaning**: Controls the direct association between words and objects.
* **Contextual Web**: Manages the relationships and interactions between subjects.
* **Abstract Concept**: Controls the overall artistic style and subtext interpretation.

---

## ⚗️ Arthemy Live Tuner - Z-Image  (LAB version)

The Lab version of these models keep the same nomenclature of the standard version, in order to help you identify to what group each layer has been assigned to, but it exposes all the layers of both the Model and Qwen to give you complete granular control.
If you're one of those people that want to help me understand what's inside each slice in order to make these tools more reliable and effective... you're welcome to delve into this version.

---

### **Tuning Modes: Real Value & Soft Value**

The Arthemy Live Tuner nodes offer two distinct mathematical modes to control how your slider movements actually translate into weight changes. Understanding these is key to mastering the "God Mode" injection without breaking the model.

#### **1. Real Value (Direct & Aggressive)**

In this mode, the slider value is used as a **direct linear multiplier**.

* **Multiplier:** If you set a slider to `1.20`, the weights are multiplied exactly by `1.20`.
* **Sensitivity:** Because Z-Image V14 uses "Total War" recursive scaling (input * output), this mode is extremely sensitive.
* **Best For:** Experienced users performing deep, structural changes or those looking to push the model to its absolute limits. Small increments (e.g., `0.01` or `0.05`) are highly recommended here.

#### **2. Soft Value (Organic & Safe)**

This is the recommended mode for general artistic tuning. It uses a **non-linear quadratic curve** to make changes feel more natural.

* **The Curve:** It implements a specific mathematical easing:
* **Lowering Values ():** It follows a quadratic path () that creates a smoother, more organic reduction.
* **Boosting Values ():** The power is significantly dampened (). This acts as a safety buffer, preventing the model from "exploding" into noise even if you crank the slider to `2.0`.
* **Qwen Sensitivity:** For the **Qwen Text Tuner**, Soft Value is even more conservative ( to  range) because Large Language Models are famously fragile when their weights are modified.
* **Best For:** Fine-tuning details, subtle lighting shifts, and surgical textural adjustments.

---

## ⚡ Technical Note: Direct Injection Workaround

Because standard patching is ignored by Z-Image loaders, these nodes use **Recursive Direct Weight Injection**. They talk directly to the tensors in your VRAM.

### ⚠️ Potential Issues & Solutions

* **Memory Corruption:** If ComfyUI crashes or is interrupted during generation, the model weights in VRAM (not the actual model, don't worry) might stay "scaled" (corrupted) because the restore step didn't finish.
* **Fix:** Simply restart ComfyUI or reload your checkpoint to clear the VRAM.

* **Chaining Nodes:** Using two Model Tuners in a row will cause conflicts. The second node will back up the already-modified weights of the first node, making it impossible to return to a "clean" 1.0 state.
* **Fix:** Always use **only one** Model Tuner per workflow.

* **Sensitivity:** This method is extremely powerful. Move sliders in tiny increments (e.g., 0.05). A value of 1.50 is often enough to completely "break" the image.

---

## Installation

1. Navigate to your `ComfyUI/custom_nodes/` folder.
2. Run: `git clone https://github.com/aledelpho/ComfyUI-Arthemy-ZImage-Tuner.git`
3. Restart ComfyUI.

---

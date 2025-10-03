# -*- coding: utf-8 -*-
import streamlit as st
import torch
# Import all necessary classes
from pipeline_stable_diffusion_3_S import StableDiffusion3SPipeline
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, StableDiffusion3Pipeline, SD3Transformer2DModel
from transformers import BitsAndBytesConfig # For quantization
from PIL import Image
from io import BytesIO
import os
import traceback

# --- 1. Page Configuration ----------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Image Generator")


# --- Sidebar ------------------------------------------------------------------------------------------------------------------------------------------------
st.sidebar.title("🛠️ Admin Tools")
if st.sidebar.button("💀 Force Model Rescan (Clear Cache)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Caches cleared! Rerunning the app...")
    st.rerun()

# --- 2. Constants and Paths ---------------------------------------------------------------------------------------------------------------------------------
MODELS_ROOT_PATHS = [
    "/root/.cache/huggingface/hub",
    "/hdd1/text_to_image_models"
]


# --- 3. Model Scanning and Loading --------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def find_models(search_paths):
    # This function is already robust and does not need changes.
    available_models = {}
    print(f">>> Scanning for models in: {search_paths}")
    for search_path in search_paths:
        if not os.path.exists(search_path):
            st.warning(f"Directory does not exist, skipping: {search_path}")
            continue
        top_level_dirs = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d)) and d.startswith("models--")]
        for model_dir_name in top_level_dirs:
            display_name = model_dir_name.replace("models--", "").replace("--", "/")
            model_top_path = os.path.join(search_path, model_dir_name)
            for root, dirs, files in os.walk(model_top_path):
                if "model_index.json" in files:
                    full_model_path = root
                    source_prefix = os.path.basename(os.path.normpath(search_path))
                    unique_display_name = f"[{source_prefix}] {display_name}"
                    available_models[unique_display_name] = full_model_path
                    dirs[:] = []
                    break
    print(f">>> Scan complete. Found {len(available_models)} model(s).")
    return available_models

# --- FINAL, ULTIMATE VERSION of the load_model function -----------------------------------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path, model_name):
    """
    Loads a model intelligently using the specific, correct pipeline for each model type.
    """
    if not model_path or not os.path.exists(model_path):
        return None

    print(f">>> Loading model '{model_name}' from '{model_path}'...")

    # --- Intelligent Pipeline Dispatcher --------------------------------------------------------------------------------------------------------------------

    # Case 1: For Stable Diffusion 3 Medium (with 4-bit quantization) 
    if "stable-diffusion-3.5-medium" in model_name:
        print(">>> Dispatching to: SD3 Medium Quantized Loader")
        #nf4_config = BitsAndBytesConfig(
        #   load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        #)
        print(">>> Loading Transformer in 4-bit...")
        #model_nf4 = SD3Transformer2DModel.from_pretrained(
        #    model_path, subfolder="transformer", quantization_config=nf4_config,torch_dtype=torch.bfloat16
        #)
        print(">>> Assembling pipeline with quantized Transformer...")
        pipe = StableDiffusion3Pipeline.from_pretrained(
           model_path, torch_dtype=torch.bfloat16
        )
    
    # Case 2: For the custom Japanese model
    elif "japanese-stable-diffusion-xl" in model_name:
        print(">>> Dispatching to: DiffusionPipeline (for custom remote code)")
        pipe = DiffusionPipeline.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16, variant="fp16"
        )

    # Case 3: For the specialized SD-Turbo model
    elif "sd-turbo" in model_name and "xl" not in model_name:
        print(">>> Dispatching to: StableDiffusionTurboPipeline (specialized for speed)")
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16"
        )

    # Case 4: SD3.5_small_preview
    elif "Stable-Diffusion-3.5-Small-Preview1" in model_name:
        pipe = StableDiffusion3SPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
            
    # Case 5: Default for all other standard models (like SDXL, etc.)
    else:
        print(">>> Dispatching to: AutoPipelineForText2Image (general purpose)")
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16", trust_remote_code=True
        )

    # Enable memory optimization for all models
    print(">>> Enabling model CPU offload...")
    pipe.enable_model_cpu_offload()
    print(">>> Model is ready with CPU offloading enabled!")
    return pipe

# --- The rest of the file (generate_image, UI) remains the same ---------------------------------------------------------------------------------------------
def generate_image(pipe, prompt, negative_prompt, steps, guidance_scale, model_name):
    try:
        kwargs = {"prompt": prompt, "negative_prompt": negative_prompt}
        if "turbo" in model_name:
            kwargs["num_inference_steps"] = 1
            kwargs["guidance_scale"] = 0.0
        else:
            kwargs["num_inference_steps"] = steps
            kwargs["guidance_scale"] = guidance_scale

        if "stable-diffusion-3.5-medium" in model_name:
            kwargs["max_sequence_length"] = 512
            
        image = pipe(**kwargs).images[0]
        torch.cuda.empty_cache()
        return image
    except Exception as e:
        st.error(f"An error occurred during image generation: {e}")
        torch.cuda.empty_cache()
        return None

# --- Main UI ------------------------------------------------------------------------------------------------------------------------------------------------
st.title("🎨 AI Image Generator")
st.header("1. Select a Model")
model_dict = find_models(MODELS_ROOT_PATHS)

if not model_dict:
    st.error("No available models found in any specified paths.")
else:
    selected_model_name = st.selectbox("Choose one model from the list:", options=sorted(list(model_dict.keys())))
    selected_model_path = model_dict.get(selected_model_name)

    if selected_model_path:
        with st.spinner(f'Loading model: {selected_model_name}, please wait...'):
            pipe = load_model(selected_model_path, selected_model_name)

        if pipe:
            st.header("2. Input your prompt and generate")
            with st.form("prompt_form"):
                prompt = st.text_area("Prompt:", "A whimsical and creative image...", height=100)
                negative_prompt = st.text_area("Negative Prompt:", "low quality, blurry, watermark", height=50)

                if "turbo" in selected_model_name and "xl" not in selected_model_name:
                    st.info("SD-Turbo model is optimized for 1 step and no guidance scale.")
                    steps = 1
                    guidance_scale = 0.0
                elif "stable-diffusion-3.5-medium" in selected_model_name:
                    col1, col2 = st.columns(2)
                    with col1:
                        steps = st.slider("Steps", 1, 50, 28)
                    with col2:
                        guidance_scale = st.slider("Guidance Scale", 0.0, 10.0, 4.5)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        steps = st.slider("Steps", 1, 50, 25)
                    with col2:
                        guidance_scale = st.slider("Guidance Scale", 0.0, 15.0, 7.0)

                submitted = st.form_submit_button("✨ Generate Image")

            if submitted:
                with st.spinner("The model is processing your request..."):
                    generated_image = generate_image(pipe, prompt, negative_prompt, steps, guidance_scale, selected_model_name)
                    if generated_image:
                        st.success("Image generated successfully!")
                        st.image(generated_image, caption=f"Generated with {selected_model_name}", use_container_width=True)
                        buf = BytesIO()
                        generated_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label="📥 Download Image",
                            data=byte_im,
                            file_name=f"{selected_model_name.split('/')[-1]}_output.png",
                            mime="image/png"
                        )
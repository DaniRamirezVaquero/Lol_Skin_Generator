import streamlit as st
import requests
import os
import base64
from io import BytesIO
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="LolSkinGenerator", page_icon="🐦‍🔥")

# Título de la página
st.title("LolSkinGenerator 🐦‍🔥")


# Function to check if the required LoRa model is available
def check_lora_model():
    api_url = "http://127.0.0.1:7860/sdapi/v1/loras"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        loras = response.json()
        for lora in loras:
            if "LeagueoflegendsSkins_concept-20" in lora["name"]:
                return True
        return False
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API. Please check your internet connection and try again.")
        return False
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return False
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return False

# Function to check if ADetailer script is available
def check_adetailer_available():
    api_url = "http://127.0.0.1:7860/sdapi/v1/extensions"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        extensions = response.json()
        return any("adetailer" in extension["name"].lower() for extension in extensions)
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API. Please check if Stable Diffusion WebUI is running.")
        return False
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while checking for ADetailer: {http_err}")
        return False
    except Exception as err:
        st.error(f"An error occurred while checking for ADetailer: {err}")
        return False

# Function to get available models
def get_available_models():
    api_url = "http://127.0.0.1:7860/sdapi/v1/sd-models"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        models = response.json()
        return [model["model_name"] for model in models]
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API. Please check your internet connection and try again.")
        return []
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return []
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return []
    
def get_samplers():
    api_url = "http://127.0.0.1:7860/sdapi/v1/samplers"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        samplers = response.json()
        return [sampler["name"] for sampler in samplers]
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API. Please check your internet connection and try again.")
        return []
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return []
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return []

# Function to call the StableDiffusionWebUI API
def generate_skin(prompt, model, multiple_poses, enable_hr):
    api_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    width = 768 if multiple_poses else 512
    height = 512 if multiple_poses else 768
    additional_prompts = "(multiple views)" if multiple_poses else ""
    lora = "<lora:LeagueoflegendsSkins_concept:0.7>"
    negative_prompt = "(low quality, worst quality:1.2), (normal quality:1.2),(worst quality, low quality, letterboxed), (deformed face), (ugly face)"
    
    # Base payload
    payload = {
        "prompt": f"{additional_prompts}, (LolPreview:1.5), {prompt}, (dynamic pose:1.2), (from above:1), (full body:1.2), (zooming out:1.2), (masterpiece:1.2), (best quality, highest quality), (ultra-detailed), (8k, 4k, intricate), {lora}",
        "negative_prompt": negative_prompt,
        "steps": 50,
        "width": width,
        "height": height,
        "restore_faces": True,
        "tiling": False,
        "sampler_index": sampler,  # Usar el sampler seleccionado
        "send_images": True,
        "cfg_scale": 8,
        "override_settings": {
            "sd_model_checkpoint": model,
            "CLIP_stop_at_last_layers": 2,
        }
    }

    # Añadir ADetailer solo si está disponible
    if check_adetailer_available():
        payload["alwayson_scripts"] = {
            "ADetailer": {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                        "ad_prompt": f"{prompt}, (high detailed face:1.2), (beautiful face), (perfect face), (detailed eyes), (detailed facial features)",
                        "ad_negative_prompt": negative_prompt,
                        "ad_confidence": 0.3,
                        "ad_mask_blur": 4,
                        "ad_denoising_strength": 0.4,
                        "ad_inpaint_width": 512,
                        "ad_inpaint_height": 512,
                        "ad_cfg_scale": 7,
                        "ad_steps": 20,
                        "ad_sampler": "Euler a"
                    }
                ]
            }
        }
    else:
        # Si ADetailer no está disponible, reforzar parámetros base para mejor calidad
        payload["restore_faces"] = True
        payload["steps"] = 60  # Aumentar pasos
        payload["cfg_scale"] = 9  # Aumentar cfg_scale
        # Añadir más énfasis en la calidad facial al prompt
        payload["prompt"] += ", (detailed face), (beautiful face), (high quality face), (detailed eyes)"
    
    # Add HR specific parameters only if HR is enabled
    if enable_hr:
        payload.update({
            "enable_hr": True,
            "hr_scale": 2.0,
            "hr_upscaler": "R-ESRGAN 4x+",
            "hr_second_pass_steps": 20,
            "denoising_strength": 0.5
        })
    else:
        payload["enable_hr"] = False
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API. Please check your internet connection and try again.")
        return None
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return None

# Function to decode base64 image
def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

# Function to convert PIL image to bytes
def image_to_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Function to store image in session state
def store_image_in_session(image):
    st.session_state.generated_image = image_to_bytes(image)

# En la sección donde se muestra la interfaz
if check_lora_model():
    models = get_available_models()
    samplers = get_samplers()
    adetailer_available = check_adetailer_available()

    if not adetailer_available:
        st.warning("ADetailer extension is not installed. Face generation quality might be reduced. For better results, install [ADetaile0r](https://github.com/Bing-su/adetailer) extension in your Stable Diffusion WebUI.")

    if models:
        model = st.selectbox("Select a model:", models, help="Note: The skin generator is only compatible with SD 1.5 models.")
        sampler = st.selectbox("Select a sampler:", samplers, index=samplers.index("DPM++ SDE"), help="Recommended Samplers: Euler a, DPM++ SDE")
        prompt = st.text_input("Enter a prompt to generate a skin:", help="Examples: - 1girl, blue hair, dress / - 1boy, paladin, full armor, golden, epic skin, black hair,")
        col1, col2 = st.columns([1, 2])
        with col1:
            multiple_poses = st.checkbox("Generate multiple poses")
        with col2:
            enable_hr = st.checkbox("Enable high resolution (More detailed but slower)")

        if st.button("Generate Skin"):
            if prompt:
                with st.spinner("Generating skin..."):
                    result = generate_skin(prompt, model, multiple_poses, enable_hr)
                    if result and "images" in result and len(result["images"]) > 0:
                        image = decode_base64_image(result["images"][0])
                        store_image_in_session(image)
                    else:
                        st.error("Failed to generate skin. Please try again.")
            else:
                st.warning("Please enter a prompt to generate a skin.")
        
        # Display the stored image if it exists
        if "generated_image" in st.session_state:
            st.image(st.session_state.generated_image, caption="Generated Skin")
            st.download_button(
                label="Download Image",
                data=st.session_state.generated_image,
                file_name="generated_skin.png",
                mime="image/png"
            )
    else:
        st.warning("No models found. Please download a model and place it in the models/Stable-diffusion folder.")
else:
    st.warning("Required LoRa model not found. Please download the model from [CivitAI](https://civitai.com/api/download/models/196344?type=Model&format=SafeTensor) and place it in your models/Lora folder.")

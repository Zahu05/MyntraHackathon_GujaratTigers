#PROMPT TO IMAGE
from flask import Flask, request, jsonify
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from PIL import Image
import torch
from pyngrok import ngrok
import base64
from flask import Flask, render_template, request
from torch import autocast

import os
os.environ["HUGGINGFACE_TOKEN"] = "HUGGING-FACE-TOKEN"
Hugging_face = "TOKEN"
app = Flask(__name__,template_folder='templates',static_folder='static')

class config :
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HEIGHT = 512
    WIDTH = 512
    NUM_INFERENCE_STEPS = 800
    GUIDANCE_SCALE = 8.5
    GENERATOR = torch.manual_seed(48)
    BATCH_SIZE = 1

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=Hugging_face)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=Hugging_face)
vae = vae.to(config.DEVICE)
device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = text_encoder.to(config.DEVICE)
unet = unet.to(config.DEVICE)

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=Hugging_face)
pipe = pipe.to(config.DEVICE)
print(f'\033[94mStable Diffusion Pipeline created !!!')

@app.route('/')

def index():
    return render_template('index5.html')

@app.route('/process_prompt', methods=['POST'])
def process_prompt():
    data = request.get_json()
    prompt = data['prompt']

    with autocast(device):
        images = pipe([prompt], num_inference_steps=config.NUM_INFERENCE_STEPS).images

    grid = image_grid(images, rows=1, cols=1)
    from io import BytesIO
    buffered = BytesIO()
    grid.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'image': img_str})


if __name__ == "__main__":
    ngrok.set_auth_token("NGROK-TOKEN")
    public_url = ngrok.connect(5000)  
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000\"".format(public_url))
    app.run(port=5000) 

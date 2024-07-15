from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionImg2ImgPipeline
import base64
from io import BytesIO

app = Flask(__name__,template_folder='templates',static_folder='static')
run_with_ngrok(app)  

class Config:
    WIDTH = 512
    HEIGHT = 512
    DEVICE = "cuda" 
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    SD_MODEL = "runwayml/stable-diffusion-v1-5"  

config = Config()

# Initialize models
clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(config.DEVICE)
clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(config.SD_MODEL, torch_dtype=torch.float16).to(config.DEVICE)

@app.route('/')
def home():
    return render_template('index3.html') 

@app.route('/generate_image', methods=['POST'])
def generate_image():
    image = request.files['image']
    prompt = request.form['prompt']

    input_image = Image.open(image).convert("RGB")
    input_image = input_image.resize((config.WIDTH, config.HEIGHT))


    inputs = clip_processor(images=input_image, return_tensors="pt")
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    init_image = input_image
    strength = 0.75  # Adjust strength as needed
    with torch.autocast(config.DEVICE):
        images = pipe(prompt=prompt, image=init_image, strength=strength, num_inference_steps=100, guidance_scale=7.5).images

    buffered = BytesIO()
    images[0].save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'image': img_str})

if __name__ == "__main__":
    ngrok.set_auth_token("NGROK-TOKEN")
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000\"".format(public_url))
    app.run()

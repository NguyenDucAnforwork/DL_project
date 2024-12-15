from flask import Flask
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from config import MODEL_CONFIG
from model import CycleGAN
from huggingface_hub import hf_hub_download

# Load the CycleGAN models
repo_ids = {
    "cyclegan_cezanne_unet_300": "batmangiaicuuthegioi/cyclegan_cezanne_unet_300",
    # "cyclegan_monet_unet_250": "batmangiaicuuthegioi/cyclegan_monet_unet_250",
    # "cyclegan_vangogh_resnet_70": "batmangiaicuuthegioi/cyclegan_vangogh_resnet_70",
    # "cyclegan_vangogh_unet_70": "batmangiaicuuthegioi/cyclegan_vangogh_unet_70",
}

model_paths = {name: hf_hub_download(
    repo_id=repo_id,
    filename="model.ckpt")  
for name, repo_id in repo_ids.items()}

models = {name: CycleGAN.load_from_checkpoint(model_path, **MODEL_CONFIG) for name, model_path in model_paths.items()}

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define the image translation function
def translate_image(input_image, style):
    model = models[style]
    image = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        translated_image = model(image)
    return transforms.ToPILImage()(translated_image.squeeze(0))

# Initialize the Gradio interface
iface = gr.Interface(
    fn=translate_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(models.keys()), label="Select Style")
    ],
    outputs=gr.Image(type="pil"),
    title="CycleGAN Image Translation",
    description="Upload an image and select a style to translate it using CycleGAN."
)

if __name__ == "__main__":
    iface.launch(debug=True)
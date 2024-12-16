from flask import Flask
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from config import MODEL_CONFIG
from model import CycleGAN, UNetGenerator
from AdaIn import *
from transfer_original import *
from keras.applications.vgg19 import preprocess_input
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# # Load the CycleGAN models
model_paths = {
    "cezanne_unet_300": "./checkpoints/cyclegan_cezanne_unet_300_epochs.ckpt",
    "monet_unet_250": "./checkpoints/cyclegan_monet_unet_250_epochs.ckpt",
    "vangogh_resnet_70": "./checkpoints/cyclegan_vangogh_resnet_70_epochs.ckpt",
    "vangogh_unet_70": "./checkpoints/cyclegan_vangogh_unet_70_epochs.ckpt",
    "ukiyoe_unet_20": "./checkpoints/G_BA_20_epoch.pth",
}

models = {}
for model_name, model_path in model_paths.items():
    config = MODEL_CONFIG
    if "resnet" in model_name:
        config["gen_name"] = "resnet"
    else:
        config["gen_name"] = "unet"
    print(model_name)

    if model_name != "ukiyoe_unet_20":
        model = CycleGAN.load_from_checkpoint(model_path, **config)
    else:
        model = UNetGenerator(hid_channels=64, in_channels=3, out_channels=3).cuda()
        model.load_state_dict(torch.load(model_path))
        
    models[model_name] = model

adain_model = torch.load("./checkpoints/adain_model")
print("neural_style_transfer")

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def convert_image(tensor):
    tensor = (tensor + 1) * 127.5  # Scale from [-1, 1] to [0, 255]
    tensor = tensor.round().clamp(0, 255).to(torch.uint8)
    return tensor

# Define the image translation function
def translate_image(input_image, style):
    print(style)
    model = models[style]
    image = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        translated_image = convert_image(model(image.cuda()))
    
    return transforms.ToPILImage()(translated_image.squeeze(0))

# Define the function for neural style transfer
def adain_neural_style_transfer(content_image, style_image):
    content_image = transform(content_image).unsqueeze(0)
    style_image = transform(style_image).unsqueeze(0)
    output_img = adain_model(content_image.to('cuda'), style_image.to('cuda'))
    return transforms.ToPILImage()(convert_image(output_img.squeeze(0)))

def original_neural_style_transfer(content_image, style_image):
    content_image = transform(content_image).unsqueeze(0).cpu().numpy() * 255.
    style_image = transform(style_image).unsqueeze(0).cpu().numpy() * 255.
    content_image = preprocess_input(content_image.transpose((0, 2, 3, 1)))
    style_image = preprocess_input(style_image.transpose((0, 2, 3, 1)))
    final_image = scale_img(transfer_original(content_image, style_image, epochs=5)) * 255.
    final_image = torch.tensor(final_image).type(torch.uint8).permute(2, 0, 1)
    return transforms.ToPILImage()(final_image)

# Function to process the input
def process_input(content_image, style_image, style, nst_type, artist):
    if style == "neural_style_transfer":
        if nst_type == "adain":
            return adain_neural_style_transfer(content_image, style_image)
        else:
            return original_neural_style_transfer(content_image, style_image)
    else:
        return translate_image(content_image, artist)

# Dynamic input visibility logic
def update_inputs(style):
    if style == "neural_style_transfer":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True),gr.update(visible=False), gr.update(visible=True)

# Initialize the Gradio interface
with gr.Blocks() as iface:
    with gr.Row():
        gr.Markdown("# CycleGAN and Neural Style Transfer")
    
    with gr.Row():
        style_dropdown = gr.Dropdown(
            choices=['cyclegan', 'neural_style_transfer'],
            label="Select Model",
            value="cyclegan"
        )

    with gr.Row():
        content_image = gr.Image(type="pil", label="Content Image")
        style_image = gr.Image(type="pil", visible=False, label="Style Image (For Style Transfer)")
        nst_type = gr.Dropdown(choices=["adain", "original"], visible=False, label="Select Neural Style Transfer Type")
        artist_dropdown = gr.Dropdown(choices=list(models.keys()), visible=True, label="Select Style")

    style_dropdown.change(
        update_inputs,
        inputs=[style_dropdown],
        outputs=[style_image, content_image, nst_type, artist_dropdown],
    )
    
    output_image = gr.Image(type="pil", label="Output Image")
    submit = gr.Button("Submit")

    submit.click(
        process_input,
        inputs=[content_image, style_image, style_dropdown, nst_type, artist_dropdown],
        outputs=output_image,
    )

if __name__ == "__main__":
    iface.launch(debug=True, share=True)

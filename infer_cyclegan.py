import argparse
import torch
from model import CycleGAN, UNetGenerator
from config import MODEL_CONFIG
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def load_checkpoint(model_path, device='cuda'):
    config = MODEL_CONFIG
    if "cyclegan" in model_path and "unet" in model_path:
        model = CycleGAN.load_from_checkpoint(model_path, **config)
    elif "resnet" in model_path:
        config["gen_name"] = "resnet"
        model = CycleGAN.load_from_checkpoint(model_path, **config)
    else:
        model = UNetGenerator(hid_channels=64, in_channels=3, out_channels=3).to(device)
        model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
    return model

def convert_image(tensor):
    tensor = (tensor + 1) * 127.5  # Scale from [-1, 1] to [0, 255]
    tensor = tensor.round().clamp(0, 255).to(torch.uint8)
    return tensor

def image_to_tensor(image_path):
    """Read an image from the given path and convert it to a tensor."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    return transform(image)

def infer(model_path, img_path, device='cuda'):
    model = load_checkpoint(model_path, device=device)
    img = image_to_tensor(img_path).unsqueeze(0)
    translated_image = convert_image(model(img.to(device)))
    final_img = transforms.ToPILImage()(translated_image.squeeze(0))
    final_img.save("output_cyclegan.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Path to the model checkpoint file."
    )
    parser.add_argument(
        '--image_path', 
        type=str, 
        required=True,
        help="Path to the input image to convert to a tensor."
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        choices=['cuda', 'cpu'], 
        help="Device to load the model on (default: 'cuda')."
    )
    args = parser.parse_args()
    infer(args.model_path, args.image_path, args.device)



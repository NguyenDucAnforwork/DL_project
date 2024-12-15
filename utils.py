import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from model import CycleGAN

# Load and preprocess the input image
def load_image(image_path, device, image_size=(256, 256)):
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

# Display the output image
def display_image(tensor_image):
    tensor_image = tensor_image.squeeze(0).cpu()  # Remove batch dimension
    tensor_image = (tensor_image * 0.5 + 0.5).clamp(0, 1)  # Denormalize
    plt.imshow(tensor_image.permute(1, 2, 0))  # CHW to HWC
    plt.axis("off")
    plt.show()

# Load the input image
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CycleGAN.load_from_checkpoint("/content/cyclegan_monet_unet_250_epochs.ckpt", **MODEL_CONFIG)


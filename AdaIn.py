import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob
from torchvision.utils import save_image


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) + 0.000000023) / (
                    x.shape[2] * x.shape[3]))

    def forward(self, x, y):
        return (self.sigma(y) * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + self.mu(y)).permute(
            [2, 3, 0, 1])


class AdaINStyle(nn.Module):
    def __init__(self):
        super().__init__()
        # For encoder load pretrained VGG19 model and remove layers upto relu4_1
        self.vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:21])
        # Create AdaIN layer
        self.ada = AdaIN()
        # Use Sequential to define decoder [Just reverse of vgg with pooling replaced by nearest neigbour upscaling]
        self.dec = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU()  #Maybe change to a sigmoid to get into 0,1 range?
        )

    def forward(self, c, s):
        self.c_emb = self.vgg(c)
        self.s_emb = self.vgg(s)
        # Use AdaIN layer to make the mean and variance of c_emb (content) into that of s_emb (style)
        self.t = self.ada(self.c_emb, self.s_emb)
        return self.dec(self.t)


def loadImage(filename):
    input_image = Image.open(filename).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)


class ImageDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = glob.glob(path + '/*.jpg')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_image = Image.open(self.files[idx]).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        return input_tensor


def mu(x):
    return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])


def sigma(x):
    return torch.sqrt(
        torch.sum((x.permute([2, 3, 0, 1]) - mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) / (x.shape[2] * x.shape[3]))


class AdaINLoss(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def contentLoss(self, content_emb, output_emb):
        return torch.norm(content_emb - output_emb)

    def styleLoss(self, style_activations, output_activations):
        mu_sum = 0
        sigma_sum = 0
        for style_act, output_act in zip(style_activations, output_activations):
            mu_sum = torch.norm(mu(style_act) - mu(output_act))
            sigma_sum = torch.norm(sigma(style_act) - sigma(output_act))
        return mu_sum + sigma_sum

    def totalLoss(self, content_emb, output_emb, style_activations, output_activations):
        content_loss = self.contentLoss(content_emb, output_emb)
        style_loss = self.styleLoss(style_activations, output_activations)
        #print(content_loss.item(), style_loss.item())
        return content_loss + self.lambda_ * style_loss

    def forward(self, content_emb, output_emb, style_activations, output_activations):
        """ For caculating single image loss please pass arguments with a batch size of 1. """
        return self.totalLoss(content_emb, output_emb, style_activations, output_activations) / content_emb.shape[0]


style_layers = ['1', '6', '10', '20']
debug_layers = [0, 3, 5, 7]
activations = [None] * 4
debug_activations = [None] * 4
debug_grads = [None] * 4


# declare hook function
def styleHook(i, module, input, output):
    global activations
    activations[i] = output


def debugHook(i, module, input, output):
    global activations
    debug_activations[i] = output


def AdaIN_infer():
    def get_image_arguments():
        # Initialize the argument parser
        parser = argparse.ArgumentParser(description="Get paths for content and style images. Then get the model use")

        # Add arguments for the two image paths
        parser.add_argument(
            "--content",
            type=str,
            required=True,
            help="Path to the content image."
        )
        parser.add_argument(
            "--style",
            type=str,
            required=True,
            help="Path to the style image."
        )

        parser.add_argument(
            "--model_path",
            type=str,
            required=False,
            help="Model Path Used"
        )

        # Parse the arguments
        args = parser.parse_args()

        return args.content, args.style, args.model_path

    content_path, style_path, model_path = get_image_arguments()
    model = torch.load(model_path)
    content_img = loadImage(content_path)
    style_img = loadImage(style_path)

    output_img = model(content_img.to('cuda'), style_img.to('cuda'))
    save_image(output_img, 'output_adain.jpg')

if __name__ == "__main__":
    AdaIN_infer()

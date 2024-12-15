import torch
DEBUG = False

MODEL_CONFIG = {
    # the type of generator, and the number of residual blocks if ResNet generator is used
    "gen_name": "unet", # types: 'unet', 'resnet'
    "num_resblocks": 6,
    # the number of filters in the first layer for the generators and discriminators
    "hid_channels": 64,
    # using DeepSpeed's FusedAdam (currently GPU only) is slightly faster
    "optimizer": torch.optim.Adam,
    # the learning rate and beta parameters for the Adam optimizer
    "lr": 3e-6,
    "betas": (0.5, 0.999),
    # the weights used in the identity loss and cycle loss
    "lambda_idt": 0,
    "lambda_cycle": (10, 10), # (MPM direction, PMP direction)
    # the size of the buffer that stores previously generated images
    "buffer_size": 100,
    # the number of epochs for training
    "num_epochs": 30 if not DEBUG else 70,
    # the number of epochs before starting the learning rate decay
    "decay_epochs": 10 if not DEBUG else 70,
}
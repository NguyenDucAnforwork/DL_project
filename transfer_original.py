from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import utils

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from scipy.optimize import fmin_l_bfgs_b

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

def get_image_arguments():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Get paths for content and style images.")

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
        "--epochs",
        type=int,
        default=5,
        required=False,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate the image paths
    if not os.path.isfile(args.content):
        raise FileNotFoundError(f"Image content not found at path: {args.content}")
    if not os.path.isfile(args.style):
        raise FileNotFoundError(f"Image style not found at path: {args.style}")

    # Return the arguments
    return args.content, args.style, args.epochs


def VGG19_AvgPool(shape):
    vgg = VGG19(input_shape=shape, weights='imagenet', include_top=False)


    i = vgg.input
    x = i
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # replace it with average pooling
            x = AveragePooling2D()(x)
        else:
            x = layer(x)

    return Model(i, x)


def gram_matrix(img):
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
    return G


def style_loss(y, t):
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def minimize(fn, epochs, batch_shape):
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(
            func=fn,
            x0=x,
            maxfun=20
        )
        x = np.clip(x, -127, 127)

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]


# load the content image
def load_img_and_preprocess(path, shape=None):
    img = utils.load_img(path, target_size=shape)

    # convert image to array and preprocess for vgg
    x = utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x

def transfer_original(content_img, style_img, epochs):
    # python style_transfer_original.py --content path/to/content.jpg --style path/to/style.jpg --epochs 5

    batch_shape = content_img.shape
    shape = content_img.shape[1:]

    vgg = VGG19_AvgPool(shape)
    content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
    content_target = K.variable(content_model.predict(content_img))
    symbolic_conv_outputs = [
        layer.get_output_at(1) for layer in vgg.layers \
        if layer.name.endswith('conv1')
    ]

    style_model = Model(vgg.input, symbolic_conv_outputs)

    style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

    style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]

    loss = K.mean(K.square(content_model.output - content_target))

    for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
        loss += w * style_loss(symbolic[0], actual[0])


    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var

    total_variation_weight = 0.2


    def total_variation_loss(image):
        x_deltas, y_deltas = high_pass_x_y(image)
        return (K.sum(K.abs(x_deltas)) + K.sum(K.abs(y_deltas))) / (127 ** 2)


    loss += total_variation_loss(content_model.output) * total_variation_weight

    # once again, create the gradients and loss + grads function
    # note: it doesn't matter which model's input you use
    # they are both pointing to the same keras Input layer in memory
    grads = K.gradients(loss, vgg.input)

    # just like theano.function
    get_loss_and_grads = K.function(
        inputs=[vgg.input],
        outputs=[loss] + grads
    )

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    final_img = minimize(get_loss_and_grads_wrapper, epochs, batch_shape)
    return final_img

if __name__ == "__main__":
    content_path, style_path, epochs = get_image_arguments()
    content_img = load_img_and_preprocess(
        content_path
    )
    h, w = content_img.shape[1:3]
    style_img = load_img_and_preprocess(
        style_path,
        (h, w)
    )
    final_img = scale_img(transfer_original(content_img, style_img, epochs))
    plt.imsave("output_nst_original.png", final_img)


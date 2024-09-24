import argparse
import logging

import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    Lambda,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam
from patchify import patchify, unpatchify
from tensorflow.keras.models import load_model

from evaluation_metrics import f1, iou

# U-Net model
# Author: Sreenivas Bhattiprolu


def build_unet_a(
    IMG_HEIGHT=256,
    IMG_WIDTH=256,
    IMG_CHANNELS=1,
    num_filters=None,
    dropout_rate=None,
    learning_rate=1e-3,
):
    """
    Build a U-Net model for image segmentation.

    :param IMG_HEIGHT: Height of the input images.
    :type IMG_HEIGHT: int, optional (default is 256)
    :param IMG_WIDTH: Width of the input images.
    :type IMG_WIDTH: int, optional (default is 256)
    :param IMG_CHANNELS: Number of channels in the input images.
    :type IMG_CHANNELS: int, optional (default is 1)
    :param num_filters: List of filter sizes for each convolutional layer.
    :type num_filters: list of int, optional
    :param dropout_rate: List of dropout rates for each layer.
    :type dropout_rate: list of float, optional
    :param learning_rate: Learning rate for the optimizer.
    :type learning_rate: float, optional (default is 1e-3)
    :return: Compiled U-Net model.
    :rtype: keras.models.Model
    :author: Michal Dziechciarz

    **Usage:**

    This function builds a U-Net model with configurable parameters for image segmentation tasks.

    **Example:**

    .. code-block:: python

        from your_module import build_model

        model = build_model(
            IMG_HEIGHT=256,
            IMG_WIDTH=256,
            IMG_CHANNELS=1,
            num_filters=[16, 32, 64, 128, 256],
            dropout_rate=[0.1, 0.1, 0.2, 0.2, 0.3],
            learning_rate=1e-3
        )

    **Details:**

    - Constructs a U-Net model with a specified number of filters and dropout rates for each layer.
    - The contraction path consists of several Conv2D, Dropout, and MaxPooling2D layers.
    - The expansive path consists of Conv2DTranspose and Conv2D layers, with concatenations from the contraction path.
    - The model is compiled with a specified learning rate, binary cross-entropy loss, and metrics including accuracy, F1 score, and IoU.
    """

    logging.info("\nBuilding U-Net model")
    if num_filters is None:
        num_filters = [16, 32, 64, 128, 256]  # Default filter sizes
    if dropout_rate is None:
        dropout_rate = [0.1, 0.1, 0.2, 0.2, 0.3]  # Default dropout rates

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path with dynamic filters and dropout
    c1 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(s)
    c1 = Dropout(dropout_rate[0])(c1)
    c1 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p1)
    c2 = Dropout(dropout_rate[1])(c2)
    c2 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p2)
    c3 = Dropout(dropout_rate[2])(c3)
    c3 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p3)
    c4 = Dropout(dropout_rate[3])(c4)
    c4 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        num_filters[4],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p4)
    c5 = Dropout(dropout_rate[4])(c5)
    c5 = Conv2D(
        num_filters[4],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c5)

    # Expansive path
    u6 = Conv2DTranspose(num_filters[3], (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u6)
    c6 = Dropout(dropout_rate[3])(c6)
    c6 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c6)

    u7 = Conv2DTranspose(num_filters[2], (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u7)
    c7 = Dropout(dropout_rate[2])(c7)
    c7 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c7)

    u8 = Conv2DTranspose(num_filters[1], (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u8)
    c8 = Dropout(dropout_rate[1])(c8)
    c8 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c8)

    u9 = Conv2DTranspose(num_filters[0], (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u9)
    c9 = Dropout(dropout_rate[0])(c9)
    c9 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", f1, iou]
    )

    return model


def build_unet_b(
    IMG_HEIGHT=256,
    IMG_WIDTH=256,
    IMG_CHANNELS=1,
    num_filters=None,
    dropout_rate=None,
    learning_rate=1e-3,
):
    logging.info("\nBuilding U-Net model")
    if num_filters is None:
        num_filters = [32, 64, 128, 256, 512]  # Default filter sizes
    if dropout_rate is None:
        dropout_rate = [0.2, 0.2, 0.3, 0.3, 0.4]  # Default dropout rates

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path with dynamic filters and dropout
    c1 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(s)
    c1 = Dropout(dropout_rate[0])(c1)
    c1 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p1)
    c2 = Dropout(dropout_rate[1])(c2)
    c2 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p2)
    c3 = Dropout(dropout_rate[2])(c3)
    c3 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p3)
    c4 = Dropout(dropout_rate[3])(c4)
    c4 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        num_filters[4],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(p4)
    c5 = Dropout(dropout_rate[4])(c5)
    c5 = Conv2D(
        num_filters[4],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c5)

    # Expansive path
    u6 = Conv2DTranspose(num_filters[3], (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u6)
    c6 = Dropout(dropout_rate[3])(c6)
    c6 = Conv2D(
        num_filters[3],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c6)

    u7 = Conv2DTranspose(num_filters[2], (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u7)
    c7 = Dropout(dropout_rate[2])(c7)
    c7 = Conv2D(
        num_filters[2],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c7)

    u8 = Conv2DTranspose(num_filters[1], (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u8)
    c8 = Dropout(dropout_rate[1])(c8)
    c8 = Conv2D(
        num_filters[1],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c8)

    u9 = Conv2DTranspose(num_filters[0], (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u9)
    c9 = Dropout(dropout_rate[0])(c9)
    c9 = Conv2D(
        num_filters[0],
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", f1, iou]
    )

    return model


def build_model(
    IMG_HEIGHT=256,
    IMG_WIDTH=256,
    IMG_CHANNELS=1,
    num_filters=None,
    dropout_rate=None,
    learning_rate=1e-3,
    architecture="unet_a",
):
    if architecture == "unet_a":
        return build_unet_a(
            IMG_HEIGHT=IMG_HEIGHT,
            IMG_WIDTH=IMG_WIDTH,
            IMG_CHANNELS=IMG_CHANNELS,
            num_filters=num_filters,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
    elif architecture == "unet_b":
        return build_unet_b(
            IMG_HEIGHT=IMG_HEIGHT,
            IMG_WIDTH=IMG_WIDTH,
            IMG_CHANNELS=IMG_CHANNELS,
            num_filters=num_filters,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
    else:
        logging.error("Invalid architecture")


def load_pretrained_model(model_path):
    """
    Load a pre-trained model with custom objects.

    :param model_path: Path to the pre-trained model file.
    :type model_path: str
    :return: Loaded pre-trained model.
    :rtype: keras.models.Model
    :author: Michal Dziechciarz

    **Usage:**

    This function loads a pre-trained model with custom objects (F1 score and IoU metrics).

    **Example:**

    .. code-block:: python

        from your_module import load_pretrained_model

        model = load_pretrained_model("models/primary.h5")

    **Details:**

    - Loads the model from the specified file path.
    - Ensures custom metrics (F1 score and IoU) are available during model loading.
    """

    custom_objects = {"f1": f1, "iou": iou}
    model = load_model(model_path, custom_objects=custom_objects)
    return model


# Another function to save the model, parameters are model and path
def save_model(model, path):
    """
    Save a trained model to a specified path.

    :param model: Trained model to be saved.
    :type model: keras.models.Model
    :param path: Path to save the model file.
    :type path: str
    :author: Michal Dziechciarz

    **Usage:**

    This function saves a trained Keras model to the specified file path.

    **Example:**

    .. code-block:: python

        from your_module import save_model

        save_model(trained_model, "models/my_model.h5")

    **Details:**

    - Saves the model in the specified path.
    - Prints a success message upon saving the model.
    """

    print("Model saved successfully in the path: ", path)
    logging.info("Model saved successfully in the path: ", path)
    model.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, help="Path to save the model")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--architecture", type=str, default="unet_a")

    args = parser.parse_args()

    model = build_model(
        learning_rate=args.learning_rate, architecture=args.architecture
    )

    save_model(model, args.model_path)

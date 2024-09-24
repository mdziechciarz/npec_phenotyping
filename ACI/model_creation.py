import argparse

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


def build_model(
    IMG_HEIGHT=256,
    IMG_WIDTH=256,
    IMG_CHANNELS=1,
    num_filters=None,
    dropout_rate=None,
    learning_rate=1e-3,
):
    print("\nBuilding U-Net model")
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


def load_pretrained_model(model_path):
    custom_objects = {"f1": f1, "iou": iou}
    model = load_model(model_path, custom_objects=custom_objects)
    return model


# Another function to save the model, parameters are model and path
def save_model(model, path):
    print("Model saved successfully in the path: ", path)
    model.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, help="Path to save the model")
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    args = parser.parse_args()

    model = build_model(learning_rate=args.learning_rate)

    save_model(model, args.model_path)

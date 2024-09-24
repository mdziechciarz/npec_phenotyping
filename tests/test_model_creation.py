import os
import sys

from keras.layers import Conv2D, Input
from keras.models import Model
from keras.optimizers import Adam

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from evaluation_metrics import f1, iou
from model_creation import (
    build_model,
    build_unet_a,
    build_unet_b,
    load_pretrained_model,
)


# Test build_model function
def test_build_model_default():
    model = build_model()
    assert isinstance(model, Model)
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 256, 256, 1)
    assert model.optimizer.learning_rate == 1e-3
    assert model.loss == "binary_crossentropy"


def test_build_model_custom():
    num_filters = [8, 16, 32, 64, 128]
    dropout_rate = [0.05, 0.05, 0.1, 0.1, 0.15]
    learning_rate = 1e-4
    model = build_model(
        IMG_HEIGHT=128,
        IMG_WIDTH=128,
        IMG_CHANNELS=3,
        num_filters=num_filters,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    assert isinstance(model, Model)
    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_shape == (None, 128, 128, 1)
    assert model.optimizer.learning_rate == learning_rate
    assert model.loss == "binary_crossentropy"


def test_build_unet_a():
    model = build_unet_a()
    assert isinstance(model, Model)
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 256, 256, 1)
    assert model.optimizer.learning_rate == 1e-3
    assert model.loss == "binary_crossentropy"

    assert model.layers[1].filters == 16


def test_build_unet_b():
    model = build_unet_b()
    assert isinstance(model, Model)
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 256, 256, 1)
    assert model.optimizer.learning_rate == 1e-3
    assert model.loss == "binary_crossentropy"

    assert model.layers[1].filters == 32


# Test successful model loading
def test_load_pretrained_model():
    model = load_pretrained_model("models/primary_model")

    assert isinstance(
        model, Model
    ), "The loaded model is not an instance of keras Model"
    assert "f1" in model.metrics_names, "Custom metric f1 not found in the model"
    assert "iou" in model.metrics_names, "Custom metric iou not found in the model"

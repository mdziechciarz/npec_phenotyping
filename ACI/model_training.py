import argparse

import keras.backend as K
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azureml.core import Workspace
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
from patchify import patchify, unpatchify

from data_loading import load_data_generators
from model_creation import build_model, load_pretrained_model


def train_model(
    model,
    train_generator,
    train_length,
    val_generator,
    val_length,
    epochs=100,
    batch_size=16,
    use_mlflow=False,
):
    if use_mlflow:
        mlflow.start_run()

    cb = EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights="True", mode="min"
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_length,
        epochs=epochs,
        callbacks=[cb],
        validation_data=val_generator,
        validation_steps=val_length,
        verbose=1,
    )

    print(history.history)

    if use_mlflow:
        # Log the model summary to MLflow
        mlflow.log_text("model_summary.txt", str(model.summary()))

        # Log the model metrics to MLflow
        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("train_f1", history.history["f1"][-1])
        mlflow.log_metric("train_iou", history.history["iou"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
        mlflow.log_metric("val_f1", history.history["val_f1"][-1])
        mlflow.log_metric("val_iou", history.history["val_iou"][-1])

        # plot the training and validation loss and accuracies for each epoch using matplotlib and log the image to MLflow
        fig = plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["accuracy"], label="train_accuracy")
        plt.plot(history.history["f1"], label="train_f1")
        plt.plot(history.history["iou"], label="train_iou")

        plt.title("Training Loss, Accuracy, F1 Score and IOU")
        plt.xlabel("Epoch #")
        plt.ylabel("Metrics")
        plt.legend(loc="lower left")
        mlflow.log_figure(fig, "train_metrics.png")

        fig = plt.figure()
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.plot(history.history["val_f1"], label="val_f1")
        plt.plot(history.history["val_iou"], label="val_iou")

        plt.title("Validation Loss, Accuracy, F1 Score and IOU")
        plt.xlabel("Epoch #")
        plt.ylabel("Metrics")
        plt.legend(loc="lower left")
        mlflow.log_figure(fig, "val_metrics.png")

        mlflow.end_run()

    return model, history


def hyperparameter_search(
    train_generator, train_length, val_generator, val_length, epochs=None
):

    hyperparameter_sets = [
        {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "dropout_rate": [0.1, 0.1, 0.2, 0.2, 0.3],
            "num_filters": [16, 32, 64, 128, 256],
            "epochs": 100,
        },
        {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "dropout_rate": [0.1, 0.1, 0.2, 0.2, 0.3],
            "num_filters": [32, 64, 128, 256, 512],
            "epochs": 120,
        },
        {
            "learning_rate": 5e-4,
            "batch_size": 16,
            "dropout_rate": [0.1, 0.2, 0.2, 0.2, 0.3],
            "num_filters": [16, 32, 64, 128, 256],
            "epochs": 150,
        },
        {
            "learning_rate": 2e-4,
            "batch_size": 32,
            "dropout_rate": [0.2, 0.2, 0.3, 0.3, 0.4],
            "num_filters": [16, 32, 64, 128, 256],
            "epochs": 100,
        },
    ]

    best_model = None
    best_history = None
    best_val_loss = float("inf")
    best_hyperparameters = None

    for params in hyperparameter_sets:
        epochs = epochs or params.get("epochs", 100)

        print(f"Training with hyperparameters: {params}")
        model = build_model(
            num_filters=params.get("num_filters", None),
            dropout_rate=params.get("dropout_rate", None),
            learning_rate=params.get("learning_rate", 1e-3),
        )
        model, history = train_model(
            model,
            train_generator,
            train_length,
            val_generator,
            val_length,
            epochs=epochs,
        )

        # Assume the best val_loss is stored in the history history.history['val_loss']
        min_val_loss = min(history.history["val_loss"])
        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_model = model
            best_history = history
            best_hyperparameters = params

    return best_model, best_history, best_hyperparameters


# Another function to save the model, parameters are model and path
def save_model(model, path):
    print("Model saved successfully in the path: ", path)
    model.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str, required=True)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-model-path", type=str, default=None)
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to save the model"
    )
    parser.add_argument(
        "--history-path", type=str, required=True, help="Path to save the history"
    )
    # Optional input model path
    # Change model-path to output-model-path

    args = parser.parse_args()

    if args.input_model_path:
        model = load_pretrained_model(args.input_model_path)
    else:
        model = build_model(learning_rate=args.learning_rate)

    (
        (train_generator, train_length),
        (test_generator, test_length),
        (val_generator, val_length),
    ) = load_data_generators(args.dataset_path, args.batch_size)

    model, history = train_model(
        model,
        train_generator,
        train_length,
        val_generator,
        val_length,
        epochs=100,
        batch_size=args.batch_size,
        use_mlflow=True,
    )

    save_model(model, args.model_path)

    np.save(args.history_path, history.history)

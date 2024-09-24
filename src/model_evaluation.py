import argparse
import glob
import json
import logging
import os

import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score

from evaluation_metrics import f1, iou
from model_creation import load_pretrained_model


def evaluate_model(model, test_dataset_path, use_mlflow=False):
    test_data_files = sorted(
        glob.glob(os.path.join(test_dataset_path, "images", "images", "*.png"))
    )
    test_masks_files = sorted(
        glob.glob(os.path.join(test_dataset_path, "masks", "masks", "*.tif"))
    )

    test_data = np.array(
        [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in test_data_files]
    )
    y_true = np.array(
        [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in test_masks_files]
    )

    y_pred = model.predict(test_data) > 0.5

    y_true = y_true.reshape(y_true.shape[0], 256, 256, 1)
    y_pred = y_pred.reshape(y_pred.shape[0], 256, 256, 1)

    iou_score = float(iou(K.variable(y_true), K.variable(y_pred)).numpy())
    accuracy = float(accuracy_score(y_true.flatten(), y_pred.flatten()))
    f1_score = float(f1(K.variable(y_true), K.variable(y_pred)).numpy())

    logging.info("IOU:", iou_score)
    logging.info("Accuracy:", accuracy)
    logging.info("F1:", f1_score)

    if use_mlflow:
        mlflow.start_run()
        # Visualize example images and predicted masks
        fig, ax = plt.subplots(2, 10, figsize=(30, 6))
        ax[0, 0].set_title("Images")
        ax[1, 0].set_title("Predictions")
        random_indices = np.random.choice(len(test_data), size=10, replace=False)
        for i, index in enumerate(random_indices):
            ax[0, i].imshow(test_data[index], cmap="gray")
            ax[0, i].axis("off")
            ax[1, i].imshow(y_pred[index].reshape(256, 256), cmap="gray")
            ax[1, i].axis("off")
        plt.tight_layout()
        mlflow.log_figure(fig, "example_outputs.png")

        mlflow.log_metric("test_iou", iou_score)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1_score)

        mlflow.end_run()

    return {
        "iou": iou_score,
        "accuracy": accuracy,
        "f1": f1_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--evaluation-results",
        type=str,
        required=True,
        help="Path to save the evaluation results",
    )

    args = parser.parse_args()

    # Load the model
    model = load_pretrained_model(args.model_path)

    # Evaluate the model
    evaluation_results = evaluate_model(
        model, os.path.join(args.dataset_path, "test"), use_mlflow=True
    )

    # Save results
    with open(os.path.join(args.evaluation_results, "metrics.json"), "w") as f:
        f.write(json.dumps(evaluation_results))

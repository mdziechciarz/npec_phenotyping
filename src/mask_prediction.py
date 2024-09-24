import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from patchify import patchify, unpatchify
from skan import Skeleton, draw, summarize
from skan.csr import skeleton_to_csgraph
from skimage.morphology import remove_small_objects, skeletonize
from tensorflow.keras import backend as K

# from tensorflow.keras.models import load_model
from model_creation import load_pretrained_model
from utils import padder


def predict_mask(model, input_img_path, patch_size=256):

    """
    Predict the segmentation mask for an input image using the provided model.

    :param model: The pre-trained model used for prediction.
    :type model: keras.models.Model
    :param input_img_path: Path to the input image.
    :type input_img_path: str
    :param patch_size: Size of the patches for model prediction.
    :type patch_size: int, optional (default is 256)
    :return: Predicted segmentation mask.
    :rtype: numpy.ndarray
    :author: Michal Dziechciarz

    **Usage:**

    This function predicts a segmentation mask for an input image using a pre-trained model.

    **Example:**

    .. code-block:: python

        from your_module import predict_mask, load_pretrained_model

        model = load_pretrained_model("models/primary.h5")
        mask = predict_mask(model, "path/to/my_image.png")

    **Details:**

    - Loads the input image in grayscale.
    - Extracts the region of interest (ROI) based on predefined coordinates and dimensions.
    - Pads the image to ensure dimensions are divisible by the patch size.
    - Splits the image into patches, performs predictions on each patch, and reassembles the patches into a full mask.
    - Applies a threshold to the predicted mask to binarize it.

    This function is useful for applying a pre-trained model to generate segmentation masks for input images.
    """

    image = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    # Extract ROI
    x, y, w, h = 740, 50, 2804, 2804
    image = image[y : y + h, x : x + w]

    image = padder(image, 256)

    patches = patchify(image, (patch_size, patch_size), step=patch_size)
    i = patches.shape[0]
    j = patches.shape[1]
    patches = patches.reshape(-1, patch_size, patch_size, 1)

    preds = model.predict(patches / 255)
    preds = preds.reshape(i, j, 256, 256)

    predicted_mask = unpatchify(preds, (image.shape[0], image.shape[1]))

    predicted_mask = predicted_mask > 0.35
    predicted_mask = predicted_mask.astype(np.uint8)

    return predicted_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-img-path", type=str, required=True)
    parser.add_argument("--output-mask-path", type=str, required=True)
    args = parser.parse_args()

    model = load_pretrained_model(args.model_path)
    mask = predict_mask(model, os.path.join(args.input_img_path, "img.png"))

    cv2.imwrite(os.path.join(args.output_mask_path, "mask.tif"), mask)

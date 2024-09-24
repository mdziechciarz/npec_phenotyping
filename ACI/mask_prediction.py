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


def predict_mask(model, image, patch_size=256):
    # Assuming image is already an array
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract ROI
    x, y, w, h = 740, 50, 2804, 2804
    roi_image = image[y : y + h, x : x + w]

    # Pad the image
    pad_height = patch_size - (roi_image.shape[0] % patch_size)
    pad_width = patch_size - (roi_image.shape[1] % patch_size)
    image_padded = np.pad(roi_image, ((0, pad_height), (0, pad_width)), mode='constant')

    patches = patchify(image_padded, (patch_size, patch_size), step=patch_size)
    i = patches.shape[0]
    j = patches.shape[1]
    patches = patches.reshape(-1, patch_size, patch_size, 1)

    preds = model.predict(patches / 255)
    preds = preds.reshape(i, j, patch_size, patch_size)

    predicted_mask = unpatchify(preds, image_padded.shape)
    predicted_mask = predicted_mask[:h, :w]

    # Convert to binary image
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    return predicted_mask
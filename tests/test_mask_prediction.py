import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from mask_prediction import predict_mask
from model_creation import load_pretrained_model
from utils import padder


def test_predict_mask():
    # Write the image to a temporary file (emulate input image path)
    input_img_path = "data/test_img.png"
    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    img = img[50 : 50 + 2804, 740 : 740 + 2804]
    img = padder(img, 256)

    model = load_pretrained_model("models/primary_model")

    # Call the function
    mask = predict_mask(model, input_img_path)

    # Check if the output mask is of the expected shape and type
    assert mask.shape == (img.shape[0], img.shape[1])
    assert mask.dtype == np.uint8
    assert np.all(np.logical_or(mask == 0, mask == 1))  # Ensure mask is binary

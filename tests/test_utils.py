import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import padder

def test_padder():
    # Test case 1: Check padding for an image with dimensions smaller than patch size
    image1 = np.zeros((100, 150, 3), dtype=np.uint8)  # Example image of size 100x150
    patch_size = 32
    padded_image1 = padder(image1, patch_size)
    assert padded_image1.shape == (128, 160, 3)  # Expected padded dimensions

    # Test case 2: Check padding for an image with dimensions larger than patch size
    image2 = np.zeros((200, 250, 3), dtype=np.uint8)  # Example image of size 200x250
    padded_image2 = padder(image2, patch_size)
    assert padded_image2.shape == (224, 256, 3)  # Expected padded dimensions

    # Test case 3: Check padding for a square image
    image3 = np.zeros((128, 128, 3), dtype=np.uint8)  # Example square image of size 128x128
    padded_image3 = padder(image3, patch_size)
    assert padded_image3.shape == (160, 160, 3)  # Expected padded dimensions

    # Test case 4: Check if padding preserves image content near edges
    # You may add more specific tests depending on your requirements

    # Example assertion (just for illustration, adjust as per your specific needs):
    assert np.all(padded_image3[:4, :4] == 0)  # Top-left corner should be black due to padding

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_preprocessing import extract_roi, preprocess_images


def test_extract_roi():
    # Load test image
    img = cv2.imread("data/test_img.png", 0)

    # Extract region of interest (ROI) from the test image
    roi, x, y, w, h = extract_roi(img)

    # Assert that the ROI is not None
    assert roi is not None

    # Assert that the ROI is a NumPy array
    assert isinstance(roi, np.ndarray)

    # Assert that the ROI is not equal to the original image
    assert not np.array_equal(roi, img)

    # Assert that the ROI is a square
    assert roi.shape[0] == roi.shape[1]


def test_preprocess_images():
    unprocessed_dataset_path = "tests/example_unprocessed_dataset"

    # Create a temporary directory and copy the example dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dataset_dir = os.path.join(tmp_dir, "dataset")
        # Copy contents of the example dataset to the temporary directory
        shutil.copytree(unprocessed_dataset_path, tmp_dataset_dir)

        # Preprocess the images in the example dataset
        preprocess_images(tmp_dataset_dir)

        # Assert that the "train", "test", and "val" directories were created
        assert os.path.exists(os.path.join(tmp_dataset_dir, "train"))
        assert os.path.exists(os.path.join(tmp_dataset_dir, "test"))
        assert os.path.exists(os.path.join(tmp_dataset_dir, "val"))

        # Assert that the "images" and "masks" directories were created in each subset
        assert os.path.exists(os.path.join(tmp_dataset_dir, "train/images/images"))
        assert os.path.exists(os.path.join(tmp_dataset_dir, "train/masks/masks"))
        assert os.path.exists(os.path.join(tmp_dataset_dir, "test/images/images"))
        assert os.path.exists(os.path.join(tmp_dataset_dir, "test/masks/masks"))
        assert os.path.exists(os.path.join(tmp_dataset_dir, "val/images/images"))
        assert os.path.exists(os.path.join(tmp_dataset_dir, "val/masks/masks"))

        # Assert that the images were preprocessed
        assert len(os.listdir(os.path.join(tmp_dataset_dir, "train/images/images"))) > 0
        assert len(os.listdir(os.path.join(tmp_dataset_dir, "train/masks/masks"))) > 0
        assert len(os.listdir(os.path.join(tmp_dataset_dir, "test/images/images"))) > 0
        assert len(os.listdir(os.path.join(tmp_dataset_dir, "test/masks/masks"))) > 0
        assert len(os.listdir(os.path.join(tmp_dataset_dir, "val/images/images"))) > 0
        assert len(os.listdir(os.path.join(tmp_dataset_dir, "val/masks/masks"))) > 0

        # Assert that the "images_raw" and "masks_raw" directories were deleted
        assert not os.path.exists(os.path.join(tmp_dataset_dir, "images_raw"))
        assert not os.path.exists(os.path.join(tmp_dataset_dir, "masks_raw"))

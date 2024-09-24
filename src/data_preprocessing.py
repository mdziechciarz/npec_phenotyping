import argparse
import glob
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split

from utils import padder


def extract_roi(image):
    """
    Extract the region of interest (ROI) from an image by identifying the largest contour,
    which is assumed to be the Petri dish, and cropping the image accordingly.

    :param image: Input image from which to extract the ROI.
    :type image: numpy.ndarray
    :return: A tuple containing the cropped ROI image and the coordinates (x, y) and dimensions (width, height) of the bounding box.
    :rtype: tuple[numpy.ndarray, int, int, int, int]
    :author: Neil Ross Daniel

    **Usage:**

    This function can be used to extract the region of interest (ROI) from an image by identifying the largest contour.

    **Example:**

    .. code-block:: python

        import cv2
        from your_module import extract_roi

        # Load an image
        image = cv2.imread('petri_dish_image.png', cv2.IMREAD_GRAYSCALE)

        # Extract the ROI
        roi, x, y, w, h = extract_roi(image)

        # Display the original image and the ROI
        cv2.imshow('Original Image', image)
        cv2.imshow('Extracted ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    **Details:**

    The function performs the following steps:

    1. Thresholds the input image to a binary image using Otsu's method.
    2. Finds contours in the binary image.
    3. Identifies the largest contour, assuming it to be the Petri dish.
    4. Calculates the bounding box of the largest contour.
    5. Crops the image to the bounding box with some padding to ensure the whole Petri dish is included.
    6. Adjusts the bounding box to get a square ROI based on the largest side.
    7. Crops the image to the adjusted ROI.

    The function returns a tuple containing:
    - `roi`: The cropped region of interest image.
    - `x`: The x-coordinate of the top-left corner of the bounding box.
    - `y`: The y-coordinate of the top-left corner of the bounding box.
    - `w`: The width of the bounding box.
    - `h`: The height of the bounding box.

    """

    # Threshold the image to binary using Otsu's method
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour in the image is the Petri dish
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box with some padding to ensure we
    # include the whole Petri dish
    padding = 15  # padding in pixels
    x, y, w, h = x - padding, y - padding, w + (padding * 2), h + (padding * 2)

    # Adjust to get a square ROI based on the largest side
    if w > h:
        y -= (w - h) // 2
        h = w
    else:
        x -= (h - w) // 2
        w = h

    # Crop the image to the ROI
    roi = image[y : y + h, x : x + w]

    return roi, x, y, w, h


patch_size = 256


def preprocess_images(
    dataset_path,
    scaling_factor=1,
    test_size=0.2,
    val_size=0.1,
    patch_size=256,
):
    """
    Preprocess raw images and masks by extracting regions of interest (ROI), resizing, patchifying,
    and distributing them into train, test, and validation subsets.

    :param dataset_path: Path to the dataset directory containing raw images and masks.
    :type dataset_path: str
    :param scaling_factor: Factor by which to scale the images and masks, defaults to 1 (no scaling).
    :type scaling_factor: float, optional
    :param test_size: Proportion of the dataset to include in the test split, defaults to 0.2.
    :type test_size: float, optional
    :param val_size: Proportion of the training set to include in the validation split, defaults to 0.1.
    :type val_size: float, optional
    :param patch_size: Size of the patches to divide the images and masks into, defaults to 256.
    :type patch_size: int, optional
    :author: Neil Ross Daniel

    **Usage:**

    This function can be used to preprocess and organize a dataset of raw images and masks for machine learning tasks.

    **Example:**

    .. code-block:: python

        from your_module import preprocess_images

        # Set dataset path
        dataset_path = '/path/to/dataset'

        # Preprocess the images
        preprocess_images(
            dataset_path,
            scaling_factor=0.5,
            test_size=0.2,
            val_size=0.1,
            patch_size=256
        )

    **Details:**

    The function performs the following steps:

    1. Collects all image file paths from the 'images_raw' directory.
    2. Splits the dataset into train, test, and validation sets.
    3. Creates the necessary output directories for train, test, and validation subsets.
    4. Distributes and processes images and their corresponding masks into the appropriate subsets:
       - Extracts the ROI from each image using the `extract_roi` function.
       - Resizes the images and masks.
       - Pads the images and masks to ensure they can be evenly divided into patches.
       - Scales the images and masks if a scaling factor is provided.
       - Divides the images and masks into patches of the specified size.
       - Saves the patches into the corresponding subset directories.
    5. Deletes the 'images_raw' and 'masks_raw' directories and their contents.

    The function returns `None`.

    """

    image_paths = glob.glob(f"{dataset_path}/images_raw/*.png")

    print(image_paths)

    # Split the dataset
    train_paths, test_paths = train_test_split(
        image_paths, test_size=test_size, random_state=42
    )
    train_paths, val_paths = train_test_split(
        train_paths, test_size=val_size / (1 - test_size), random_state=42
    )

    # Create output directories
    subsets = ["train", "test", "val"]
    for subset in subsets:
        os.makedirs(f"{dataset_path}/{subset}/images/images", exist_ok=True)
        os.makedirs(f"{dataset_path}/{subset}/masks/masks", exist_ok=True)

    def distribute_files(image_paths, subset):
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            img_name, _ = os.path.splitext(base_name)

            # Check for root mask existence
            root_mask_path = f"{dataset_path}/masks_raw/{img_name}_root_mask.tif"
            if not os.path.exists(root_mask_path):
                continue

            # Patchify image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img, x, y, w, h = extract_roi(img)
            img = cv2.resize(img, (2804, 2804))
            img = padder(img, patch_size)

            if scaling_factor != 1:
                img = cv2.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor)

            img_patches = patchify(img, (patch_size, patch_size), step=patch_size)
            img_patches = img_patches.reshape(-1, patch_size, patch_size, 1)

            for i, patch in enumerate(img_patches):
                cv2.imwrite(
                    f"{dataset_path}/{subset}/images/images/{img_name}_{i}.png",
                    patch,
                )

            # Patchify root mask
            mask = cv2.imread(root_mask_path, 0)
            mask = mask[y : y + h, x : x + w]
            mask = cv2.resize(mask, (2804, 2804))
            mask = padder(mask, patch_size)

            if scaling_factor != 1:
                mask = cv2.resize(mask, (0, 0), fx=scaling_factor, fy=scaling_factor)

            mask_patches = patchify(mask, (patch_size, patch_size), step=patch_size)
            mask_patches = mask_patches.reshape(-1, patch_size, patch_size, 1)

            for i, patch in enumerate(mask_patches):
                cv2.imwrite(
                    f"{dataset_path}/{subset}/masks/masks/{img_name}_root_mask_{i}.tif",
                    patch,
                )

    distribute_files(train_paths, "train")
    distribute_files(test_paths, "test")
    distribute_files(val_paths, "val")

    # Remove "images_raw" and "masks_raw" directories and their contents using
    # shutil.rmtree
    shutil.rmtree(f"{dataset_path}/images_raw", ignore_errors=True)
    shutil.rmtree(f"{dataset_path}/masks_raw", ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--preprocessed-dataset-path", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)

    args = parser.parse_args()

    # Copy the dataset to the output directory
    shutil.copytree(args.dataset_path, args.output_dataset_path)

    preprocess_images(args.output_dataset_path)

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
from patchify import patchify, unpatchify


def segment_instances(mask):
    """
    Segment instances in a given binary mask using morphological operations and the watershed algorithm.

    :param mask: Input binary mask where instances are to be segmented.
    :type mask: numpy.ndarray
    :return: A labeled mask where each instance has a unique label.
    :rtype: numpy.ndarray
    :author: Neil Ross Daniel

    **Usage:**

    This function can be used to segment individual instances from a binary mask using a combination of morphological operations and connected components analysis.

    **Example:**

    .. code-block:: python

        import cv2
        import numpy as np
        from your_module import segment_instances

        # Load a binary mask image
        mask = cv2.imread('binary_mask.png', cv2.IMREAD_GRAYSCALE)

        # Segment instances in the mask
        labeled_mask = segment_instances(mask)

        # Save or display the result
        cv2.imwrite('labeled_mask.png', labeled_mask)
        cv2.imshow('Labeled Mask', labeled_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    **Details:**

    The function performs the following steps:
    
    1. Applies morphological closing operation to the input mask using a rectangular structuring element.
    2. Detects connected components in the processed mask.
    3. Filters components based on defined width and height thresholds.
    4. Sorts the components by area and keeps the top 5 largest components.
    5. Checks if the centroids of the top components fall within expected plant locations.
    6. Labels the top components in the output image with unique colors.

    The function returns an output image where each segmented instance is labeled with a unique color.

    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Detect connectedComponents
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Create an output image that will hold the segmented plants considering width and height
    output_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Generate unique colors
    unique_colors = [tuple(np.random.randint(0, 256, 3)) for _ in range(1, retval)]

    detected_instances = 0

    # Define width and height thresholds
    min_width = 15  # minimum width of a plant
    min_height = 70  # minimum height of a plant
    max_height = 3000  # minimum height of a plant

    filtered_components = []
    top_components = []

    # Go through all detected components
    for i in range(1, retval):  # We start from 1 to ignore the background component
        # Extract the width and height of the bounding box of the component
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = width * height
        centroid_x = centroids[i][0]

        # If the component's width and height are larger than the thresholds, we consider it a plant
        if width >= min_width and max_height > height >= min_height:
            # output_image[labels == i] = unique_colors[i - 1]
            # detected_instances += 1
            filtered_components.append((i, area, centroid_x))

        # Sort the components by area in descending order and keep the top 5
        filtered_components.sort(key=lambda x: x[1], reverse=True)
        top_components = filtered_components[:5]

    # Initialize a list to hold the boolean values for plant locations
    expected_plant_locations = [390, 910, 1430, 1930, 2450]
    plant_location_status = [False, False, False, False, False]

    # Iterate over each plant location
    for plant_loc in expected_plant_locations:
        # Check each top component's centroid
        for _, _, centroid_x in top_components:
            # If the centroid is within ±200px of the plant location, mark it as True
            if abs(centroid_x - plant_loc) <= 200:
                plant_location_status[expected_plant_locations.index(plant_loc)] = True
                break  # No need to check other components if one is found

    if len(top_components) != plant_location_status.count(True):
        top_components = top_components[: plant_location_status.count(True)]

    # Draw top components
    for i, _, _ in top_components:
        output_image[labels == i] = unique_colors[i - 1]
        detected_instances += 1

    return output_image

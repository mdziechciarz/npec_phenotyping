import argparse
import json
import logging
import os

import cv2
import numpy as np
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

from instance_segmentation import segment_instances


def detect_landmarks(mask):
    """
    Detect landmarks such as primary root start and end points, and lateral root tips
    from the provided mask.

    :param mask: Binary mask representing the segmented instances.
    :type mask: numpy.ndarray
    :return: List of dictionaries containing the coordinates of primary root start and end points,
             and lateral root tips.
    :rtype: list[dict]
    :author: Rens van den Berg

    **Usage:**

    This function can be used to detect landmarks such as the primary root start and end points,
    and lateral root tips from a binary mask of segmented instances.

    **Example:**

    .. code-block:: python

        import cv2
        import numpy as np
        from your_module import detect_landmarks

        # Load a binary mask image
        mask = cv2.imread('binary_mask.png', cv2.IMREAD_GRAYSCALE)

        # Detect landmarks in the mask
        landmarks = detect_landmarks(mask)

        # Print detected landmarks
        for plant in landmarks:
            print("Primary Root Start:", plant["primary_root_start"])
            print("Primary Root End:", plant["primary_root_end"])
            print("Lateral Root Tips:", plant["l_root_tips"])

    **Details:**

    The function performs the following steps:

    1. Segments the instances in the input mask using the `segment_instances` function.
    2. Converts the segmented mask to a binary format.
    3. Applies skeletonization to the binary mask.
    4. Summarizes the skeleton data to extract root landmarks.
    5. Identifies and records the coordinates of primary root start and end points, and lateral root tips.
    6. Returns a list of dictionaries containing the detected landmarks for each plant.

    The function returns a list of dictionaries, where each dictionary contains:
    - `primary_root_start`: Coordinates of the primary root start point.
    - `primary_root_end`: Coordinates of the primary root end point.
    - `l_root_tips`: List of coordinates for the lateral root tips.

    """    
    segmented = segment_instances(mask)

    binary = segmented[:, :, 0]
    binary[binary != 0] = 1

    s = skeletonize(binary)

    if True not in np.unique(s):
        return []

    skeleton_data = summarize(Skeleton(s))
    skeleton_data = skeleton_data.reset_index()

    cords = []

    for i in range(skeleton_data["skeleton-id"].max() + 1):
        plant_skeleton = skeleton_data[skeleton_data["skeleton-id"] == i]

        primary_root_start_index = plant_skeleton[
            plant_skeleton["coord-src-0"] == plant_skeleton["coord-src-0"].min()
        ].index[0]
        primary_root_start_x = int(
            plant_skeleton.loc[primary_root_start_index]["coord-src-1"]
        )
        primary_root_start_y = int(
            plant_skeleton.loc[primary_root_start_index]["coord-src-0"]
        )

        primary_root_end_index = plant_skeleton[
            plant_skeleton["coord-dst-0"] == plant_skeleton["coord-dst-0"].max()
        ].index[0]

        primary_root_end_x = int(
            plant_skeleton.loc[primary_root_end_index]["coord-dst-1"]
        )

        primary_root_end_y = int(
            plant_skeleton.loc[primary_root_end_index]["coord-dst-0"]
        )

        l_root_tips = []
        lateral_root_tips = plant_skeleton[plant_skeleton["branch-type"] == 1]
        for index, row in lateral_root_tips.iterrows():
            if index != primary_root_start_index and index != primary_root_end_index:
                x1 = int(row["coord-src-1"])
                y1 = int(row["coord-src-0"])

                x2 = int(row["coord-dst-1"])
                y2 = int(row["coord-dst-0"])

                l_root_tips.append((x1, y1))
                l_root_tips.append((x2, y2))

        cords.append(
            {
                "primary_root_start": (primary_root_start_x, primary_root_start_y),
                "primary_root_end": (primary_root_end_x, primary_root_end_y),
                "l_root_tips": l_root_tips,
            }
        )

        cords = sorted(cords, key=lambda k: k["primary_root_start"][0])

    return cords


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect landmarks in a plant image")
    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to the image file",
    )
    parser.add_argument(
        "--landmarks-path",
        type=str,
        required=True,
        help="Path to save the detected landmarks",
    )

    args = parser.parse_args()

    # Load the mask
    mask = cv2.imread(os.path.join(args.mask_path, "mask.tif"), cv2.IMREAD_GRAYSCALE)

    landmarks = detect_landmarks(mask)

    # Save detected landmarks in a JSON file
    with open(os.path.join(args.landmarks_path, "landmarks.json"), "w") as file:
        json.dump(landmarks, file)

    logging.info(
        "Landmarks detected and saved successfully, path:",
        os.path.join(args.landmarks_path, "landmarks.json"),
    )

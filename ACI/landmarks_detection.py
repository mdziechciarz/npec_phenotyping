import argparse
import glob
import json
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
from tensorflow.keras.models import load_model

from instance_segmentation import segment_instances
from mask_prediction import predict_mask


def detect_landmarks(mask):
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
import argparse
import json
import logging
import os

import cv2
import networkx as nx
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

from instance_segmentation import segment_instances


def measure_root_lengths(mask):
    segmented = segment_instances(mask)
    binary = segmented[:, :, 0]
    binary[binary != 0] = 1

    s = skeletonize(binary)
    skeleton_data = summarize(Skeleton(s))
    skeleton_data = skeleton_data.reset_index()

    G = nx.from_pandas_edgelist(
        skeleton_data,
        source="node-id-src",
        target="node-id-dst",
        edge_attr="branch-distance",
    )

    plants = []

    for i in range(skeleton_data["skeleton-id"].max() + 1):
        plant_skeleton = skeleton_data[skeleton_data["skeleton-id"] == i]

        p_root_start_node_id = plant_skeleton[
            plant_skeleton["coord-src-0"] == plant_skeleton["coord-src-0"].min()
        ]["node-id-src"].values[0]

        p_root_end_node_id = plant_skeleton[
            plant_skeleton["coord-dst-0"] == plant_skeleton["coord-dst-0"].max()
        ]["node-id-dst"].values[0]

        p_root_length = nx.dijkstra_path_length(
            G, p_root_start_node_id, p_root_end_node_id, weight="branch-distance"
        )

        logging.info(
            f"Plant: {i}",
            f"Root length: {p_root_length}" "\n",
        )

        primary_root_start_index = plant_skeleton[
            plant_skeleton["coord-src-0"] == plant_skeleton["coord-src-0"].min()
        ].index[0]

        primary_root_start_x = int(
            plant_skeleton.loc[primary_root_start_index]["coord-src-1"]
        )

        lateral_roots = plant_skeleton[
            (plant_skeleton["branch-type"] == 1)
            & (plant_skeleton["node-id-src"] != p_root_start_node_id)
            & (plant_skeleton["node-id-dst"] != p_root_end_node_id)
        ]

        l_root_list = []

        for index, row in lateral_roots.iterrows():
            root_length = nx.dijkstra_path_length(
                G, row["node-id-src"], row["node-id-dst"], weight="branch-distance"
            )
            l_root_list.append(root_length)

        plants.append(
            {
                "x": primary_root_start_x,
                "p_root_length": p_root_length,
                "l_root_lengths": l_root_list,
            }
        )

    sorted_plants = sorted(plants, key=lambda plant: plant["x"])

    return sorted_plants


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    mask = cv2.imread(os.path.join(args.mask_path, "mask.tif"), cv2.IMREAD_GRAYSCALE)
    root_lengths = measure_root_lengths(mask)
    logging.info("Measured root lengths:", root_lengths)

    # Save the root lengths in JSON file
    with open(os.path.join(args.output_path, "root_lengths.json"), "w") as f:
        json.dump(root_lengths, f)

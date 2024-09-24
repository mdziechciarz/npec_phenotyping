import cv2
import numpy as np
import pytest
import sys
import os

# Update the sys.path to include the src directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from instance_segmentation import segment_instances


@pytest.fixture
def example_mask():
    # Load the test mask image
    mask_path = "data/test_mask.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask


def test_segment_instances(example_mask):
    # Call the function with the example mask
    markers = segment_instances(example_mask)
    
    # Check if the output is of the correct type
    assert isinstance(markers, (np.ndarray, np.generic)) or hasattr(markers, "__array__")

    # Check if the output has the correct shape (height, width, 3)
    assert markers.shape == (example_mask.shape[0], example_mask.shape[1], 3)

    # Check if the output contains multiple segments (excluding background)
    assert np.any(markers > 0)

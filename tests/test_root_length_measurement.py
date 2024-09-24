import cv2
import os
import sys
import pytest

# Adjust the path to include the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from root_length_measurement import measure_root_lengths

def test_measure_root_lengths():
    mask = cv2.imread("data/test_mask.png", cv2.IMREAD_GRAYSCALE)

    # Call the function with the mock mask1
    result = measure_root_lengths(mask)

    # Assert the result matches the expected output
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "x" in result[0]
    assert "p_root_length" in result[0]
    assert "l_root_lengths" in result[0]
    assert isinstance(result[0]["x"], int)
    assert isinstance(result[0]["p_root_length"], float or int)
    assert isinstance(result[0]["l_root_lengths"], list)
    assert isinstance(result[0]["l_root_lengths"][0], float or int)

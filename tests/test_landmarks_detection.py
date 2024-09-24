import unittest
import os
import cv2
import json
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from landmarks_detection import detect_landmarks

class TestDetectLandmarks(unittest.TestCase):
    
    def setUp(self):
        # Load test image
        self.test_image = cv2.imread('data/test_mask.png', cv2.IMREAD_GRAYSCALE)
        
        # Create a temporary directory for test output
        self.test_output_dir = 'tests/data/output'
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def tearDown(self):
        # Clean up: remove temporary test output directory and files
        import shutil
        shutil.rmtree(self.test_output_dir)
    
    def test_save_landmarks_to_json(self):
        # Call the detect_landmarks function
        landmarks = detect_landmarks(self.test_image)
    
        # Save detected landmarks to a JSON file
        test_landmarks_path = os.path.join(self.test_output_dir, 'test_landmarks.json')
        with open(test_landmarks_path, 'w') as f:
            json.dump(landmarks, f)
    
        # Assert that the JSON file was created and contains the expected data
        self.assertTrue(os.path.isfile(test_landmarks_path))
        with open(test_landmarks_path, 'r') as f:
            loaded_landmarks = json.load(f)
            self.assertEqual(len(loaded_landmarks), len(landmarks))
            for loaded, original in zip(loaded_landmarks, landmarks):
                self.assertEqual(tuple(loaded["primary_root_start"]), original["primary_root_start"])

if __name__ == '__main__':
    unittest.main()
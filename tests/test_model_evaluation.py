import os
import sys

# Add the '../src' directory to the system path to allow importing from the src folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from model_creation import load_pretrained_model
from model_evaluation import evaluate_model


def test_evaluate_model():
    """
    Test the evaluate_model function.

    This test loads a pretrained model, evaluates it on a test dataset,
    and checks that the results contain the expected metrics with the correct types.
    """
    # Load the pretrained model from the specified path
    model = load_pretrained_model("models/primary_model")

    # Evaluate the model on the test dataset
    results = evaluate_model(model, "data/testing_dataset/test")

    # Check that the results are returned as a dictionary
    assert isinstance(results, dict)

    # Check that the results dictionary contains the expected keys
    assert "iou" in results
    assert "accuracy" in results
    assert "f1" in results

    # Check that the values for each key in the results dictionary are floats
    assert isinstance(results["iou"], float)
    assert isinstance(results["accuracy"], float)
    assert isinstance(results["f1"], float)

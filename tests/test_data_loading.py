import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_loading import load_data_generators


def test_load_data_generators():
    dataset_path = "data/testing_dataset"
    batch_size = 16
    (
        (train_generator, train_length),
        (test_generator, test_length),
        (val_generator, val_length),
    ) = load_data_generators(dataset_path, batch_size)

    # Check the types of returned generators
    assert isinstance(train_generator, zip)
    assert isinstance(test_generator, zip)
    assert isinstance(val_generator, zip)

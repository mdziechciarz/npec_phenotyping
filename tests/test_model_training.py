import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_loading import load_data_generators
from model_creation import build_model
from model_training import hyperparameter_search, save_model, train_model


def test_train_model():
    # Test the train_model function
    # Create a simple model
    model = build_model()

    # Load small data generators
    (
        (train_generator, train_length),
        (test_generator, test_length),
        (val_generator, val_length),
    ) = load_data_generators("data/testing_dataset")

    # Train the model
    model, history = train_model(
        model,
        train_generator,
        train_length,
        val_generator,
        val_length,
        epochs=1,
    )
    # Check that the model is not None
    assert model is not None
    # Check that the history is not None
    assert history is not None
    # Check that the history has the correct keys
    assert "loss" in history.history
    assert "val_loss" in history.history


# Test the hyperparameter_search function
def test_hyperparameter_search():
    #     # Load small data generators
    (
        (train_generator, train_length),
        (test_generator, test_length),
        (val_generator, val_length),
    ) = load_data_generators("data/testing_dataset")

    best_model, best_history, best_hyperparameters = hyperparameter_search(
        train_generator, train_length, val_generator, val_length, epochs=1
    )

    # Assertions
    assert best_model is not None
    assert best_history is not None
    assert best_hyperparameters is not None

    # Check if the hyperparameters of the best model match one of the sets
    assert best_hyperparameters in [
        {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "dropout_rate": [0.1, 0.1, 0.2, 0.2, 0.3],
            "num_filters": [16, 32, 64, 128, 256],
            "epochs": 100,
        },
        {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "dropout_rate": [0.1, 0.1, 0.2, 0.2, 0.3],
            "num_filters": [32, 64, 128, 256, 512],
            "epochs": 120,
        },
        {
            "learning_rate": 5e-4,
            "batch_size": 16,
            "dropout_rate": [0.1, 0.2, 0.2, 0.2, 0.3],
            "num_filters": [16, 32, 64, 128, 256],
            "epochs": 150,
        },
        {
            "learning_rate": 2e-4,
            "batch_size": 32,
            "dropout_rate": [0.2, 0.2, 0.3, 0.3, 0.4],
            "num_filters": [16, 32, 64, 128, 256],
            "epochs": 100,
        },
    ]


def test_save_model():
    # Create a simple model
    model = build_model()

    # Save the model
    save_model(model, "tests/model")

    # Check if the model file exists
    assert os.path.exists("tests/model")

    # Remove the directory containing the model file
    shutil.rmtree("tests/model")


def test_script_execution():
    # Test the script execution

    os.system(
        "python src/model_training.py --dataset-path=data/testing_dataset --model-path=tests/example_model --history-path=tests/example_history --epochs=1 --no-mlflow"
    )

    # Check that the model file exists
    assert os.path.exists("tests/example_model")
    shutil.rmtree("tests/example_model")

    # Check that the history file exists
    assert os.path.exists("tests/example_history.npy")
    os.remove("tests/example_history.npy")

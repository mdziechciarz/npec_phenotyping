import os

import cv2
import typer

from data_loading import load_data_generators
from instance_segmentation import segment_instances
from landmarks_detection import detect_landmarks
from mask_prediction import predict_mask
from model_creation import build_model, load_pretrained_model
from model_training import hyperparameter_search, save_model, train_model
from root_length_measurement import measure_root_lengths


def train_new_model(
    new_model_name, custom_dataset_path=None, custom_hyperparameters=None
):
    # If custom dataset path is given, preprocess the images and load the data generators, else load the default dataset
    if custom_dataset_path:
        # preprocess_images(
        #     f"data/user_custom_datasets/{custom_dataset_path}/dataset_raw",
        #     f"data/user_custom_datasets/{custom_dataset_path}/dataset_patched",
        # )
        # print("Images preprocessed successfully")

        (
            (train_generator, train_length),
            (test_generator, test_length),
            (val_generator, val_length),
        ) = load_data_generators(f"data/user_custom_datasets/{custom_dataset_path}")
        print("Data generators loaded successfully")

    else:
        (
            (train_generator, train_length),
            (test_generator, test_length),
            (val_generator, val_length),
        ) = load_data_generators("data/testing_dataset")
        print("Data generators loaded successfully")

    # If no hyperparameters given, train the model with default hyperparameters, else perform hyperparameter search
    if custom_hyperparameters is None:
        print("Performing hyperparameter search")
        model, history, best_hypeparameters = hyperparameter_search(
            train_generator, train_length, val_generator, val_length
        )
        print(
            "Hyperparameter search completed and the best model is trained successfully.",
            "Best hyperparameters:",
            best_hypeparameters,
        )

    else:
        print("Training the model with custom hyperparameters:", custom_hyperparameters)
        model = build_model(
            num_filters=custom_hyperparameters["num_filters"],
            dropout_rate=custom_hyperparameters["dropout_rate"],
            learning_rate=custom_hyperparameters["learning_rate"],
        )
        model, history = train_model(
            model,
            train_generator,
            train_length,
            val_generator,
            val_length,
            epochs=custom_hyperparameters["epochs"],
        )
        print("Model trained successfully")

    print("Evaluating the model")
    # TODO evaluate_model(model, history, validation_data_generator)

    save_model(model, f"models/user_custom_models/{new_model_name}")
    print("Model saved successfully", f"model/user_custom_models/{new_model_name}")


def predict_using_custom_model(image_path, custom_model_name):
    # Load the model
    model = load_pretrained_model(f"models/user_custom_models/{custom_model_name}")

    # Predict the mask
    mask = predict_mask(model, image_path)
    segmentation_mask = segment_instances(mask)
    landmarks = detect_landmarks(mask)
    root_lengths = measure_root_lengths(mask)

    for i, plant in enumerate(landmarks):
        print()
        print(f"Plant {i}:")
        print(f"Primary root start: {plant['primary_root_start']}")
        print(f"Primary root end: {plant['primary_root_end']}")
        print(f"Lateral root tips: {plant['l_root_tips']}")
        print(f"Primary root length: {root_lengths[i]['p_root_length']}")

    cv2.imshow("Original Image", cv2.resize(cv2.imread(image_path), (960, 540)))
    cv2.imshow("Segmentation Mask", cv2.resize(segmentation_mask, (960, 540)))
    cv2.waitKey(0)


def predict_using_primary_model(image_path):
    # Load the model
    print("predicting using primary model")
    model = load_pretrained_model("models/primary_model")

    # Predict the mask
    mask = predict_mask(model, image_path)
    segmentation_mask = segment_instances(mask)
    landmarks = detect_landmarks(mask)
    root_lengths = measure_root_lengths(mask)

    for i, plant in enumerate(landmarks):
        print()
        print(f"Plant {i}:")
        print(f"Primary root start: {plant['primary_root_start']}")
        print(f"Primary root end: {plant['primary_root_end']}")
        print(f"Lateral root tips: {plant['l_root_tips']}")
        print(f"Primary root length: {root_lengths[i]['p_root_length']}")

    cv2.imshow("Original Image", cv2.resize(cv2.imread(image_path), (960, 540)))
    cv2.imshow("Segmentation Mask", cv2.resize(segmentation_mask, (960, 540)))
    cv2.waitKey(0)


app = typer.Typer()


@app.command()
def infer(img_path: str, custom_model_name: str = typer.Option(default=None)):
    print("Infering")
    print(custom_model_name)

    if custom_model_name:
        # Check if the user custom model exists
        if os.path.exists(f"models/user_custom_models/{custom_model_name}"):
            predict_using_custom_model(img_path, custom_model_name)
        else:
            print("Model does not exist.")
    else:
        predict_using_primary_model(img_path)


@app.command()
def train(
    new_model_name: str,
    custom_dataset_path: str = typer.Option(None, help="Path to the custom dataset"),
    learning_rate: float = typer.Option(None, help="Learning rate for training"),
    batch_size: int = typer.Option(None, help="Batch size for training"),
    dropout_rate: str = typer.Option(
        None, help="Dropout rates for training, comma-separated"
    ),
    num_filters: str = typer.Option(
        None, help="Number of filters for each layer, comma-separated"
    ),
    epochs: int = typer.Option(None, help="Number of epochs for training"),
):
    # Check if new_model_name is free
    if os.path.exists(f"models/user_custom_models/{new_model_name}"):
        print("Model name already exists.")
        return

    # # Check if the user custom dataset exists, if not exit
    # if custom_dataset_path and not os.path.exists(f"{custom_dataset_path}"):
    #     print("Dataset does not exist.")
    #     return

    # # Validate the dataset
    # if custom_dataset_path:
    #     print("Validating the dataset")
    #     validate_dataset(custom_dataset_path)

    # if none hyperparameters are given, hyperparameters = None
    if not any([learning_rate, batch_size, dropout_rate, num_filters, epochs]):
        custom_hyperparameters = None
    else:
        custom_hyperparameters = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout_rate": (
                [float(x) for x in dropout_rate.split(",")] if dropout_rate else None
            ),
            "num_filters": (
                [int(x) for x in num_filters.split(",")] if num_filters else None
            ),
            "epochs": epochs,
        }
    custom_hyperparameters = None

    train_new_model(new_model_name, custom_dataset_path, custom_hyperparameters)


if __name__ == "__main__":
    app()


# Automated Plant Phenotyper - Computer Vision Pipeline

This repository contains the code for a machine learning pipeline designed to process images, perform instance segmentation, and extract relevant features for further analysis.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Function Descriptions](#function-descriptions)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mdziechciarz/npec_phenotyping.git
    ```
2. Navigate to the project directory:
    ```bash
    cd 2023-24d-fai2-adsai-group-cv-3
    ```
3. Create a conda environment from the provided environment file:
    ```bash
    conda env create -f environment.yaml
    ```
4. Activate the environment:
    ```bash
    conda activate mark-processor
    ```

## Usage

### Running the FastAPI Server

To start the FastAPI server, run:
```bash
docker-compose up --build
```

### Data Preprocessing

To preprocess images, use the following command:
```bash
python src/data_preprocessing.py --dataset-path /path/to/dataset --preprocessed-dataset-path /path/to/output
```

### Training the Model

To train a new model, run:
```bash
python src/model_training.py --dataset-path /path/to/dataset --model-path /path/to/save/model --history-path /path/to/save/history
```

### Instance Segmentation

To segment instances, use:
```bash
python src/instance_segmentation.py --mask-path /path/to/mask --landmarks-path /path/to/save/landmarks
```

### Mask Prediction

To predict masks from images:
```bash
python src/mask_prediction.py --model-path /path/to/model --input-img-path /path/to/image --output-mask-path /path/to/save/mask
```

## Dependencies


This project requires the following dependencies as specified in the `environment.yaml` file:


```yaml
name: mark-processor
dependencies:
  - python>=3.8.13,<3.11.4
  - pandas>=2.2.2,<3.0.0
  - numpy>=1.26.4,<2.0.0
  - matplotlib>=3.8.4,<4.0.0
  - seaborn>=0.13.2,<0.14.0
  - scikit-learn>=1.4.2,<2.0.0
  - tensorflow>2.9,<2.16.1
  - opencv-python-headless>=4.7.0.72,<5.0.0.0
  - patchify>=0.2.3,<0.3.0
  - networkx>=3.1,<4.0
  - scikit-image>=0.21.0,<0.22.0
  - skan>=0.11.0,<0.12.0
  - typer>=0.9.0,<0.10.0
```

## Authorship

### src/APIS_components.py

- Description: Contains the main FastAPI application, API key security, and functions to call Azure ML components.

### src/data_loading.py

- Description: Contains functions to load data generators and validate the dataset.

### src/data_preprocessing.py

- Description: Contains functions to preprocess images, including extracting ROIs and patchifying images.

### src/evaluation_metrics.py

- Description: Contains custom evaluation metrics such as F1 score and IoU.

### src/instance_segmentation.py

- Description: Contains functions to segment instances and detect landmarks in plant images.

### src/main.py

- Description: Main script to run training and inference commands using Typer.

### src/mask_prediction.py

- Description: Contains functions to predict masks from images using a trained model.

### src/model_creation.py

- Description: Contains functions to build, save, and load the U-Net model.

### src/model_training.py

- Description: Contains functions for model training and hyperparameter search.

### src/root_length_measurement.py

- Description: Contains functions to measure root lengths from segmented masks.

## Function Descriptions

### FastAPI Endpoints (src/APIS_components.py)
- `/call_component/{component_name}`: Calls a specific Azure ML component by name.
- `/root_length_measurement`: Calls the root length measurement component.
- `/landmarks_detection`: Calls the landmarks detection component.
- `/mask_prediction`: Calls the mask prediction component.
- `/data_preprocessing`: Calls the data preprocessing component.
- `/model_registration`: Calls the model registration component.
- `/model_evaluation`: Calls the model evaluation component.
- `/model_training`: Calls the model training component.

### Data Loading (src/data_loading.py)
- `load_data_generators(dataset_path)`: Loads data generators for training, testing, and validation.
- `validate_dataset(dataset_path)`: Validates the dataset to ensure it has the required structure.

### Data Preprocessing (src/data_preprocessing.py)
- `extract_roi(image)`: Extracts the region of interest from an image.
- `preprocess_images(dataset_path, scaling_factor, test_size, val_size, patch_size)`: Preprocesses images by patchifying and distributing them into training, testing, and validation sets.

### Evaluation Metrics (src/evaluation_metrics.py)
- `f1(y_true, y_pred)`: Computes the F1 score.
- `iou(y_true, y_pred)`: Computes the Intersection over Union (IoU).

### Instance Segmentation (src/instance_segmentation.py)
- `detect_landmarks(mask)`: Detects landmarks in a segmented mask.

### Mask Prediction (src/mask_prediction.py)
- `predict_mask(model, input_img_path, patch_size)`: Predicts a mask from an input image using a trained model.

### Model Creation (src/model_creation.py)
- `build_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_filters, dropout_rate, learning_rate)`: Builds the U-Net model.
- `load_pretrained_model(model_path)`: Loads a pretrained model.
- `save_model(model, path)`: Saves the model to the specified path.

### Model Training (src/model_training.py)
- `train_model(model, train_generator, train_length, val_generator, val_length, epochs)`: Trains the model.
- `hyperparameter_search(train_generator, train_length, val_generator, val_length, epochs)`: Performs hyperparameter search for the best model configuration.

### Root Length Measurement (src/root_length_measurement.py)
- `measure_root_lengths(mask)`: Measures root lengths from a segmented mask.


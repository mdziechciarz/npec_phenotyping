# Azure ML Training Pipeline Overview

## High-Level Functions Flowchart
      +----------------------------------------+
      |               Azure ML                 |
      |            Training Pipeline           |
      +----------------------------------------+
            |           |               |
            v           v               v
      +-------------+ +-------------+ +------------------+
      | Data        | | Model       | | Model Training   |
      |Preprocessing| | Creation    | |   Component:     |
      |   Component:| | Component:  | |                  |
      |             | |             | | train_model()    |
      |             | |             | | hyperparameter_  |
      |             | |             | | search()         |
      |             | |             | | save_model()     |
      |extract_roi()| |build_model()| | load_pretrained_ |
      | preprocess_ | |save_model() | | model()          |
      | images()    | |             | +------------------+
      +-------------+ +-------------+            |
            |           |                        v
            v           v                     +------------------+
      +--------------+ +-------------------+  | Model Evaluation |
      | Mask         | | Landmarks         |  |   Component:     |
      | Prediction   | | Detection         |  |                  |
      | Component:   | | Component:        |  | evaluate_model() |
      |predict_mask()| |detect_landmarks() |  | f1()             |
      |              | |segment_instances()|  | iou()            |
      +--------------+ +-------------------+  +------------------+
            |                   |                        |
            v                   v                        v
      +-----------------------+ +----------------+ +------------------+
      | Root Length           | | Model          | |Model Registration|
      | Measurement           | | Registration   | |   Component:     |
      | Component:            | | Component:     | |                  |
      |measure_root_lengths() | |register_model()| |                  |
      |                       | |                | | register_if_IOU_ |
      |                       | |                | | above_threshold()|
      +-----------------------+ +----------------+ +------------------+



## Low-Level Functions Detail

### Data Preprocessing Component
- **Functions Used:**
  - `extract_roi()` (from `data_processing.py`)
  - `preprocess_images()` (from `data_processing.py`)

### Data Loading Component
- **Functions Used:**
  - `load_data_generators()` (from `data_loading.py`)
  - `validate_dataset()` (from `data_loading.py`)

### Model Creation Component
- **Functions Used:**
  - `build_model()` (from `model_creation.py`)
  - `save_model()` (from `model_creation.py`)
  - `load_pretrained_model()` (from `model_creation.py`)
  - `f1()` (from `evaluation_metrics.py`)
  - `iou()` (from `evaluation_metrics.py`)

### Instance Segmentation Component
- **Functions Used:**
  - `segment_instances()` (from `instance_segmentation.py`)

### Landmarks Detection Component
- **Functions Used:**
  - `detect_landmarks()` (from `landmark_detection.py`)
  - `segment_instances()` (from `instance_segmentation.py`)
  - `predict_mask()` (from `mask_prediction.py`)

### Model Training Component
- **Functions Used:**
  - `train_model()` (from `model_training.py`)
  - `hyperparameter_search()` (from `model_training.py`)
  - `save_model()` (from `model_training.py`)
  - `load_data_generators()` (from `data_loading.py`)
  - `build_model()` (from `model_creation.py`)
  - `load_pretrained_model()` (from `model_creation.py`)

### Mask Prediction Component
- **Functions Used:**
  - `predict_mask()` (from `mask_prediction.py`)
  - `load_pretrained_model()` (from `model_creation.py`)
  - `padder()` (from `utils.py`)

### Model Evaluation Component
- **Functions Used:**
  - `evaluate_model()` (from `model_evaluation.py`)
  - `f1()` (from `evaluation_metrics.py`)
  - `iou()` (from `evaluation_metrics.py`)
  - `load_pretrained_model()` (from `model_creation.py`)

### Model Registration Component
- **Functions Used:**
  - `register_model()` (from `model_registration.py`)
  - `register_if_IOU_above_threshold()` (from `model_registration.py`)

### Root Length Measurement Component
- **Functions Used:**
  - `measure_root_lengths()` (from `root_length_measurement.py`)
  - `segment_instances()` (from `instance_segmentation.py`)
  - `predict_mask()` (from `mask_prediction.py`)


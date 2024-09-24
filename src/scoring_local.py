import base64
from io import BytesIO
import json
import os
import numpy as np
from PIL import Image
import cv2
from patchify import patchify, unpatchify
from model_creation import load_pretrained_model
from landmarks_detection import detect_landmarks


def padder(image, patch_size):
    pad_h = (patch_size - image.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - image.shape[1] % patch_size) % patch_size
    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return image


def predict_mask(model, input_img_path, patch_size=256):
    image = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    # Extract ROI
    x, y, w, h = 740, 50, 2804, 2804
    image = image[y : y + h, x : x + w]

    image = padder(image, 256)

    patches = patchify(image, (patch_size, patch_size), step=patch_size)
    i = patches.shape[0]
    j = patches.shape[1]
    patches = patches.reshape(-1, patch_size, patch_size, 1)

    preds = model.predict(patches / 255)
    preds = preds.reshape(i, j, 256, 256)

    predicted_mask = unpatchify(preds, (image.shape[0], image.shape[1]))

    predicted_mask = predicted_mask > 0.35
    predicted_mask = predicted_mask.astype(np.uint8)

    return predicted_mask


def init():
    # Define the model as a global variable to be used later in the predict function
    global model

    # Get the path where the model is saved
    base_path = os.getenv("AZUREML_MODEL_DIR")
    print(f"base_path: {base_path}")

    # show the files in the model_path directory
    # print(f"list files in the model_path directory")
    # list files and dirs in the model_path directory
    list_files(base_path)

    # add the model file name to the base_path
    model_path = os.path.join(base_path, 'primary_model')  # local
    # model_path = os.path.join(base_path, "INPUT_model", 'model.keras') # azure
    # print the model_path to check if it is correct
    print(f"model_path: {model_path}")

    # Load the model
    model = load_pretrained_model(model_path)
    print("Model loaded successfully")


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))


def run(raw_data):
    try:
        # Log the raw data
        print("Raw data received:", raw_data)

        # Parse the JSON data
        data = json.loads(raw_data)
        print("Parsed data:", data)

        # Check if 'data' key exists
        if 'data' not in data:
            return json.dumps({"error": "'data' key not found in input data"})

        # Decode the image
        image_data = base64.b64decode(data['data'])
        print("Base64 decoded image data")

        # Load the image with PIL
        image = Image.open(BytesIO(image_data)).convert("L")
        image = np.array(image)

        # Save the image temporarily to pass the file path to the predict_mask function
        temp_image_path = "/tmp/temp_image.png"
        cv2.imwrite(temp_image_path, image)

        # Predict the mask
        predicted_mask = predict_mask(model, temp_image_path, patch_size=256)
        landmarks = detect_landmarks(predicted_mask)

        # Convert the predicted mask to an image
        predicted_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))

        # Convert the image to base64
        buffered = BytesIO()
        predicted_mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return json.dumps({"mask": mask_base64, "landmarks": landmarks})
    except Exception as e:
        error_message = str(e)
        print("Error:", error_message)
        return json.dumps({"error": error_message})

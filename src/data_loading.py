import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data_type = "root", "occluded_root", "seed", "shoot" or "background"


def load_data_generators(dataset_path, batch_size=16):

    """
    Load data generators for training, validation, and testing datasets.

    :param dataset_path: Path to the dataset directory.
    :type dataset_path: str
    :param batch_size: Size of the data batches, defaults to 16.
    :type batch_size: int, optional
    :return: A tuple containing generators and lengths for training, testing, and validation datasets.
    :rtype: tuple
    :author: Neil Ross Daniel

    **Usage:**

    This function can be used to load data generators for images and masks in training, validation, and testing datasets.

    **Example:**

    .. code-block:: python

        from your_module import load_data_generators

        # Set dataset path
        dataset_path = '/path/to/dataset'

        # Load data generators
        (train_gen, train_len), (test_gen, test_len), (val_gen, val_len) = load_data_generators(dataset_path, batch_size=16)

        # Iterate through the training generator
        for (train_images, train_masks) in train_gen:
            # Your training code here
            pass

    **Details:**

    The function performs the following steps:

    1. Creates an image data generator for training images with rescaling and horizontal flipping.
    2. Creates an image data generator for training masks with horizontal flipping.
    3. Creates a zip generator for training images and masks.
    4. Creates an image data generator for test images with rescaling.
    5. Creates an image data generator for test masks.
    6. Creates a zip generator for test images and masks.
    7. Creates an image data generator for validation images with rescaling.
    8. Creates an image data generator for validation masks.
    9. Creates a zip generator for validation images and masks.

    The function returns a tuple containing three tuples:
    - (train_generator, train_length)
    - (test_generator, test_length)
    - (val_generator, val_length)

    Each generator is a zip object combining image and mask data. The lengths indicate the number of batches available in each generator.

    """


    # Training images
    train_image_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)

    train_image_generator = train_image_datagen.flow_from_directory(
        f"{dataset_path}/train/images",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=42,
    )

    # Training masks
    train_mask_datagen = ImageDataGenerator(horizontal_flip=True)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        f"{dataset_path}/train/masks",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=42,
    )

    train_generator = zip(train_image_generator, train_mask_generator)
    train_length = len(train_image_generator)

    # Test images
    test_image_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_image_generator = test_image_datagen.flow_from_directory(
        f"{dataset_path}/test/images",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=42,
    )

    # Test masks
    test_mask_datagen = ImageDataGenerator()

    test_mask_generator = test_mask_datagen.flow_from_directory(
        f"{dataset_path}/test/masks",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=42,
    )

    test_generator = zip(test_image_generator, test_mask_generator)
    test_length = len(test_image_generator)

    # Validation images
    val_image_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_image_generator = val_image_datagen.flow_from_directory(
        f"{dataset_path}/val/images",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=42,
    )

    # Validation masks
    val_mask_datagen = ImageDataGenerator()

    val_mask_generator = val_mask_datagen.flow_from_directory(
        f"{dataset_path}/val/masks",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=42,
    )

    val_generator = zip(val_image_generator, val_mask_generator)
    val_length = len(val_image_generator)

    return (
        (train_generator, train_length),
        (test_generator, test_length),
        (val_generator, val_length),
    )


# Validate that dataset has subfolders "images" and "masks" with subfolders "train", "test" and "val"
# Each subfolders can't be empty
def validate_dataset(dataset_path):
    """
    Validate the structure and contents of the dataset directory.

    :param dataset_path: Path to the dataset directory.
    :type dataset_path: str
    :raises AssertionError: If the dataset path or any required subfolder does not exist,
                            or if any required subfolder is empty.
    :author: Neil Ross Daniel

    **Usage:**

    This function can be used to ensure that the dataset directory contains the required structure
    and is not empty.

    **Example:**

    .. code-block:: python

        from your_module import validate_dataset

        # Set dataset path
        dataset_path = '/path/to/dataset'

        # Validate the dataset structure and contents
        try:
            validate_dataset(dataset_path)
            print("Dataset is valid.")
        except AssertionError as e:
            print(f"Dataset validation failed: {e}")

    **Details:**

    The function performs the following steps:

    1. Checks if the dataset path exists.
    2. Checks if the 'images' and 'masks' subfolders exist within the dataset path.
    3. Checks if the 'train', 'test', and 'val' subfolders exist within both the 'images' and 'masks' subfolders.
    4. Ensures that each of the 'train', 'test', and 'val' subfolders is not empty.

    The function raises an `AssertionError` if any of these checks fail, providing a descriptive error message.

    """
    assert os.path.exists(dataset_path), "Dataset path does not exist"

    for subfolder in ["images", "masks"]:
        assert os.path.exists(
            f"{dataset_path}/{subfolder}"
        ), f"{subfolder} folder does not exist"

        for dataset_type in ["train", "test", "val"]:
            assert os.path.exists(
                f"{dataset_path}/{subfolder}/{dataset_type}"
            ), f"{dataset_type} folder does not exist"
            assert (
                len(os.listdir(f"{dataset_path}/{subfolder}/{dataset_type}")) > 0
            ), f"{dataset_type} folder is empty"

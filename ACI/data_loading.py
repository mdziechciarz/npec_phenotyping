import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data_type = "root", "occluded_root", "seed", "shoot" or "background"


def load_data_generators(dataset_path, batch_size=16):
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

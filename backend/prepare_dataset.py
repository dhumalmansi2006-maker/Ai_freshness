import os
import shutil
import random


# ==========================================
# PATHS
# ==========================================

# Run this script from the main FreshnessDetection folder
SOURCE_DIR = "dataset/raw/Processed Data"
DATASET_DIR = "dataset"


# ==========================================
# DATASET SPLIT
# ==========================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================================
# RANDOM SEED
# ==========================================

# Keeps the same images in each split every time
random.seed(42)


# ==========================================
# CLASS MAPPING
# ==========================================

# Dataset folder names → our model classes

classes = {
    "fresh": "fresh",
    "semi": "semi_fresh",
    "rotten": "spoiled"
}


# ==========================================
# CREATE DESTINATION FOLDERS
# ==========================================

for split in ["train", "validation", "test"]:

    for class_name in classes.values():

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        os.makedirs(folder, exist_ok=True)


# ==========================================
# CHECK SOURCE FOLDER
# ==========================================

if not os.path.exists(SOURCE_DIR):

    print("ERROR: Dataset source folder was not found.")
    print()
    print("Expected location:")
    print(os.path.abspath(SOURCE_DIR))
    print()
    print("Please check that:")
    print("dataset/raw/Processed Data/")
    print("exists inside your FreshnessDetection project.")

    exit()


# ==========================================
# PROCESS EACH FOLDER
# ==========================================

for folder_name in os.listdir(SOURCE_DIR):

    folder_path = os.path.join(
        SOURCE_DIR,
        folder_name
    )

    # Ignore files
    if not os.path.isdir(folder_path):
        continue


    # Convert folder name to lowercase
    folder_lower = folder_name.lower()


    # ======================================
    # IDENTIFY CLASS
    # ======================================

    if folder_lower.startswith("fresh"):

        target_class = "fresh"

    elif folder_lower.startswith("semi"):

        target_class = "semi_fresh"

    elif folder_lower.startswith("rotten"):

        target_class = "spoiled"

    else:

        print("Skipped unknown folder:", folder_name)

        continue


    # ======================================
    # FIND IMAGE FILES
    # ======================================

    images = []

    for file_name in os.listdir(folder_path):

        if file_name.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ):

            images.append(file_name)


    # If no images found
    if len(images) == 0:

        print(
            f"{folder_name}: No images found. Skipped."
        )

        continue


    # ======================================
    # SHUFFLE IMAGES
    # ======================================

    random.shuffle(images)


    # ======================================
    # CALCULATE SPLITS
    # ======================================

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = train_end + int(
        total * VALIDATION_RATIO
    )


    train_images = images[:train_end]

    validation_images = images[
        train_end:validation_end
    ]

    test_images = images[
        validation_end:
    ]


    # ======================================
    # COPY TRAIN IMAGES
    # ======================================

    for image in train_images:

        source = os.path.join(
            folder_path,
            image
        )

        # Add folder name to avoid duplicate filenames
        new_name = f"{folder_name}_{image}"

        destination = os.path.join(
            DATASET_DIR,
            "train",
            target_class,
            new_name
        )

        shutil.copy2(
            source,
            destination
        )


    # ======================================
    # COPY VALIDATION IMAGES
    # ======================================

    for image in validation_images:

        source = os.path.join(
            folder_path,
            image
        )

        new_name = f"{folder_name}_{image}"

        destination = os.path.join(
            DATASET_DIR,
            "validation",
            target_class,
            new_name
        )

        shutil.copy2(
            source,
            destination
        )


    # ======================================
    # COPY TEST IMAGES
    # ======================================

    for image in test_images:

        source = os.path.join(
            folder_path,
            image
        )

        new_name = f"{folder_name}_{image}"

        destination = os.path.join(
            DATASET_DIR,
            "test",
            target_class,
            new_name
        )

        shutil.copy2(
            source,
            destination
        )


    # ======================================
    # DISPLAY RESULT
    # ======================================

    print(
        f"{folder_name}: "
        f"{total} images → "
        f"Train: {len(train_images)}, "
        f"Validation: {len(validation_images)}, "
        f"Test: {len(test_images)}"
    )


# ==========================================
# FINISHED
# ==========================================

print()
print("=" * 50)
print("Dataset preparation completed!")
print("=" * 50)
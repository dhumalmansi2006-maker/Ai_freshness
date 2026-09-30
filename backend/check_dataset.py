import os


DATASET_DIR = "dataset"

splits = ["train", "validation", "test"]
classes = ["fresh", "semi_fresh", "spoiled"]


print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)


total_images = 0


for split in splits:

    print(f"\n{split.upper()}")

    split_total = 0

    for class_name in classes:

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        if os.path.exists(folder):

            count = len([
                file
                for file in os.listdir(folder)
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                )
            ])

        else:
            count = 0

        print(f"{class_name:12} : {count}")

        split_total += count

    print(f"Total {split}: {split_total}")

    total_images += split_total


print("\n" + "=" * 50)
print(f"TOTAL IMAGES: {total_images}")
print("=" * 50)

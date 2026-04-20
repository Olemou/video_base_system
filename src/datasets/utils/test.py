import os
import csv
import glob
from pathlib import Path


def rename_and_create_csv(
    image_folder,
    output_csv="folder_images_2.csv",
    prefix="spill",
    excelfileName="datasets/folderImage2/spill"
):
    image_folder = Path(image_folder)

    # Get all images
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]

    images = []
    for ext in image_extensions:
        images.extend(image_folder.glob(ext))

    images = sorted(images)

    if not images:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(images)} images")

    csv_data = []

    for idx, old_path in enumerate(images, start=1):
        ext = old_path.suffix.lower()

        # ✅ SAFE filename (no path inside name)
        new_filename = f"{prefix}_{idx:06d}{ext}"
        print(f"Processing: {old_path.name} -> {new_filename}")
        new_path = image_folder / new_filename

        # ✅ FIX: avoid FileExistsError
        if new_path.exists():
            os.remove(new_path)

        # rename safely
        os.replace(old_path, new_path)

        print(f"Renamed: {old_path.name} -> {new_filename}")

        # label logic
        label = 0 if "no_spill" in new_filename else 1
        
        excel = f"{excelfileName}_{idx:06d}{ext}"
        csv_data.append([str(excel), label])

    # write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(csv_data)

    print(f"\n✅ CSV file created: {output_csv}")
    print(f"Total samples: {len(csv_data)}")

    print("\nFirst 5 entries:")
    for i in range(min(5, len(csv_data))):
        print(f"  {csv_data[i][0]} {csv_data[i][1]}")


# Usage
if __name__ == "__main__":
    path = r"C:\Users\efo6780\Documents\folderImage81"

    rename_and_create_csv(
        image_folder=path,
        output_csv="folder_images_81.csv",
        prefix="no_spill",
        excelfileName="datasets/folderImage81/no_spill"
    )
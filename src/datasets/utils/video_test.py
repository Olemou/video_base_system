import os
import csv
from pathlib import Path


def rename_and_create_video_csv(
    video_folder,
    output_csv="folder_videos.csv",
    prefix="spill",
    excelfileName="datasets/folderVideo/spill"
):
    video_folder = Path(video_folder)

    # video extensions
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv"]

    videos = []
    for ext in video_extensions:
        videos.extend(video_folder.glob(ext))

    videos = sorted(videos)

    if not videos:
        print(f"No videos found in {video_folder}")
        return

    print(f"Found {len(videos)} videos")

    csv_data = []

    for idx, old_path in enumerate(videos, start=1):
        ext = old_path.suffix.lower()

        # safe new filename
        new_filename = f"{prefix}_{idx:06d}{ext}"
        new_path = video_folder / new_filename

        print(f"Processing: {old_path.name} -> {new_filename}")

        # avoid overwrite conflict
        if new_path.exists():
            os.remove(new_path)

        # rename video
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
    print(f"Total videos: {len(csv_data)}")

    print("\nFirst 5 entries:")
    for i in range(min(5, len(csv_data))):
        print(f"  {csv_data[i][0]} {csv_data[i][1]}")


# =========================
# Usage
# =========================
if __name__ == "__main__":
    path = r"C:\Users\efo6780\Documents\folderVideo81"

    rename_and_create_video_csv(
        video_folder=path,
        output_csv="folder_videos_81.csv",
        prefix="no_spill",
        excelfileName="datasets/folderVideo81/no_spill"
    )
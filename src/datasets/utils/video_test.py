import os
import csv
import cv2
from pathlib import Path


def video_to_frames_and_csv_15fps(
    video_path,
    output_folder,
    output_csv="video_frames_15fps.csv",
    prefix="spill",
    excelfileName="datasets/videoFrames/spill",
    target_fps=15
):
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        print("Could not read FPS, defaulting to 30")
        video_fps = 30

    # how many frames to skip to achieve ~15 FPS
    frame_interval = max(1, round(video_fps / target_fps))

    print(f"Video FPS: {video_fps}")
    print(f"Target FPS: {target_fps}")
    print(f"Frame interval: {frame_interval}")

    frame_idx = 0
    saved_idx = 0
    csv_data = []

    print(f"Processing video: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # keep only 15 FPS equivalent frames
        if frame_idx % frame_interval != 0:
            continue

        saved_idx += 1

        ext = ".jpg"
        new_filename = f"{prefix}_{saved_idx:06d}{ext}"
        new_path = output_folder / new_filename

        cv2.imwrite(str(new_path), frame)

        label = 0 if "no_spill" in prefix else 1

        excel_path = f"{excelfileName}_{saved_idx:06d}{ext}"
        csv_data.append([excel_path, label])

        print(f"Saved: {new_filename}")

    cap.release()

    # write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(csv_data)

    print(f"\n✅ Done!")
    print(f"Frames saved: {len(csv_data)}")
    print(f"CSV created: {output_csv}")

    print("\nFirst 5 entries:")
    for i in range(min(5, len(csv_data))):
        print(csv_data[i])


# =========================
# Usage
# =========================
if __name__ == "__main__":
    video_to_frames_and_csv_15fps(
        video_path=r"C:\Users\efo6780\Documents\video81.mp4",
        output_folder=r"C:\Users\efo6780\Documents\videoFrames81",
        output_csv="video_frames_81_15fps.csv",
        prefix="no_spill",
        excelfileName="datasets/videoFrames81/no_spill",
        target_fps=15
    )
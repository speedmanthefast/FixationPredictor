from vaODV import vaODV
import os

# --- Configuration ---
# 1. Path to the folder containing your MP4 videos
ODV_FOLDER = '/home/speedman/Projects/School Programs/Computer Science Project/Machine Learning Test/videos/ambix/'

# 2. Path to the folder containing your user gaze data
USER_DATASET = '/home/speedman/Projects/School Programs/Computer Science Project/Machine Learning Test/dataset_public/'

# 3. The modality you want to process (must match the subfolder in USER_DATASET)
MODALITY = 'ambix'
# ---------------------

def main():
    """
    This function initializes the analyzer and runs the fixation generation process.
    """
    print("Initializing the ODV analyzer...")
    # Create an instance of the vaODV class with your folder configuration
    analyzer = vaODV(odv_folder=ODV_FOLDER, user_dataset=USER_DATASET, modality=MODALITY)

    # The script automatically finds the video subfolders in your user data path
    # We can get the names from the odv_list it creates.
    video_names = [os.path.basename(p.strip('/')) for p in analyzer.odv_list]

    if not video_names:
        print("Error: No video data found! Check your folder paths and structure.")
        return

    print(f"Found {len(video_names)} videos to process: {video_names}")

    # Loop through each video and generate its fixation maps
    for i, name in enumerate(video_names):
        print(f"\n--- Processing video {i+1}/{len(video_names)}: {name} ---")
        analyzer.generate_fixations(name)
        print(f"--- Finished processing {name} ---")

    print("\n✅ Analysis complete! Check the 'output/fixations' folder for your results.")


if __name__ == "__main__":
    main()

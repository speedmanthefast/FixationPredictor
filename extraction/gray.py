import shutil

# This just copies the file because it will be converted to gray scale later anyways
def extract_gray(input_video_path, output_path, verbose=False):

    try:
        shutil.copy(input_video_path, output_path)
        verbose and print(f"Successfully moved '{input_video_path}' to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Source file '{input_video_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

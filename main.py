from predictor import FixationPredictor
from features import extract_single_video
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Runs the fixation predictor. Assumes features.py has been run in the same CWD and that the folder 'dataset/' now exists")
    parser.add_argument("--save", action="store_true", help="Forces a save of the model weights")
    parser.add_argument("--load", action="store_true", help="Loads existing model weights from CWD")
    parser.add_argument("--infer", default=None, help="Runs inference on a specific video after weights have been obtained")
    args = parser.parse_args()

    save = args.save
    load = args.load
    infer = args.infer

    # Choose which features to load (does not effect generation)
    active_features = {'frequency', 'surprise', 'people', 'vehicles', 'animals', 'handheld', 'edges', 'motion', 'gray', 'xgrad', 'ygrad'}

    fixation = FixationPredictor(active_features)

    if load:
        fixation.load()
    else:
        fixation.loadData("dataset")
        fixation.fit()
        fixation.test()

    if infer is not None:

        basename = os.path.basename(infer)
        video_id, _ = os.path.splitext(basename)
        datapath = os.path.join("./dataset/", video_id, "features/")

        extract_single_video(infer, output_dir=datapath)
        fixation.infer(datapath, output_file=f"{video_id}_prediction.mp4")

    if save:
        fixation.save()


if __name__ == "__main__":
    main()


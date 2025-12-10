from predictor import FixationPredictor
from features import extract_single_video
import argparse


def main():
    parser = argparse.ArgumentParser(description="Runs the fixation predictor")
    parser.add_argument("--save", action="store_true", help="Forces a save of the model weights")
    parser.add_argument("--load", action="store_true", help="Loads existing model weights from CWD")
    parser.add_argument("--infer", nargs='?', const="./video.mp4", default=None, help="Runs inference on a specific video after weights have been obtained")
    args = parser.parse_args()

    save = args.save
    load = args.load
    infer = args.infer

    # Choose which features to load (does not effect generation)
    active_features = {'people', 'vehicles', 'animals', 'handheld', 'nonhandheld', 'frequency', 'surprise', 'edges', 'motion', 'gray', 'xgrad', 'equator'}

    fixation = FixationPredictor(active_features)

    if load:
        fixation.load()
    else:
        fixation.loadData("dataset")
        fixation.fit()
        fixation.test()

    if infer is not None:
        extract_single_video(infer, "./inference/")
        fixation.infer("./inference/")

    if save:
        fixation.save()


if __name__ == "__main__":
    main()

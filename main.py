from predictor import FixationPredictor


def main():
    fixation = FixationPredictor("dataset")
    fixation.fit()
    fixation.infer("./dataset/1An41lDIJ6Q/features")


if __name__ == "__main__":
    main()

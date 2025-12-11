Fixation predicton using machine learning

Created with Python 3.13.7 and Pip 25.2
The fixations originally used in this project were generated using this gitub code
https://github.com/cozcinar/360_Audio_Visual_ICMEW2020


# HOW TO RUN

## Setup venv and install dependencies

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Populate dataset directory

1. Ensure your CWD is in the same directory as the dataset directory and the directory with your input videos
2. Fill a folder with the videos you would like to train on
3. Run `python3 features.py <video_dir>`
4. Wait for the scripts to generate the features (this will take a while). You can enable and disable specific feature generation by commenting/uncommenting the lines in wrap_features() in features.py

## Train the model

1. Once you have your features generated into ./dataset/ you need to make sure to add the targets into a folder called 'target' that is next to the folders called 'features' for each video.  It is expected that these targets are images, not videos.
3. You can change which features are active in main.py. They should have the same names as in features.py.
2. Run `python3 main.py --save` which will begin loading the data and then training the model.

## Use the model

1. The weights will be saved to 'model.pth'
2. You can now run inference using `python3 main.py --load --infer <path to video>`. NOTE: if you try to load weights for a model trained on a different architecture (such as different features active), it will crash the program
3. This will extract features manually if they don't already exist in ./dataset/
4. Prediction will be output to CWD.

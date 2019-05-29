# Computervision-Project

Indoor Location Tracking in Museums by Painting Detection and Matching

### Requirements

- opencv-contrib-python==3.4.2.16
- numpy==1.16.2

Or simply run:

```bash
pip install -r requirements.txt
```

## Usage

```
python main.py COMMAND [ARGS]

Commands:

    build [IMG_DIR]
        Build database from folder (recursively)


    eval_corners [IMG_DIR] [GROUND_TRUTH]
        Evaluate corner detection (IoU) based on ground truth csv

        Optional arguments:
            -o [OUTPUT_DIR] Path to output directory


    eval_hall [IMG_DIR] [GROUND_TRUTH]
        Evaluate hall prediction based on ground truth csv

        Optional arguments:
            -o [OUTPUT_DIR] Path to output directory


    infer [VIDEO]
        See the model in action on a video

        Optional arguments:
            -m [GROUND_TRUTH] csv with ground truth per  frame idx  to measure the accuracy
            -r [ROOM_FILE]    csv file with the rooms of the museum and how they are connected
            --gopro           If passed as an argument, then the infer function will treat the video file as a gopro video
            -s, --silent      Do not show any images, only output to commandline
            -p MAP            Passes an image that represents the map of the building
            -t [COORDS]       Passes a csv file that represents the coordinates of all the rooms present on the map of the building


Arguments:

    -v                    set logging to WARNING (default: CRITICAL)
    -v -v                 set logging to INFO    (default: CRITICAL)
    -v -v -v              set logging to DEBUG   (default: CRITICAL)
    --config, -c [CONFIG] Path to hparams file
```

For example,

```bash
python main.py build images/ -v -v
python main.py infer videos/MSK_01.mp4 -v -v
```

# Project_Computervisie

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```
python main.py COMMAND [ARGS]

Commands:

build [IMG_DIR]                 Build database from folder (recursively)
eval  [IMG_DIR] [GROUND_TRUTH]  Evaluate corner detection based on ground truth csv
infer [VIDEO]                   See the model in action on a video

Options:

-v       set logging to WARNING (default: CRITICAL)
-v -v    set logging to INFO    (default: CRITICAL)
-v -v -v set logging to DEBUG   (default: CRITICAL)
```

For example,

```bash
python main.py build images/ -v -v
```

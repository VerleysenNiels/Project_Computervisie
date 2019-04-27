# Project_Computervisie

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To build the database of images, run

```
python main.py build [IMAGE_FOLDER]

Options:

-v     set logging to INFO (default: WARNING)
-v -v  set logging to DEBUG (default: WARNING)
```

For example,

```bash
python main.py build images/ -v -v
```

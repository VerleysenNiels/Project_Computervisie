# Project_Computervisie

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```
python main.py COMMAND [ARGS]

Commands:

build [DIRECTORY]  Build database from folder (recursively)
eval               Evaluate classifier on database

Options:

-v     set logging to INFO (default: WARNING)
-v -v  set logging to DEBUG (default: WARNING)
```

For example,

```bash
python main.py build images/ -v -v
```

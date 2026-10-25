# MMA3001 YOLOv8 source code package

These files are adapted from the uploaded development notebook. They are not a replacement for the notebook or the dataset.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run inference

```bash
python -m src.predict --weights path/to/best.pt --image path/to/image.jpg
```

## Evaluate the selected model on the independent test set

```bash
python -m src.evaluate --weights path/to/best.pt --data path/to/data.yaml --split test --imgsz 800
```

## Train the selected configuration (optional, expensive)

```bash
python -m src.train --data path/to/data.yaml --output runs
```

## Benchmark

```bash
python -m src.benchmark --weights path/to/best.pt --images-dir path/to/test/images
```

## Tests

```bash
python -m pytest -q
```

## HTML documentation

```bash
python -m pdoc -o docs src
```

## Notes

- The final configuration follows the notebook's 50-epoch, 800-pixel, batch-16 AdamW experiment with initial learning rate 0.0001 and seed 42.
- The checkpoint and dataset are not included. Provide them separately for model inference and evaluation.
- The independent test results should not be confused with validation results.
- Test functions check the Python interface; they do not establish industrial defect-detection accuracy.

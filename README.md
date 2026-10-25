# MMA3001 — Pork Packaging Defect Detection Using YOLOv8

## Project Overview

This project investigates the application of machine learning and computer vision to automated quality control in pork packaging. The objective is to develop, optimise and evaluate a YOLOv8n object detection model capable of identifying and localising packaging defects.

The project was developed for MMA3001 at Monash University and focuses on computational implementation, numerical performance evaluation and optimisation rather than physical deployment in a food processing facility.

## Engineering Problem

Manual inspection of food packaging can be labour-intensive, time-consuming and susceptible to human error. Packaging defects may compromise product integrity and contribute to inconsistent quality control.

The proposed computational solution uses object detection to identify potential defects in digital images of packaged pork products. The intended application is to assist quality control personnel by highlighting packaging abnormalities that may require further inspection.

The model is a computational proof of concept and has not been validated for autonomous industrial quality-control decisions.

## Dataset and Defect Classes

The dataset contains images of packaged pork products with YOLO-format bounding-box annotations.

Five defect categories are considered:

1. `loose-meat`
2. `packaging-error`
3. `twisted-meat`
4. `unsealed`
5. `wrinkle`

Dataset split:

| Dataset | Images |
|---|---:|
| Training | 1,764 |
| Validation | 115 |
| Testing | 76 |

Annotations use the format:

`class_id x_center y_center width height`

Bounding-box coordinates and dimensions are normalised between 0 and 1.

The dataset is not included in this repository. Users must supply an appropriately structured dataset and update the dataset configuration path before training or evaluation.

## Model and Optimisation

The computational implementation uses the Ultralytics YOLOv8n object detection architecture.

The initial baseline model was trained using 10 epochs, a batch size of 16 and an image resolution of 640 × 640 pixels.

Further experiments investigated training duration, image resolution, data augmentation and learning rate.

The selected final configuration was:

| Parameter | Value |
|---|---|
| Architecture | YOLOv8n |
| Training epochs | 50 |
| Image resolution | 800 × 800 pixels |
| Batch size | 16 |
| Optimiser | AdamW |
| Initial learning rate | 0.0001 |
| Random seed | 42 |

The final configuration was selected using validation performance and subsequently evaluated using the independent test dataset.

## Model Performance

### Validation results

| Metric | Final model |
|---|---:|
| Precision | 0.8070 |
| Recall | 0.6679 |
| mAP@50 | 0.7397 |
| mAP@50–95 | 0.3811 |

### Independent test results

| Metric | Recreated baseline | Final model |
|---|---:|---:|
| Precision | 0.4536 | 0.5070 |
| Recall | 0.6604 | 0.5797 |
| mAP@50 | 0.5978 | 0.5371 |
| mAP@50–95 | 0.2543 | 0.2143 |

Although the final configuration improved validation performance relative to the original baseline, its independent test mAP was lower than that of the recreated baseline. This demonstrates the importance of evaluating generalisation rather than relying exclusively on validation metrics.

## Computational Performance

Computational profiling and benchmarking were performed to investigate the relationship between detection performance and computational requirements.

For the selected model at 800 × 800 resolution, an inference benchmark on an NVIDIA L4 GPU produced approximately:

- Mean inference latency: 11.823 ms/image
- Throughput: 84.58 images/second
- Peak PyTorch GPU memory allocation: 235.86 MB

These results are hardware- and benchmark-dependent and should not be interpreted as guaranteed performance on other devices.

## Repository Structure

```text
MMA3001-Pork-Packaging-Defect-Detection/
├── notebooks/
│   └── GITHUB (1).ipynb
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── benchmark.py
├── tests/
│   ├── test_predict.py
│   └── test_evaluate.py
├── README.md
├── LICENSE
├── requirements.txt
└── pytest.ini
```

The original Colab notebook documents the experimental workflow, while the Python modules provide reusable functionality for training, prediction, evaluation and benchmarking.

## Installation

Python and the required packages must be installed before running the source code.

Clone the repository:

```bash
git clone https://github.com/ibig0002-ops/MMA3001-Pork-Packaging-Defect-Detection.git
cd MMA3001-Pork-Packaging-Defect-Detection
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

The dataset and trained model checkpoint are not bundled with the repository.

## Running the Model

### Object detection

Run inference on an image using a trained YOLOv8 checkpoint:

```bash
python -m src.predict --weights path/to/best.pt --image path/to/image.jpg
```

### Model evaluation

Evaluate the trained model on the independent test dataset:

```bash
python -m src.evaluate --weights path/to/best.pt --data path/to/data.yaml --split test --imgsz 800
```

### Model training

Reproduce the selected training configuration:

```bash
python -m src.train --data path/to/data.yaml --output runs
```

Training requires the dataset and suitable computational resources.

### Benchmarking

Measure inference performance using images from a specified directory:

```bash
python -m src.benchmark --weights path/to/best.pt --images-dir path/to/test/images
```

## Automated Testing

Automated tests are implemented using `pytest`.

Run the tests from the repository's root directory:

```bash
python -m pytest -q
```

The tests examine selected aspects of the Python implementation and output handling. They do not independently verify the accuracy of the trained model or its suitability for industrial deployment.

## HTML Documentation

HTML source-code documentation can be generated using `pdoc`:

```bash
python -m pip install pdoc
python -m pdoc -o docs src
```

This generates HTML documentation based on the Python modules and their docstrings.

## Limitations

The model is limited to the five defect categories represented in its training data. Dataset imbalance and limited examples of certain defect classes restrict the reliability of class-specific performance estimates.

In particular, the validation and testing datasets contain no annotated wrinkle defects, preventing meaningful independent evaluation of that category.

Model performance may also vary with changes in lighting, packaging appearance, camera position and operating conditions. These variations have not been comprehensively evaluated in an industrial environment.

The system is intended to support further research and visual inspection, not replace established food-safety verification procedures.

## AI Assistance

ChatGPT was used as a supporting tool for code organisation, debugging, interpretation of computational methods and documentation. The MMA3001 lecture materials and independently obtained experimental results informed the methodology and evaluation. AI-generated material was subject to review and verification.

## Licence

This repository includes an MIT licence for the original project code. The Ultralytics YOLO framework and any third-party datasets, model weights or dependencies remain subject to their respective licensing terms.

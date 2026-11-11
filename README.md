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

## Dataset Access

The dataset used in this project consists of images of packaged pork products annotated with YOLO-format bounding boxes across five defect categories.

## Dataset source: [Download the original dataset](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging)

The dataset is not included directly in this GitHub repository due to its size. Users wishing to reproduce the training and evaluation procedures must download the dataset from the source linked above.

Following download, the dataset should be extracted and organised into the appropriate training, validation and testing directories, with the corresponding paths specified in the data.yaml configuration file.

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
├── models/
│   └── best.pt
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

## Running the Trained Model in Google Colab

The following instructions provide a complete workflow for running the trained YOLOv8 object detection model in Google Colab. The repository includes the final trained model weights (models/best.pt), allowing users to perform object detection and evaluate model performance using the publicly available Roboflow dataset without retraining the model or accessing the developer's Google Drive.


### Step 1 — Open Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **New notebook**.
3. Select **Runtime → Change runtime type → CPU**.

CPU execution is sufficient for testing the trained model on individual images. A GPU may be selected for faster processing.

### Step 2 — Clone the GitHub repository

Run the following code in a new Colab cell:

```python
!git clone https://github.com/ibig0002-ops/MMA3001-Pork-Packaging-Defect-Detection.git
%cd /content/MMA3001-Pork-Packaging-Defect-Detection
```

This downloads the repository, including the Python source code and trained model weights (`models/best.pt`).

Verify that the weights are available:

```python
from pathlib import Path

assert Path("models/best.pt").is_file(), "Model weights not found!"
print("Trained model weights found.")
```

### Step 3 — Install dependencies

```python
!pip install -r requirements.txt
```

This installs the Python packages required to execute the object detection model.

### Step 4 — Download and upload the dataset

The model was developed using the publicly available **Pork Rasher Error-Packaging** dataset hosted on Roboflow.

Dataset: [Pork Rasher Error-Packaging — Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging)

1. Open the dataset link above.
2. Select the dataset version used for this project.
3. Download the dataset in **YOLOv8 format** as a ZIP file.
4. Return to Google Colab.
5. Open the **Files** sidebar using the folder icon.
6. Upload the downloaded ZIP file to `/content/`.
7. Wait until the upload is fully complete before proceeding.

The dataset does not need to be uploaded to GitHub.

### Step 5 — Extract the dataset

Run the following code:

```python
from pathlib import Path
import zipfile

zip_files = list(Path("/content").glob("*.zip"))

assert len(zip_files) == 1, (
    "Expected exactly one dataset ZIP file in /content"
)

zip_path = zip_files[0]
assert zipfile.is_zipfile(zip_path), (
    "Invalid ZIP file. Check that the upload is complete."
)

dataset_dir = Path("/content/pork_dataset")

with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(dataset_dir)

print("Dataset extracted successfully!")
```

The extraction may create an additional folder inside `pork_dataset`, depending on how the dataset ZIP was generated.

### Step 6 — Locate a test image

The following code automatically locates the test images, including when the dataset is stored inside an additional folder. It excludes hidden macOS metadata files that cannot be processed as images.

```python
from pathlib import Path

dataset_root = Path("/content/pork_dataset")

test_dirs = [
    p for p in dataset_root.rglob("images")
    if p.parent.name == "test"
    and "__MACOSX" not in p.parts
]

assert test_dirs, "No valid test image directory found!"

test_dir = test_dirs[0]

test_images = sorted([
    p for p in test_dir.iterdir()
    if p.is_file()
    and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    and not p.name.startswith(".")
])

assert test_images, "No valid test images found!"

test_image = str(test_images[0])

print("Selected test image:", test_image)
```

### Step 7 — Run object detection

Execute the repository's prediction script using the trained model weights:

```python
!python -m src.predict \
    --weights models/best.pt \
    --image "{test_image}"
```

The script loads the trained YOLOv8 model and performs object detection on the selected image.

Predictions are returned with the following information:

- `class_id`: Numerical identifier of the detected defect.
- `class_name`: Predicted packaging defect category.
- `confidence`: Confidence score associated with the detection.
- `xyxy_pixels`: Bounding-box coordinates in pixels, represented as `[x_min, y_min, x_max, y_max]`.

For example, the output may resemble:

```text
{
    'class_id': 3,
    'class_name': 'unsealed',
    'confidence': 0.8026,
    'xyxy_pixels': [0.96, 355.23, 335.91, 540.0]
}
```

The exact predictions depend on the selected image.

### Step 8 — Configure the dataset for evaluation

Locate the Roboflow dataset configuration file:

```python
from pathlib import Path

yaml_files = [
    p for p in Path("/content/pork_dataset").rglob("data.yaml")
    if "__MACOSX" not in p.parts
]

assert len(yaml_files) == 1, (
    "Expected exactly one dataset configuration file"
)

dataset_yaml = yaml_files[0]
print("Dataset configuration:", dataset_yaml)
```

Update the dataset paths to reference the extracted folders:

```python
import yaml

with open(dataset_yaml) as f:
    config = yaml.safe_load(f)

dataset_folder = dataset_yaml.parent.resolve()

config["path"] = str(dataset_folder)
config["train"] = "train/images"
config["val"] = "valid/images"
config["test"] = "test/images"

for split in ["train", "valid", "test"]:
    assert (dataset_folder / split / "images").is_dir(), (
        f"Missing {split} image directory"
    )

with open(dataset_yaml, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print("Dataset configuration updated.")
```

This step ensures that the evaluation script can locate the training, validation and testing datasets regardless of the extraction location.

### Step 9 — Evaluate model performance

Run the repository's evaluation script on the test dataset:

```python
!python -m src.evaluate \
    --weights models/best.pt \
    --data "{dataset_yaml}" \
    --split test \
    --imgsz 800
```

The script evaluates the trained model using the annotated test images and reports performance metrics including precision, recall, mAP@50 and mAP@50–95.

For meaningful comparison with the results presented in the project report, use the same Roboflow dataset version and evaluation settings.

### Step 10 — Run object detection on another image (optional)

To evaluate a different image, replace `test_image` with the path to another supported `.jpg`, `.jpeg` or `.png` file.

For example:

```python
custom_image = "/content/example.jpg"

!python -m src.predict \
    --weights models/best.pt \
    --image "{custom_image}"
```

The image must exist at the specified location. Users can upload their own images through the Colab Files sidebar.

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

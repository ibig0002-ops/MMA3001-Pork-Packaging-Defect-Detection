"""Train the YOLOv8n pork-packaging defect detector.

Configuration follows the final AdamW learning-rate experiment in the Colab notebook.
"""
import argparse
from pathlib import Path


def train_model(data: str, output: str = 'runs', learning_rate: float = 0.0001):
    """Train YOLOv8n on a YOLO-format dataset and return training results.

    Args:
        data: Path to dataset data.yaml.
        output: Directory for Ultralytics training outputs.
        learning_rate: Initial AdamW learning rate (default: selected 0.0001).
    """
    # Fail early if the YOLO dataset configuration is missing.
    if not Path(data).is_file():
        raise FileNotFoundError(f'Dataset configuration not found: {data}')
    # Import Ultralytics only when training is requested.
    from ultralytics import YOLO
    # Initialise the nano-sized YOLOv8 model from pretrained weights.
    model = YOLO('yolov8n.pt')
    # Use the selected 50-epoch, 800-pixel configuration with AdamW optimisation.
    return model.train(data=data, epochs=50, imgsz=800, batch=16,
                       optimizer='AdamW', lr0=learning_rate, seed=42,
                       degrees=10, translate=0.1, scale=0.2,
                       hsv_h=0.01, hsv_s=0.3, hsv_v=0.2, mosaic=0.0,
                       project=output, name=f'pork_lr_{str(learning_rate).replace(".", "_")}')


# Provide a command-line interface without starting training when imported.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='runs')
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    args = parser.parse_args()
    train_model(args.data, args.output, args.learning_rate)

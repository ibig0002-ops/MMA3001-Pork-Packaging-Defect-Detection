"""Evaluate YOLOv8n on the validation or independent test split."""
import argparse
from pathlib import Path


def metrics_dict(results):
    """Extract overall box precision, recall, mAP50 and mAP50-95."""
    # Convert Ultralytics' aggregate bounding-box metrics to plain Python floats.
    # This makes the results easier to print, compare and store in reports.
    return {'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map)}


def evaluate_model(weights: str, data: str, split: str = 'test', imgsz: int = 800):
    """Run Ultralytics validation on an explicitly selected dataset split."""
    # Restrict evaluation to the held-out splits rather than training images.
    if split not in {'val', 'test'}:
        raise ValueError("split must be 'val' or 'test'")
    # Check that both the trained checkpoint and dataset YAML file exist.
    for filename in (weights, data):
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    # Import YOLO only when evaluation is requested.
    from ultralytics import YOLO
    # Evaluate at the requested image size and generate diagnostic plots.
    result = YOLO(weights).val(data=data, split=split, imgsz=imgsz,
                               batch=16, plots=True)
    # Return the four summary metrics in a consistent dictionary format.
    return metrics_dict(result)


if __name__ == '__main__':
    # Expose the evaluation settings as command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--imgsz', type=int, default=800)
    args = parser.parse_args()
    print(evaluate_model(args.weights, args.data, args.split, args.imgsz))

"""Run pork-packaging defect detection with a trained YOLOv8 checkpoint."""
import argparse
from pathlib import Path


def validate_image(path: str) -> Path:
    """Check that an inference input is an existing supported image file."""
    # Resolve the supplied path and reject missing or unsupported image inputs.
    image = Path(path)
    if not image.is_file():
        raise FileNotFoundError(f'Image not found: {image}')
    if image.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise ValueError('Supported extensions: .jpg, .jpeg, .png')
    return image


def predict_image(weights: str, image: str, confidence: float = 0.25, imgsz: int = 800):
    """Return detected defects with class, confidence, and xyxy pixel coordinates.

    An empty list means no detections above the threshold, not confirmed safety.
    """
    # Check input validity before loading the trained model.
    validate_image(image)
    if not Path(weights).is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {weights}')
    # Confidence thresholds must be dimensionless values between zero and one.
    if not 0 <= confidence <= 1:
        raise ValueError('confidence must be between 0 and 1')
    from ultralytics import YOLO
    # Load the trained checkpoint for inference rather than retraining the model.
    model = YOLO(weights)
    # Apply the confidence threshold and retain predictions for the input image.
    result = model.predict(source=image, imgsz=imgsz, conf=confidence,
                           save=False, verbose=False)[0]
    # Convert Ultralytics box objects into serialisable prediction records.
    detections = []
    for box in result.boxes:
        # xyxy coordinates are pixel positions: left, top, right and bottom.
        class_id = int(box.cls.item())
        detections.append({'class_id': class_id,
                           'class_name': result.names[class_id],
                           'confidence': float(box.conf.item()),
                           'xyxy_pixels': [float(v) for v in box.xyxy[0].tolist()]})
    # An empty list indicates no detections above the selected threshold.
    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--confidence', type=float, default=0.25)
    args = parser.parse_args()
    for detection in predict_image(args.weights, args.image, args.confidence):
        print(detection)

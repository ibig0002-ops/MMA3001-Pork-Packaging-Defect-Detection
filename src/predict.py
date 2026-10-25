"""Run pork-packaging defect detection with a trained YOLOv8 checkpoint."""
import argparse
from pathlib import Path


def validate_image(path: str) -> Path:
    """Check that an inference input is an existing supported image file."""
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
    validate_image(image)
    if not Path(weights).is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {weights}')
    if not 0 <= confidence <= 1:
        raise ValueError('confidence must be between 0 and 1')
    from ultralytics import YOLO
    model = YOLO(weights)
    result = model.predict(source=image, imgsz=imgsz, conf=confidence,
                           save=False, verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({'class_id': class_id,
                           'class_name': result.names[class_id],
                           'confidence': float(box.conf.item()),
                           'xyxy_pixels': [float(v) for v in box.xyxy[0].tolist()]})
    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--confidence', type=float, default=0.25)
    args = parser.parse_args()
    for detection in predict_image(args.weights, args.image, args.confidence):
        print(detection)

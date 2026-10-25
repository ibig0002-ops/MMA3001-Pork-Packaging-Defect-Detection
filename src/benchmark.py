"""Benchmark sequential YOLOv8 inference with GPU synchronisation when available."""
import argparse
from pathlib import Path
import statistics
import time


def benchmark_model(weights: str, images_dir: str, imgsz: int = 800,
                    repeats: int = 5, max_images: int = 30):
    """Return mean inference latency (ms/image) and throughput (images/s).

    Includes Python prediction overhead; hardware and software affect results.
    """
    if repeats < 1 or max_images < 1:
        raise ValueError('repeats and max_images must be positive')
    if not Path(weights).is_file():
        raise FileNotFoundError(weights)
    folder = Path(images_dir)
    if not folder.is_dir():
        raise NotADirectoryError(images_dir)
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})[:max_images]
    if not images:
        raise ValueError('No supported images found')
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warm up model loading and GPU kernels before timing.
    for _ in range(5):
        model.predict(str(images[0]), imgsz=imgsz, device=device,
                      verbose=False, save=False)
    durations = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        for image in images:
            model.predict(str(image), imgsz=imgsz, device=device,
                          verbose=False, save=False)
        sync()
        durations.append(time.perf_counter() - start)
    mean_seconds = statistics.mean(durations)
    return {'image_count': len(images), 'repeats': repeats,
            'latency_ms_per_image': mean_seconds * 1000 / len(images),
            'throughput_images_per_second': len(images) / mean_seconds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--imgsz', type=int, default=800)
    args = parser.parse_args()
    print(benchmark_model(args.weights, args.images_dir, args.imgsz))

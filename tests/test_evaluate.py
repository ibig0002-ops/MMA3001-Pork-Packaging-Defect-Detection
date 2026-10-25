"""Tests for metric extraction and split validation."""
from types import SimpleNamespace
import pytest
from src.evaluate import metrics_dict, evaluate_model


def test_metrics_dict():
    result = SimpleNamespace(box=SimpleNamespace(mp=0.5, mr=0.6, map50=0.7, map=0.3))
    assert metrics_dict(result) == {'precision': 0.5, 'recall': 0.6,
                                    'map50': 0.7, 'map50_95': 0.3}


def test_invalid_split():
    with pytest.raises(ValueError):
        evaluate_model('weights.pt', 'data.yaml', split='train')

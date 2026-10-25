"""Fast tests of image input handling (no GPU or checkpoint required)."""
import pytest
from src.predict import validate_image, predict_image


def test_valid_image(tmp_path):
    image = tmp_path / 'sample.jpg'
    image.write_bytes(b'test')
    assert validate_image(str(image)) == image


def test_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_image(str(tmp_path / 'missing.jpg'))


def test_unsupported_extension(tmp_path):
    image = tmp_path / 'sample.txt'
    image.write_text('not an image')
    with pytest.raises(ValueError):
        validate_image(str(image))


def test_missing_checkpoint(tmp_path):
    image = tmp_path / 'sample.jpg'
    image.write_bytes(b'test')
    with pytest.raises(FileNotFoundError):
        predict_image(str(tmp_path / 'missing.pt'), str(image))

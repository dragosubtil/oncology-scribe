"""Security floors and the supported Gradio/Pillow image path."""

import importlib.metadata
from pathlib import Path

import numpy as np
import pytest
from packaging.version import Version
from PIL import Image


@pytest.mark.parametrize(("package", "minimum"), [
    ("brotli", "1.2.1"),
    ("certifi", "2026.10.18"),
    ("filelock", "4.2.4"),
    ("opentelemetry-api", "1.45.1"),
    ("pillow", "13.0.0"),
])
def test_supported_environment_meets_security_floor(package, minimum):
    assert Version(importlib.metadata.version(package)) >= Version(minimum)


@pytest.mark.parametrize("format", ["png", "webp"])
def test_gradio_pillow_rgb_cache_roundtrip(tmp_path, format):
    from gradio import image_utils

    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :, 1] = 128
    stored = Path(image_utils.save_image(Image.fromarray(pixels), str(tmp_path), format=format))
    assert stored.is_file()
    assert stored.is_relative_to(tmp_path)
    with Image.open(stored) as decoded:
        decoded.load()
        assert decoded.size == (8, 8)
        assert decoded.mode == "RGB"
        if format == "png":
            np.testing.assert_array_equal(np.asarray(decoded), pixels)

import unittest
import pytest
from pathlib import Path
from PIL.Image import Image as PILImage
from mlutils.utils.imgutils.image_edge_detection import edge_detector
from mlutils.utils import load_img, ImageLoadersEnumType
from tests.config_tests import TEST_ROOT_DIR

@pytest.fixture()
def get_test_image():
    image_path = Path(TEST_ROOT_DIR) / "test_data/images/f16.jpeg"
    return load_img(path=image_path, loader=ImageLoadersEnumType.PIL)


def test_pil_edge_detection(get_test_image):

    edges = edge_detector(image=get_test_image)
    assert isinstance(edges, PILImage)


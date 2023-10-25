import unittest
from pathlib import Path
from mlutils.utils.imgutils.img_transformers import chuckify_image_from_path
from mlutils.utils.imgutils.image_enums import ImageLoadersEnumType
from tests.config_tests import TEST_ROOT_DIR


class TestImageTransformers(unittest.TestCase):

    def test_chuckify_image_from_path(self):
        image = Path(TEST_ROOT_DIR) / "test_data/images/f16.jpeg"

        chunks = chuckify_image_from_path(img=image,
                                          image_type=ImageLoadersEnumType.CV2,
                                          chunk_size=(256, 256),
                                          output_dir=None)

        self.assertTrue(len(chunks) != 0)
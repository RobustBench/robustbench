from unittest import TestCase

from robustbench import load_model
from tests.config import get_test_config


class Test(TestCase):
    def test_load_model(self):
        config = get_test_config()
        model_name = "Standard"
        load_model(model_name, model_dir=config["model_dir"])

    def test_load_model_norm(self):
        model_name = "Standard"
        config = get_test_config()
        with self.assertWarns(DeprecationWarning):
            load_model(model_name, model_dir=config["model_dir"], norm="L2")

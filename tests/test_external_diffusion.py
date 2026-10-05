import unittest

from navine.image.external import diffusers_available, load_external_config
from navine.learn.from_external import external_status
from navine.utils.config import load_config


class ExternalDiffusionTests(unittest.TestCase):
    def test_image_external_config_secondary_path(self):
        config = load_external_config()
        infer = config.get("inference") or {}
        self.assertTrue(config.get("secondary_path"))
        self.assertFalse(infer.get("custom_trained_only"))
        self.assertTrue((infer.get("diffusers") or {}).get("enabled"))
        self.assertEqual((infer.get("backend_order") or [None])[0], "diffusers")

    def test_primary_configs_unchanged(self):
        for name in ("image", "image_enterprise"):
            config = load_config(name)
            infer = config.get("inference") or {}
            self.assertTrue(infer.get("custom_trained_only"))
            self.assertFalse((infer.get("diffusers") or {}).get("enabled"))

    def test_video_external_config(self):
        config = load_config("video_external")
        infer = config.get("inference") or {}
        self.assertTrue(config.get("secondary_path"))
        self.assertTrue(infer.get("use_external_diffusion"))
        self.assertFalse(infer.get("use_learned_model"))

    def test_diffusers_availability_report(self):
        ok, detail = diffusers_available()
        self.assertIsInstance(ok, bool)
        if not ok:
            self.assertIsNotNone(detail)
        status = external_status()
        self.assertEqual(status["image_config"], "image_external")
        self.assertEqual(status["diffusers_installed"], ok)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

from sglang.srt.server_args import ServerArgs


class ServerArgsTest(unittest.TestCase):
    def test_mistral_native_detection_requires_params_json(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(
                os.path.join(model_dir, "consolidated.00.safetensors"),
                "w",
                encoding="utf-8",
            ):
                pass

            args = ServerArgs.__new__(ServerArgs)
            args.model_path = model_dir

            self.assertFalse(args._is_mistral_native_format())

    def test_mistral_native_detection_accepts_params_and_consolidated_weights(self):
        with tempfile.TemporaryDirectory() as model_dir:
            for filename in ("params.json", "consolidated.00.safetensors"):
                with open(os.path.join(model_dir, filename), "w", encoding="utf-8"):
                    pass

            args = ServerArgs.__new__(ServerArgs)
            args.model_path = model_dir

            self.assertTrue(args._is_mistral_native_format())

    def test_mistral_native_detection_accepts_params_and_consolidated_pt_weights(self):
        with tempfile.TemporaryDirectory() as model_dir:
            for filename in ("params.json", "consolidated.00.pt"):
                with open(os.path.join(model_dir, filename), "w", encoding="utf-8"):
                    pass

            args = ServerArgs.__new__(ServerArgs)
            args.model_path = model_dir

            self.assertTrue(args._is_mistral_native_format())


if __name__ == "__main__":
    unittest.main()

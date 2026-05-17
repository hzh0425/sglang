import os
import tempfile
import unittest

from sglang.srt.environ import envs
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

    def test_deepep_waterfill_takes_precedence_over_megamoe_env(self):
        args = ServerArgs.__new__(ServerArgs)
        args.enable_deepep_waterfill = True
        args.moe_a2a_backend = "allgather"
        args.deepep_mode = "auto"
        args.tp_size = 4
        args.ep_size = 1
        args.disable_cuda_graph = False
        args.disable_shared_experts_fusion = True
        args.enforce_shared_experts_fusion = False

        with envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.override(True):
            args._handle_a2a_moe()

        self.assertEqual(args.moe_a2a_backend, "deepep")
        self.assertEqual(args.ep_size, args.tp_size)
        self.assertFalse(args.disable_shared_experts_fusion)
        self.assertTrue(args.enforce_shared_experts_fusion)


if __name__ == "__main__":
    unittest.main()

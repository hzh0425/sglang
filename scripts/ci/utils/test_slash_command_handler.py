import sys
import types
import unittest

github_stub = types.ModuleType("github")
github_stub.Auth = types.SimpleNamespace(Token=lambda token: token)
github_stub.Github = object
sys.modules.setdefault("github", github_stub)

from scripts.ci.utils import slash_command_handler


class SlashCommandHandlerTest(unittest.TestCase):
    def test_detect_suite_resolves_legacy_cuda_suite(self):
        info = slash_command_handler.detect_suite(
            "registered/tokenizer/test_multi_detokenizer.py"
        )

        self.assertIsNone(info["error"])
        self.assertFalse(info["is_cpu"])
        self.assertEqual(info["suite"], "base-b-test-1-gpu-large")
        self.assertEqual(info["runner_label"], "1-gpu-h100")
        self.assertEqual(
            info["install_script"], "scripts/ci/cuda/ci_install_dependency.sh"
        )
        self.assertEqual(info["install_timeout"], "20")

    def test_legacy_nightly_cuda_suite_stays_unsupported(self):
        runner_config = slash_command_handler._runner_config_from_legacy_cuda_suite(
            "nightly-1-gpu"
        )

        self.assertIsNone(runner_config)


if __name__ == "__main__":
    unittest.main()

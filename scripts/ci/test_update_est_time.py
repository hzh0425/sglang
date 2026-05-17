import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.ci import update_est_time


class UpdateEstTimeTest(unittest.TestCase):
    def test_update_files_handles_multiline_registrations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            relpath = "python/sglang/test/test_example.py"
            test_file = tmp_root / relpath
            test_file.parent.mkdir(parents=True)
            test_file.write_text(textwrap.dedent("""
                    from sglang.test.ci import register_cpu_ci, register_cuda_ci

                    register_cuda_ci(
                        stage="base-b",
                        est_time=120,
                        runner_config="1-gpu-large",
                    )

                    register_cpu_ci(
                        runner_config="cpu",
                        stage="base-a",
                        est_time=5,
                    )

                    register_cuda_ci(
                        40,
                        "nightly-test-1-gpu",
                    )
                    """).lstrip())

            old_root = update_est_time.REPO_ROOT
            update_est_time.REPO_ROOT = tmp_root
            try:
                changes = update_est_time.update_files(
                    {
                        "est": {
                            "base-b-test-1-gpu-large": {relpath: 180},
                            "base-a-test-cpu": {relpath: 8},
                            "nightly-test-1-gpu": {relpath: 45},
                        }
                    }
                )
            finally:
                update_est_time.REPO_ROOT = old_root

            self.assertEqual(
                changes,
                [
                    (relpath, "base-b-test-1-gpu-large", 120, 180),
                    (relpath, "base-a-test-cpu", 5, 8),
                    (relpath, "nightly-test-1-gpu", 40, 45),
                ],
            )
            content = test_file.read_text()
            self.assertIn("est_time=180", content)
            self.assertIn("est_time=8", content)
            self.assertIn('45,\n    "nightly-test-1-gpu"', content)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from unittest import mock

from sglang.test.ci.ci_utils import _repo_relative_path


class RepoRelativePathTest(unittest.TestCase):
    def test_absolute_package_path_is_relative_to_checkout_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkout = os.path.join(tmpdir, "sglang")
            target = os.path.join(
                checkout, "python", "sglang", "jit_kernel", "tests", "test_fa3.py"
            )

            with mock.patch.dict(
                os.environ, {"GITHUB_WORKSPACE": checkout}, clear=False
            ):
                self.assertEqual(
                    _repo_relative_path(target),
                    "python/sglang/jit_kernel/tests/test_fa3.py",
                )

    def test_absolute_path_can_fall_back_to_cwd(self):
        with tempfile.TemporaryDirectory() as checkout:
            target = os.path.join(checkout, "test", "srt", "test_decode.py")

            with mock.patch.dict(os.environ, {"GITHUB_WORKSPACE": ""}, clear=False):
                with mock.patch("os.getcwd", return_value=checkout):
                    self.assertEqual(
                        _repo_relative_path(target),
                        "test/srt/test_decode.py",
                    )


if __name__ == "__main__":
    unittest.main()

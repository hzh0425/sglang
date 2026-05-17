import pathlib
import re
import tomllib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _pinned_dependency_version(name: str) -> str:
    pyproject = tomllib.loads((REPO_ROOT / "python/pyproject.toml").read_text())
    prefix = f"{name}=="
    for dependency in pyproject["project"]["dependencies"]:
        if dependency.startswith(prefix):
            return dependency[len(prefix) :]
    raise AssertionError(f"missing dependency pin for {name}")


class DependencyVersionsTest(unittest.TestCase):
    def test_flashinfer_runtime_check_matches_dependency_pin(self):
        engine_py = (REPO_ROOT / "python/sglang/srt/entrypoints/engine.py").read_text()
        match = re.search(
            r'assert_pkg_version\(\s*"flashinfer_python",\s*"([^"]+)"',
            engine_py,
            re.DOTALL,
        )

        self.assertIsNotNone(match)
        self.assertEqual(
            match.group(1), _pinned_dependency_version("flashinfer_python")
        )


if __name__ == "__main__":
    unittest.main()

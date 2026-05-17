import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CHECK_CHANGES_WORKFLOW = REPO_ROOT / ".github/workflows/_pr-test-check-changes.yml"
SGL_KERNEL_EXPR = (
    "${{ steps.filter.outputs.sgl_kernel || steps.run-mode.outputs.run_all_tests }}"
)


class CheckChangesWorkflowTest(unittest.TestCase):
    def test_run_all_tests_enables_sgl_kernel_output(self):
        workflow = CHECK_CHANGES_WORKFLOW.read_text()

        self.assertIn(f"sgl_kernel: {SGL_KERNEL_EXPR}", workflow)
        self.assertIn(f'sgl_kernel="{SGL_KERNEL_EXPR}"', workflow)
        self.assertIn(f'echo "| sgl_kernel        | {SGL_KERNEL_EXPR} |"', workflow)


if __name__ == "__main__":
    unittest.main()

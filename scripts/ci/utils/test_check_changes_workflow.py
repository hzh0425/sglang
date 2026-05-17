import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CHECK_CHANGES_WORKFLOW = REPO_ROOT / ".github/workflows/_pr-test-check-changes.yml"
PR_TEST_WORKFLOW = REPO_ROOT / ".github/workflows/pr-test.yml"
PR_TEST_EXTRA_WORKFLOW = REPO_ROOT / ".github/workflows/pr-test-extra.yml"
SGL_KERNEL_EXPR = (
    "${{ steps.filter.outputs.sgl_kernel || "
    "steps.sgl-kernel-run-mode.outputs.sgl_kernel_run_all_tests }}"
)


class CheckChangesWorkflowTest(unittest.TestCase):
    def test_sgl_kernel_run_all_tests_requires_explicit_caller_opt_in(self):
        workflow = CHECK_CHANGES_WORKFLOW.read_text()

        self.assertIn("enable_sgl_kernel_run_all_tests:", workflow)
        self.assertIn("id: sgl-kernel-run-mode", workflow)
        self.assertIn(f"sgl_kernel: {SGL_KERNEL_EXPR}", workflow)
        self.assertIn(f'sgl_kernel="{SGL_KERNEL_EXPR}"', workflow)
        self.assertIn(f'echo "| sgl_kernel        | {SGL_KERNEL_EXPR} |"', workflow)

    def test_callers_only_opt_in_when_kernel_wheels_are_built(self):
        pr_test = PR_TEST_WORKFLOW.read_text()
        pr_test_extra = PR_TEST_EXTRA_WORKFLOW.read_text()

        self.assertIn(
            "enable_sgl_kernel_run_all_tests: ${{ inputs.run_all_tests == true "
            "&& inputs.test_parallel_dispatch != true "
            "&& github.event_name != 'schedule' }}",
            pr_test,
        )
        self.assertIn(
            "enable_sgl_kernel_run_all_tests: ${{ inputs.run_all_tests == true }}",
            pr_test_extra,
        )


if __name__ == "__main__":
    unittest.main()

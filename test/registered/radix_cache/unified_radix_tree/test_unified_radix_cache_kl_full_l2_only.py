import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import (
    UnifiedRadixTreeL2OnlyEvalMixin,
    UnifiedRadixTreeTestMixin,
)
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=320, stage="base-c", runner_config="2-gpu-large")

FULL_MODEL = "/home/t4/models/Qwen3-32B"


class TestUnifiedFullRadixCacheL2Only(
    UnifiedRadixTreeL2OnlyEvalMixin, UnifiedRadixTreeTestMixin, CustomTestCase
):
    """UnifiedTree L2-only mode on a full-attention model."""

    kl_threshold = 0.0025
    gsm8k_threshold = 0.85
    mmlu_threshold = 0.65
    num_gsm8k_questions = 20
    gsm8k_parallel = 2
    gsm8k_max_new_tokens = 2048
    num_mmlu_examples = 16
    mmlu_num_threads = 4

    @classmethod
    def setUpClass(cls):
        cls.model = FULL_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.80",
                "--max-running-requests",
                "64",
                "--page-size",
                "64",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--unified-tree-l2-only-mode",
                "--enable-cache-report",
                "--enable-metrics",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()

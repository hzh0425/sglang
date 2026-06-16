import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import (
    UnifiedRadixTreeL2OnlyEvalMixin,
    UnifiedRadixTreeTestMixin,
)
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

DSV4_FLASH_MODEL = (
    "/home/t4/models/deepseek-v4-flash-fp8/sgl-project/DeepSeek-V4-Flash-FP8"
)
DSV4_FLASH_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=1400, stage="base-c", runner_config="4-gpu-h100")


def _assert_dsv4_decode_cached_tokens(result, history_len, output_len, label):
    expected = history_len + output_len
    actual = result["meta_info"]["cached_tokens"]
    lower = max(0, expected - 256)
    assert actual >= lower, f"{label}: expected cached_tokens>={lower}, got {actual}"


class TestUnifiedDeepSeekV4FlashHiCacheL2Only(
    UnifiedRadixTreeL2OnlyEvalMixin, UnifiedRadixTreeTestMixin, CustomTestCase
):
    """DeepSeek V4 Flash FP8 with UnifiedTree L2-only mode."""

    kl_threshold = 0.005
    sampling_temperature = 0
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5
    decode_cache_assert = staticmethod(_assert_dsv4_decode_cached_tokens)
    gsm8k_threshold = 0.85
    mmlu_threshold = 0.65
    num_gsm8k_questions = 20
    gsm8k_parallel = 1
    gsm8k_max_new_tokens = 2048
    num_mmlu_examples = 16
    mmlu_num_threads = 4
    host_cache_settle_time_s = 5.0
    max_running_requests = 4

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_multiturn_logprobs_match(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DSV4_FLASH_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--attention-backend",
                "compressed",
                "--page-size",
                "256",
                "--chunked-prefill-size",
                "8192",
                "--mem-fraction-static",
                "0.9",
                "--disable-shared-experts-fusion",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--swa-full-tokens-ratio",
                "0.25",
                "--max-total-tokens",
                "20000",
                "--max-running-requests",
                str(cls.max_running_requests),
                "--unified-tree-l2-only-mode",
                "--enable-cache-report",
                "--enable-metrics",
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            },
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()

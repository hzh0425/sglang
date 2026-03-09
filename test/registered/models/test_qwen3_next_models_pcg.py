"""
Qwen3 Next piecewise CUDA graph tests.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=400,
    suite="stage-c-test-4-gpu-h100",
)

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
QWEN3_NEXT_FP8_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"

ACC_THRESHOLDS = {
    QWEN3_NEXT_MODEL: {"kl_div": 0.0025, "gsm8k": 0.93},
    QWEN3_NEXT_FP8_MODEL: {"gsm8k": 0.62},
}


class TestQwen3NextPiecewiseCudaGraph(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(
            metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )


class TestQwen3NextHiCacheFileBackend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_FP8_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="qwen3-next-hicache-")
        env = {
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--tp",
                "2",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--page-size",
                "64",
                "--max-total-tokens",
                "65536",
                "--hicache-mem-layout",
                "page_first_direct",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-size",
                "0",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=512,
            parallel=10,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        return run_eval_few_shot_gsm8k(args)

    def test_gsm8k(self):
        first_metrics = self._run_gsm8k()
        print(f"first_metrics={first_metrics}")
        self.assertGreaterEqual(
            first_metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )

        flush_resp = requests.post(f"{self.base_url}/flush_cache", timeout=60)
        self.assertEqual(
            flush_resp.status_code,
            200,
            f"flush_cache failed: {flush_resp.status_code} - {flush_resp.text}",
        )
        self.assertIn("Cache flushed", flush_resp.text)

        second_metrics = self._run_gsm8k()
        print(f"second_metrics={second_metrics}")
        self.assertGreaterEqual(
            second_metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )
        self.assertLessEqual(
            abs(second_metrics["accuracy"] - first_metrics["accuracy"]),
            0.05,
            f"HiCache prefetch accuracy drift too large: "
            f"first={first_metrics['accuracy']}, second={second_metrics['accuracy']}",
        )


if __name__ == "__main__":
    unittest.main()

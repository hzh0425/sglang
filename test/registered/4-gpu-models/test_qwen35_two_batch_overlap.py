import shutil
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci

# This eval harness applies the chat_template, which is critical for qwen3.5
# to get good accuracy on gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, suite="stage-c-test-4-gpu-h100")

QWEN35_27B_MODEL = "/home/t4/models/Qwen/Qwen35-35B-FP8"
ACC_THRESHOLDS = {QWEN35_27B_MODEL: {"gsm8k": 0.93}}


class TestQwen35(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_27B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="qwen35-hicache-")
        env = {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "128",
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "1",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
                "--enable-two-batch-overlap",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=100,
            max_tokens=16000,
            num_threads=50,
            repeat=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        return run_eval(args)

    def test_gsm8k(self):
        first_metrics = self._run_gsm8k()
        print(f"first_metrics={first_metrics}")
        self.assertGreaterEqual(
            first_metrics["score"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )

if __name__ == "__main__":
    unittest.main()


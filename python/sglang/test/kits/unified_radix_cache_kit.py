import re
import random
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.test.kl_multiturn_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper,
    test_input_output_logprobs_match_helper,
    test_input_output_logprobs_match_prefill_cache_hit_helper,
)


def _random_suffixes(n, length, seed):
    """Generate n random token-id lists of the given length."""
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


class UnifiedRadixTreeTestMixin:
    """Mixin: gsm8k, mmlu and multi-turn KL tests with multi-branch interleaving."""

    kl_threshold: float = 0.003
    max_new_tokens: int = 512
    num_groups: int = 3
    branches_per_group: int = 3
    prefix_len: int = 512
    prefill_cache_assert = None
    decode_cache_assert = None
    sampling_temperature: float = 1
    decode_hit_request_batch_size: int | None = None
    decode_hit_inter_batch_delay_s: float = 0

    gsm8k_threshold: float = 0.93
    mmlu_threshold: float = 0.8
    num_gsm8k_questions: int = 200
    gsm8k_parallel: int = 128
    gsm8k_max_new_tokens: int = 16000
    num_mmlu_examples: int = 64
    mmlu_num_threads: int = 32

    def _run_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=10,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=self.gsm8k_max_new_tokens,
            parallel=self.gsm8k_parallel,
            host=url.hostname,
            port=int(url.port),
        )
        return run_few_shot_gsm8k(args)

    def _run_mmlu(self):
        from sglang.test.run_eval import run_eval as run_simple_eval

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=self.num_mmlu_examples,
            num_threads=self.mmlu_num_threads,
        )
        return run_simple_eval(args)

    def test_gsm8k(self):
        """Few-shot GSM8K math reasoning accuracy."""
        metrics = self._run_gsm8k()
        print(
            f"[{self.__class__.__name__}] GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_threshold)

    def test_mmlu(self):
        """Simple-evals MMLU multi-task accuracy."""
        metrics = self._run_mmlu()
        print(
            f"[{self.__class__.__name__}] MMLU score: {metrics['score']:.3f} "
            f"(threshold: {self.mmlu_threshold})"
        )
        self.assertGreaterEqual(metrics["score"], self.mmlu_threshold)

    def test_multiturn_logprobs_match(self):
        """Helper 1: 3-turn, no explicit cache seeding."""
        ids = self.input_ids[:4]
        n = len(ids)
        t2 = _random_suffixes(n, 512, seed=100)
        t3 = _random_suffixes(n, 256, seed=200)
        test_input_output_logprobs_match_helper(
            self.base_url,
            self.model,
            self.kl_threshold,
            ids,
            turn_suffixes=[t2, t3],
            assert_decode_cached_tokens=self.decode_cache_assert,
            max_new_tokens=self.max_new_tokens,
            sampling_temperature=self.sampling_temperature,
        )

    def test_multiturn_prefill_cache_hit_branching(self):
        """Helper 2: prefill hit + 2 decode-hit turns, multi-branch interleaved."""
        num_groups = self.num_groups
        branches = self.branches_per_group
        n = num_groups * branches
        rng = random.Random(456)
        prefix_ids, full_ids = [], []
        for g in range(num_groups):
            prefix = self.input_ids[g][: self.prefix_len]
            for b in range(branches):
                suffix = [rng.randint(1, 30000) for _ in range(256 + b * 64)]
                prefix_ids.append(list(prefix))
                full_ids.append(prefix + suffix)

        t2 = _random_suffixes(n, 512, seed=789)
        t3 = _random_suffixes(n, 256, seed=890)
        test_input_output_logprobs_match_prefill_cache_hit_helper(
            self.base_url,
            self.model,
            self.kl_threshold,
            prefix_input_ids=prefix_ids,
            full_input_ids=full_ids,
            turn_suffixes=[t2, t3],
            assert_prefill_cached_tokens=self.prefill_cache_assert,
            assert_decode_cached_tokens=self.decode_cache_assert,
            branches_per_group=branches,
            max_new_tokens=self.max_new_tokens,
            sampling_temperature=self.sampling_temperature,
        )

    def test_multiturn_decode_cache_hit_branching(self):
        """Helper 3: 3-turn decode hit, multi-branch interleaved."""
        num_groups = self.num_groups
        branches = self.branches_per_group
        n = num_groups * branches
        first_turn = []
        for g in range(num_groups):
            base = self.input_ids[g][: self.prefix_len]
            for _ in range(branches):
                first_turn.append(list(base))

        t2 = _random_suffixes(n, 512, seed=300)
        t3 = _random_suffixes(n, 256, seed=400)
        test_input_output_logprobs_match_decode_cache_hit_helper(
            self.base_url,
            self.model,
            self.kl_threshold,
            first_turn,
            turn_suffixes=[t2, t3],
            assert_decode_cached_tokens=self.decode_cache_assert,
            branches_per_group=branches,
            max_new_tokens=self.max_new_tokens,
            sampling_temperature=self.sampling_temperature,
            request_batch_size=self.decode_hit_request_batch_size,
            inter_batch_delay_s=self.decode_hit_inter_batch_delay_s,
        )


class UnifiedRadixTreeL2OnlyEvalMixin:
    """Mixin: verify second-pass eval traffic reuses host-backed UnifiedTree prefixes."""

    host_cache_settle_time_s: float = 2.0
    host_probe_output_len: int = 4

    def _read_cached_tokens_total(self, cache_source: str) -> float:
        response = None
        last_exc = None
        for attempt in range(5):
            try:
                response = requests.get(self.base_url + "/metrics", timeout=30)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        if response is None:
            raise AssertionError(f"Failed to read /metrics: {last_exc}")

        pattern = re.compile(
            r'^sglang:cached_tokens_total\{[^}]*cache_source="' +
            re.escape(cache_source) +
            r'"[^}]*\}\s+([\d.eE+-]+)$'
        )
        total = 0.0
        for line in response.text.splitlines():
            match = pattern.match(line)
            if match:
                total += float(match.group(1))
        return total

    def _probe_host_cached_tokens(self) -> int:
        if not getattr(self, "input_ids", None):
            raise AssertionError("input_ids are required for host cache probe")

        payload = {
            "input_ids": list(self.input_ids[0]),
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": self.host_probe_output_len,
            },
        }

        first = requests.post(self.base_url + "/generate", json=payload, timeout=120)
        first.raise_for_status()
        time.sleep(self.host_cache_settle_time_s)
        second = requests.post(self.base_url + "/generate", json=payload, timeout=120)
        second.raise_for_status()

        meta = second.json().get("meta_info") or {}
        details = meta.get("cached_tokens_details") or {}
        return int(details.get("host", 0))

    def _assert_second_pass_hits_host(
        self,
        *,
        name: str,
        run_fn,
        metric_key: str,
        threshold: float,
    ) -> None:
        first_metrics = run_fn()
        first_score = first_metrics[metric_key]
        print(f"[{self.__class__.__name__}] {name} pass 1 {metric_key}: {first_score:.3f}")
        self.assertGreaterEqual(
            first_score,
            threshold,
            f"{name} pass 1 {metric_key} {first_score:.3f} < threshold {threshold}",
        )

        time.sleep(self.host_cache_settle_time_s)
        host_before = self._read_cached_tokens_total("host")

        second_metrics = run_fn()
        second_score = second_metrics[metric_key]
        host_after = self._read_cached_tokens_total("host")
        host_delta = host_after - host_before
        print(
            f"[{self.__class__.__name__}] {name} pass 2 {metric_key}: {second_score:.3f}, "
            f"host_cached_tokens_delta={host_delta:.0f}"
        )
        self.assertGreaterEqual(
            second_score,
            threshold,
            f"{name} pass 2 {metric_key} {second_score:.3f} < threshold {threshold}",
        )
        if host_delta > 0:
            return

        probe_host_tokens = self._probe_host_cached_tokens()
        print(
            f"[{self.__class__.__name__}] {name} metrics host delta unavailable; "
            f"fallback probe host_cached_tokens={probe_host_tokens}"
        )
        self.assertGreater(
            probe_host_tokens,
            0,
            f"Expected host cached tokens during {name} pass 2, got {host_delta=} "
            f"and fallback probe host_cached_tokens={probe_host_tokens}",
        )

    def test_gsm8k_second_pass_hits_host(self):
        self._assert_second_pass_hits_host(
            name="GSM8K",
            run_fn=self._run_gsm8k,
            metric_key="accuracy",
            threshold=self.gsm8k_threshold,
        )

    def test_mmlu_second_pass_hits_host(self):
        self._assert_second_pass_hits_host(
            name="MMLU",
            run_fn=self._run_mmlu,
            metric_key="score",
            threshold=self.mmlu_threshold,
        )

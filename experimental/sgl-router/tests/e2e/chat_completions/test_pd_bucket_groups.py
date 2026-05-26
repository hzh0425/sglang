# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU acceptance coverage for PD bucket group routing."""

from __future__ import annotations

import os

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import _get_open_port, spawn_worker
from infra.model_specs import get_model_spec

_PD_TRANSFER_BACKEND = os.environ.get("SGLANG_ROUTER_E2E_PD_BACKEND", "mooncake")
_BASE_PD_WORKER_ARGS = [
    "--disaggregation-transfer-backend",
    _PD_TRANSFER_BACKEND,
    "--mem-fraction-static",
    "0.18",
    "--disable-cuda-graph",
    "--chunked-prefill-size",
    "2048",
    "--max-prefill-tokens",
    "4096",
]


def _pd_worker_args(gpu_id: int) -> list[str]:
    args = list(_BASE_PD_WORKER_ARGS)
    ib_device = os.environ.get("SGLANG_ROUTER_E2E_PD_IB_DEVICE")
    if ib_device is None:
        candidate = f"mlx5_ib{gpu_id}"
        if os.path.isdir(f"/sys/class/infiniband/{candidate}"):
            ib_device = candidate
    if ib_device:
        args.extend(["--disaggregation-ib-device", ib_device])
    return args


def _chat(router_url: str, model_id: str, prompt: str) -> httpx.Response:
    return httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "stream": False,
        },
        timeout=120.0,
    )


def _assert_route(
    response: httpx.Response,
    *,
    group: str,
    prefill_worker: str,
    decode_worker: str,
) -> None:
    assert response.status_code == 200, response.text
    assert response.headers.get("x-sglang-prefill-group") == group
    assert response.headers.get("x-sglang-decode-group") == group
    assert response.headers.get("x-sglang-prefill-worker") == prefill_worker
    assert response.headers.get("x-sglang-decode-worker") == decode_worker


@pytest.mark.real_gpu
@pytest.mark.slow
def test_pd_bucket_groups_route_short_and_long_requests(
    router_binary,  # noqa: ARG001 - fixture forces release-binary presence
    gpu_allocator,
) -> None:
    spec = get_model_spec("qwen3-0.6b")
    gpus = gpu_allocator.acquire(3)
    try:
        short_prefill_bootstrap = _get_open_port()
        long_prefill_bootstrap = _get_open_port()
        long_prefill_gpu = gpus[1]
        with (
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[0]],
                disagg_mode="prefill",
                bootstrap_port=short_prefill_bootstrap,
                extra_args=_pd_worker_args(gpus[0]),
            ) as short_prefill,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[1]],
                disagg_mode="decode",
                bootstrap_port=short_prefill_bootstrap,
                extra_args=_pd_worker_args(gpus[1]),
            ) as short_decode,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[long_prefill_gpu],
                disagg_mode="prefill",
                bootstrap_port=long_prefill_bootstrap,
                extra_args=_pd_worker_args(long_prefill_gpu),
            ) as long_prefill,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[2]],
                disagg_mode="decode",
                bootstrap_port=long_prefill_bootstrap,
                extra_args=_pd_worker_args(gpus[2]),
            ) as long_decode,
            Gateway() as router,
        ):
            worker_groups = [
                (short_prefill.url, "short"),
                (short_decode.url, "short"),
                (long_prefill.url, "long"),
                (long_decode.url, "long"),
            ]
            router.start_pd(
                model_id=spec["model"],
                tokenizer_path=spec["model"],
                prefill_urls=[short_prefill.url, long_prefill.url],
                decode_urls=[short_decode.url, long_decode.url],
                policy="sticky_session_load_based",
                worker_groups=worker_groups,
                pd_bucket={
                    "short_group": "short",
                    "long_group": "long",
                    "prefill_long_threshold": 64,
                    "decode_long_threshold": 64,
                },
                timeout=120.0,
            )

            short_resp = _chat(router.base_url, spec["model"], "short bucket request")
            _assert_route(
                short_resp,
                group="short",
                prefill_worker=short_prefill.url,
                decode_worker=short_decode.url,
            )

            long_prompt = "long bucket routing validation " * 128
            long_resp = _chat(router.base_url, spec["model"], long_prompt)
            _assert_route(
                long_resp,
                group="long",
                prefill_worker=long_prefill.url,
                decode_worker=long_decode.url,
            )
    finally:
        gpu_allocator.release(gpus)


@pytest.mark.real_gpu
@pytest.mark.slow
def test_pd_bucket_groups_route_three_token_buckets(
    router_binary,  # noqa: ARG001 - fixture forces release-binary presence
    gpu_allocator,
) -> None:
    spec = get_model_spec("qwen3-0.6b")
    gpus = gpu_allocator.acquire(3)
    try:
        small_bootstrap = _get_open_port()
        medium_bootstrap = _get_open_port()
        large_bootstrap = _get_open_port()
        with (
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[0]],
                disagg_mode="prefill",
                bootstrap_port=small_bootstrap,
                extra_args=_pd_worker_args(gpus[0]),
            ) as small_prefill,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[1]],
                disagg_mode="decode",
                bootstrap_port=small_bootstrap,
                extra_args=_pd_worker_args(gpus[1]),
            ) as small_decode,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[1]],
                disagg_mode="prefill",
                bootstrap_port=medium_bootstrap,
                extra_args=_pd_worker_args(gpus[1]),
            ) as medium_prefill,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[2]],
                disagg_mode="decode",
                bootstrap_port=medium_bootstrap,
                extra_args=_pd_worker_args(gpus[2]),
            ) as medium_decode,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[2]],
                disagg_mode="prefill",
                bootstrap_port=large_bootstrap,
                extra_args=_pd_worker_args(gpus[2]),
            ) as large_prefill,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[0]],
                disagg_mode="decode",
                bootstrap_port=large_bootstrap,
                extra_args=_pd_worker_args(gpus[0]),
            ) as large_decode,
            Gateway() as router,
        ):
            worker_groups = [
                (small_prefill.url, "ctx0_16k"),
                (small_decode.url, "ctx0_16k"),
                (medium_prefill.url, "ctx16_32k"),
                (medium_decode.url, "ctx16_32k"),
                (large_prefill.url, "ctx32_64k"),
                (large_decode.url, "ctx32_64k"),
            ]
            router.start_pd(
                model_id=spec["model"],
                tokenizer_path=spec["model"],
                prefill_urls=[
                    small_prefill.url,
                    medium_prefill.url,
                    large_prefill.url,
                ],
                decode_urls=[small_decode.url, medium_decode.url, large_decode.url],
                policy="sticky_session_load_based",
                worker_groups=worker_groups,
                pd_bucket={
                    "groups": [
                        {"group": "ctx0_16k", "max_tokens": 16},
                        {"group": "ctx16_32k", "max_tokens": 96},
                        {"group": "ctx32_64k", "max_tokens": 1024},
                    ],
                },
                timeout=120.0,
            )

            small_resp = _chat(router.base_url, spec["model"], "short bucket request")
            _assert_route(
                small_resp,
                group="ctx0_16k",
                prefill_worker=small_prefill.url,
                decode_worker=small_decode.url,
            )

            medium_prompt = "middle bucket routing validation " * 12
            medium_resp = _chat(router.base_url, spec["model"], medium_prompt)
            _assert_route(
                medium_resp,
                group="ctx16_32k",
                prefill_worker=medium_prefill.url,
                decode_worker=medium_decode.url,
            )

            large_prompt = "long bucket routing validation " * 80
            large_resp = _chat(router.base_url, spec["model"], large_prompt)
            _assert_route(
                large_resp,
                group="ctx32_64k",
                prefill_worker=large_prefill.url,
                decode_worker=large_decode.url,
            )
    finally:
        gpu_allocator.release(gpus)

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v4_cp
from sglang.srt.models.deepseek_v4 import prepare_dsv4_cp_v2_input_ids
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4CPV2Metadata(CustomTestCase):
    @staticmethod
    def _metadata(num_rows: int) -> DSV4AttnMetadata:
        rows = torch.arange(num_rows, dtype=torch.int32)
        matrix = torch.stack((rows, rows + 100), dim=1)
        metadata = DSV4AttnMetadata(
            page_size=256,
            page_table=matrix.clone(),
            raw_out_loc=rows.clone(),
            cuda_int32_kwargs={"dtype": torch.int32},
            seq_lens_casual=rows.clone(),
            positions_casual=rows.clone(),
            swa_page_indices=matrix.clone(),
            swa_topk_lengths=rows.clone(),
            c4_sparse_topk=512,
        )
        metadata.swa_out_cache_loc = rows.clone()
        metadata.c4_out_loc = rows.clone()
        metadata.c128_out_loc = rows.clone()
        metadata.c4_topk_lengths_raw = rows.clone()
        metadata.c4_topk_lengths_clamp1 = rows.clone()
        metadata.c128_page_indices = matrix.clone()
        metadata.c128_topk_lengths_clamp1 = rows.clone()
        return metadata

    def test_reindex_trims_only_global_write_fields(self):
        metadata = self._metadata(num_rows=16)

        with get_parallel().override(attn_cp_rank=2, attn_cp_size=4):
            metadata.apply_cp_reindex(logical_global_len=10)

        expected_local_rows = [2, 6, 10, 14]
        for field_name in metadata._CP_REINDEX_FIELDS:
            self.assertEqual(
                getattr(metadata, field_name)[:, 0].tolist()
                if getattr(metadata, field_name).ndim == 2
                else getattr(metadata, field_name).tolist(),
                expected_local_rows,
            )
        for field_name in metadata._CP_GLOBAL_FIELDS:
            self.assertEqual(getattr(metadata, field_name).shape[0], 10)

    def test_reindex_without_logical_length_preserves_cp_v1_contract(self):
        metadata = self._metadata(num_rows=8)

        with get_parallel().override(attn_cp_rank=1, attn_cp_size=4):
            metadata.apply_cp_reindex()

        self.assertEqual(metadata.seq_lens_casual.tolist(), [1, 5])
        for field_name in metadata._CP_GLOBAL_FIELDS:
            self.assertEqual(getattr(metadata, field_name).shape[0], 8)

    def test_reindex_rejects_invalid_logical_length(self):
        metadata = self._metadata(num_rows=8)

        with (
            get_parallel().override(attn_cp_rank=0, attn_cp_size=4),
            self.assertRaisesRegex(ValueError, "logical_global_len=9"),
        ):
            metadata.apply_cp_reindex(logical_global_len=9)


class TestDSV4CPV2ModelInputs(CustomTestCase):
    def test_dsv4_canonical_cp_selects_only_dsa_legacy_alias(self):
        server_args = SimpleNamespace(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            enable_prefill_context_parallel=True,
            enable_dsa_prefill_context_parallel=False,
            dsa_prefill_cp_mode="in-seq-split",
            enable_dp_attention=False,
            moe_dense_tp_size=8,
            attn_cp_size=1,
            dp_size=1,
            tp_size=8,
            ep_size=1,
            moe_a2a_backend="none",
        )

        with patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set"
        ):
            validate_deepseek_v4_cp(server_args)

        self.assertTrue(server_args.enable_dsa_prefill_context_parallel)
        self.assertFalse(server_args.enable_prefill_context_parallel)
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")

    def test_dsv4_arch_enables_layer_cp_under_cp_v2(self):
        hf_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        server_args = SimpleNamespace(
            get_model_config=lambda: SimpleNamespace(hf_config=hf_config)
        )

        with (
            get_parallel().override(attn_cp_size=4),
            patch("sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_server_args",
                return_value=server_args,
            ),
        ):
            self.assertTrue(is_dsa_enable_prefill_cp())

    def test_non_a2a_input_ids_follow_rank_major_physical_rows(self):
        local_ids = torch.tensor([2, 6, 0, 0])
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[4] * 4)
        )
        moe_backend = MagicMock()
        moe_backend.is_none.return_value = True

        def gather(output, input_ids):
            self.assertTrue(torch.equal(input_ids, local_ids))
            output.copy_(torch.arange(16))

        with (
            patch(
                "sglang.srt.models.deepseek_v4.cp_shard_hidden_states",
                return_value=local_ids,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_moe_a2a_backend",
                return_value=moe_backend,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.attn_cp_all_gather_into_tensor",
                side_effect=gather,
            ) as all_gather,
        ):
            input_ids = prepare_dsv4_cp_v2_input_ids(torch.arange(10), forward_batch)

        self.assertEqual(input_ids.tolist(), list(range(16)))
        all_gather.assert_called_once()

    def test_a2a_input_ids_stay_rank_local_and_physical(self):
        local_ids = torch.tensor([2, 6, 0, 0])
        forward_batch = SimpleNamespace()
        moe_backend = MagicMock()
        moe_backend.is_none.return_value = False

        with (
            patch(
                "sglang.srt.models.deepseek_v4.cp_shard_hidden_states",
                return_value=local_ids,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_moe_a2a_backend",
                return_value=moe_backend,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.attn_cp_all_gather_into_tensor"
            ) as all_gather,
        ):
            input_ids = prepare_dsv4_cp_v2_input_ids(torch.arange(10), forward_batch)

        self.assertIs(input_ids, local_ids)
        all_gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import unittest

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index


class MultiTokenizerMixinTest(unittest.TestCase):
    def test_batch_embedding_split_preserves_required_fields(self):
        output = BatchEmbeddingOutput(
            rids=["rid-0", "rid-1"],
            http_worker_ipcs=["ipc-0", "ipc-1"],
            finished_reasons=[{"type": "length"}, {"type": "stop"}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            prompt_tokens=[11, 13],
            cached_tokens=[5, 7],
            cached_tokens_details=[{"device": 5}, {"device": 7}],
            time_stats=["time-0", "time-1"],
            placeholder_tokens_idx=[[0], [1]],
            placeholder_tokens_val=[[2], [3]],
            retraction_counts=[2, 3],
            pooled_hidden_states=["pooled-0", "pooled-1"],
        )

        split = _handle_output_by_index(output, 1)

        self.assertIsInstance(split, BatchEmbeddingOutput)
        self.assertEqual(split.rids, ["rid-1"])
        self.assertEqual(split.finished_reasons, [{"type": "stop"}])
        self.assertEqual(split.embeddings, [[0.3, 0.4]])
        self.assertEqual(split.prompt_tokens, [13])
        self.assertEqual(split.cached_tokens, [7])
        self.assertEqual(split.cached_tokens_details, [{"device": 7}])
        self.assertEqual(split.time_stats, ["time-1"])
        self.assertEqual(split.placeholder_tokens_idx, [[1]])
        self.assertEqual(split.placeholder_tokens_val, [[3]])
        self.assertEqual(split.retraction_counts, [3])
        self.assertEqual(split.pooled_hidden_states, ["pooled-1"])

    def test_batch_token_id_split_preserves_load_once(self):
        load = object()
        output = BatchTokenIDOutput(
            rids=["rid-0", "rid-1"],
            http_worker_ipcs=["ipc-0", "ipc-1"],
            finished_reasons=[{"type": "length"}, {"type": "stop"}],
            decoded_texts=["foo", "bar"],
            decode_ids=[1, 2],
            read_offsets=[0, 1],
            output_ids=[[10], [20]],
            skip_special_tokens=[True, False],
            spaces_between_special_tokens=[True, True],
            no_stop_trim=[False, True],
            prompt_tokens=[11, 13],
            reasoning_tokens=[0, 1],
            completion_tokens=[3, 5],
            cached_tokens=[7, 9],
            spec_verify_ct=[0, 1],
            spec_num_correct_drafts=[0, 1],
            spec_correct_drafts_histogram=[[], [1]],
            input_token_logprobs_val=[[], []],
            input_token_logprobs_idx=[[], []],
            output_token_logprobs_val=[[], []],
            output_token_logprobs_idx=[[], []],
            input_top_logprobs_val=[[], []],
            input_top_logprobs_idx=[[], []],
            output_top_logprobs_val=[[], []],
            output_top_logprobs_idx=[[], []],
            input_token_ids_logprobs_val=[[], []],
            input_token_ids_logprobs_idx=[[], []],
            output_token_ids_logprobs_val=[[], []],
            output_token_ids_logprobs_idx=[[], []],
            output_token_entropy_val=[0.1, 0.2],
            output_hidden_states=[None, None],
            routed_experts=[None, None],
            indexer_topk=[None, None],
            placeholder_tokens_idx=[None, None],
            placeholder_tokens_val=[None, None],
            retraction_counts=[2, 3],
            token_steps=[[1], [2]],
            load=load,
            customized_info={"tag": ["first", "second"]},
            cached_tokens_details=[{"device": 7}, {"device": 9}],
            dp_ranks=[0, 1],
            time_stats=["time-0", "time-1"],
        )

        first_split = _handle_output_by_index(output, 0)
        second_split = _handle_output_by_index(output, 1)

        self.assertIs(first_split.load, load)
        self.assertIsNone(second_split.load)
        self.assertEqual(first_split.rids, ["rid-0"])
        self.assertEqual(second_split.rids, ["rid-1"])

    def test_batch_str_split_preserves_load_once(self):
        load = object()
        output = BatchStrOutput(
            rids=["rid-0", "rid-1"],
            http_worker_ipcs=["ipc-0", "ipc-1"],
            finished_reasons=[{"type": "length"}, {"type": "stop"}],
            output_strs=["foo", "bar"],
            output_ids=[[10], [20]],
            prompt_tokens=[11, 13],
            reasoning_tokens=[0, 1],
            completion_tokens=[3, 5],
            cached_tokens=[7, 9],
            spec_verify_ct=[0, 1],
            spec_num_correct_drafts=[0, 1],
            spec_correct_drafts_histogram=[[], [1]],
            input_token_logprobs_val=[[], []],
            input_token_logprobs_idx=[[], []],
            output_token_logprobs_val=[[], []],
            output_token_logprobs_idx=[[], []],
            input_top_logprobs_val=[[], []],
            input_top_logprobs_idx=[[], []],
            output_top_logprobs_val=[[], []],
            output_top_logprobs_idx=[[], []],
            input_token_ids_logprobs_val=[[], []],
            input_token_ids_logprobs_idx=[[], []],
            output_token_ids_logprobs_val=[[], []],
            output_token_ids_logprobs_idx=[[], []],
            output_token_entropy_val=[0.1, 0.2],
            output_hidden_states=[None, None],
            routed_experts=[None, None],
            indexer_topk=[None, None],
            placeholder_tokens_idx=[None, None],
            placeholder_tokens_val=[None, None],
            retraction_counts=[2, 3],
            token_steps=[[1], [2]],
            load=load,
            customized_info={"tag": ["first", "second"]},
            cached_tokens_details=[{"device": 7}, {"device": 9}],
            dp_ranks=[0, 1],
            time_stats=["time-0", "time-1"],
        )

        first_split = _handle_output_by_index(output, 0)
        second_split = _handle_output_by_index(output, 1)

        self.assertIs(first_split.load, load)
        self.assertIsNone(second_split.load)
        self.assertEqual(first_split.rids, ["rid-0"])
        self.assertEqual(second_split.rids, ["rid-1"])


if __name__ == "__main__":
    unittest.main()

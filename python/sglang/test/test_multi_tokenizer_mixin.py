import unittest

from sglang.srt.managers.io_struct import BatchEmbeddingOutput
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


if __name__ == "__main__":
    unittest.main()

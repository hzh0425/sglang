import unittest

from sglang.srt.models.apertus import ApertusForCausalLM


class ApertusModelTest(unittest.TestCase):
    def test_get_module_name_from_weight_name_handles_stacked_mapping(self):
        model = ApertusForCausalLM.__new__(ApertusForCausalLM)
        model.stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        module_name, shard_count = model.get_module_name_from_weight_name(
            "model.layers.0.self_attn.q_proj.weight"
        )

        self.assertEqual(module_name, "model.layers.0.self_attn.qkv_proj")
        self.assertEqual(shard_count, 3)


if __name__ == "__main__":
    unittest.main()

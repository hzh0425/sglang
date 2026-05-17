import types
import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers.quantization import mxfp4


class Mxfp4LinearMethodTest(unittest.TestCase):
    def test_apply_accepts_non_contiguous_3d_input(self):
        method = mxfp4.Mxfp4LinearMethod(mxfp4.Mxfp4Config())
        layer = types.SimpleNamespace(
            weight=torch.empty(5, 4),
            weight_scale=torch.empty(1),
        )
        x = torch.randn(2, 4, 3).transpose(1, 2)
        self.assertFalse(x.is_contiguous())

        def fake_quant(tensor, shuffle=True):
            self.assertEqual(tensor.shape, (6, 4))
            return tensor, torch.empty(1)

        def fake_gemm(x_fp4, weight, x_scale, weight_scale):
            return torch.zeros(x_fp4.shape[0], weight.shape[0])

        with mock.patch.object(mxfp4, "mxfp_supported", return_value=True):
            with mock.patch.object(
                mxfp4, "dynamic_mxfp4_quant", side_effect=fake_quant, create=True
            ):
                with mock.patch.object(
                    mxfp4, "gemm_a4w4", side_effect=fake_gemm, create=True
                ):
                    y = method.apply(layer, x)

        self.assertEqual(y.shape, (2, 3, 5))


if __name__ == "__main__":
    unittest.main()

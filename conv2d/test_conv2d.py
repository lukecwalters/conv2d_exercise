import unittest
import numpy as np
from conv2d import Conv2D


class TestConv2D(unittest.TestCase):

    def test_init(self):
        layer = Conv2D()
        layer.kernel = np.random.randn(2, 3, 3, 3).astype(np.float32)
        x_in = np.random.randn(16,3,9,9).astype(np.float32)
        y, xs, ks, ys = layer(x_in)


if __name__ == '__main__':
    unittest.main()
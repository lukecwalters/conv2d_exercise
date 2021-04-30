import unittest
import numpy as np
from conv2d import Conv2D


def conv2d_python_ref(kernel: np.ndarray, x_in: np.ndarray, stride_h=1, stride_w=1):
    """Perform an all python reference implementation of the conv2d op."""

    N, Cin_x, H, W = x_in.shape
    Cout, Cin, Kh, Kw = kernel.shape

    assert Cin_x == Cin, "kernel and x_in have different Cin values"

    Th = int(np.floor((H - Kh + stride_h) / stride_h))  # number of tiles in H dimension
    Tw = int(np.floor((W - Kw + stride_w) / stride_w))  # number of tiles in W dimension

    y_out = np.zeros([N, Cout, Th, Tw], dtype=np.float32)

    for c in range(Cout):
        kh, kw = np.meshgrid(np.arange(Kh), np.arange(Kw))
        kernel_flat = kernel[c, :, kh.ravel(), kw.ravel()].ravel()
        for n in range(N):
            for th, h in enumerate(range(0, Th * stride_h, stride_h)):
                for tw, w in enumerate(range(0, Tw * stride_w, stride_w)):
                    kh, kw = np.meshgrid(np.arange(h, h + Kh), np.arange(w, w + Kw))
                    x_tile = x_in[n, :, kh.ravel(), kw.ravel()].ravel()
                    y_out[n, c, th, tw] = np.dot(kernel_flat, x_tile)

    return y_out


def ser_db(pred, target):
    err = pred.ravel() - target.ravel()
    target_db = 10 * np.log10(np.mean(target.ravel() ** 2) + 1e-16)
    error_db = 10 * np.log10(np.mean(err ** 2) + 1e-16)
    return target_db - error_db


class TestConv2D(unittest.TestCase):
    def test_randn_kernel_input(self):
        for _ in range(10):

            Cout, Cin, Kh, Kw = np.random.randint(1, high=5, size=4)
            N, H, W = np.random.randint(10, 64, size=3)
            stride_h, stride_w = np.random.randint(1, 5, size=2)

            kernel = np.random.randn(Cout, Cin, Kh, Kw).astype(np.float32)
            layer = Conv2D(kernel=kernel, stride_h=stride_h, stride_w=stride_w)

            x_in = np.random.randn(N, Cin, H, W).astype(np.float32)
            yc = layer(x_in)
            yp = conv2d_python_ref(kernel, x_in, stride_h=stride_h, stride_w=stride_w)

            self.assertGreater(ser_db(yc, yp), 135, "Signal to Error ratio too low.")

    def test_post_init_cfg(self):

        Cout, Cin, Kh, Kw = np.random.randint(1, high=5, size=4)
        N, H, W = np.random.randint(10, 64, size=3)
        stride_h, stride_w = np.random.randint(1, 5, size=2)
        kernel = np.random.randn(Cout, Cin, Kh, Kw).astype(np.float32)

        layer = Conv2D(kernel=np.random.randn(1, 2, 3, 4))
        layer.kernel = kernel
        layer.stride_h = stride_h
        layer.stride_w = stride_w

        x_in = np.random.randn(N, Cin, H, W).astype(np.float32)
        yc = layer(x_in)
        yp = conv2d_python_ref(kernel, x_in, stride_h=stride_h, stride_w=stride_w)

        self.assertGreater(ser_db(yc, yp), 135, "Signal to Error ratio too low.")

    def test_output_noise_variance(self):
        Cout, Cin, Kh, Kw = np.random.randint(1, high=5, size=4)
        H, W = 16, 16
        N = 2048
        stride_h, stride_w = 1, 1

        kernel = np.random.randn(Cout, Cin, Kh, Kw).astype(np.float32)

        output_noise_var = np.abs(np.random.randn())
        layer = Conv2D(
            kernel=kernel,
            stride_h=stride_h,
            stride_w=stride_w,
            output_noise_var=output_noise_var,
        )

        x_in = np.random.randn(N, Cin, H, W).astype(np.float32)
        y_noise = layer(x_in)
        layer.output_noise_var = 0.0  # now set to zero and rerun
        y_none = layer(x_in)
        noise = y_noise - y_none  # get added noise from diff..
        # check variance of added noise against target variance
        self.assertEqual(
            np.round(np.var(noise.ravel()), 2).astype(np.float32),
            np.round(output_noise_var, 2).astype(np.float32),
        )


if __name__ == "__main__":
    unittest.main()
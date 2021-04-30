import ctypes
import os
import warnings
import numpy as np

lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "c/conv2d.so"))


class Conv2D(object):
    """A class for performing multichannel 2D convolution via im2row-like tensor-multiplication.

    An instance is configured with a 4D kernel tensor with shape [Cout, Cin, Kh, Kw],
    and optionally strides for the height and width dimensions.

    """

    def __init__(
        self,
        kernel: np.ndarray,
        stride_h: int = 1,
        stride_w: int = 1,
        output_noise_var: float = 0.0,
    ):
        """Instaniate a callable multichannel 2D convolution layer.

        Args:
            kernel (np.ndarray): A kernel of shape (Cout, Cin, Kw, Kh). Defaults to None.
            stride_h (int, optional): kernel stride along the height dimension. Defaults to 1.
            stride_w (int, optional): kernel stride along the width dimension. Defaults to 1.
            output_noise_var (float, optional): variance of gaussian noise added to output tensor
        """
        self._warn_float32("kernel", kernel.dtype)
        self._kernel = kernel.astype(np.float32)
        self._stride_h = stride_h
        self._stride_w = stride_w
        self._output_noise_std = np.sqrt(output_noise_var)

        self._fun = lib.conv2d
        N, Cin, H, W, Kh, Kw, stride_h, stride_w, Cout, Th, Tw = [ctypes.c_int] * 11
        X, K, Y = 3 * [
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="C_CONTIGUOUS")
        ]

        X_2d, K_2d, Y_2d = 3 * [
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="C_CONTIGUOUS")
        ]

        self._fun.argtypes = [
            N,
            Cin,
            H,
            W,
            Kh,
            Kw,
            stride_h,
            stride_w,
            Cout,
            Th,
            Tw,
            X,
            K,
            Y,
            X_2d,
            K_2d,
            Y_2d,
        ]
        self._fun.restype = None

    @property
    def kernel(self):
        """The instantance's kernel as an np.ndarray with dtype np.float32.

        The setter will throw a warning and cast the provided kernel to np.float32 if not already.

        Returns:
            kernel: The 4D kernel tensor of shape [Cout, Cin, Kw, Kh].
        """
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: np.ndarray):
        assert (
            kernel.ndim == 4
        ), "kernel must be a 4D tensor of shape [Cout, Cin, Kw, Kh]."

        self._warn_float32("kernel", kernel.dtype)
        self._kernel = kernel.astype(np.float32)

    @property
    def stride_h(self):
        """The kernel's integer valued stride along the height dimension."""
        return self._stride_h

    @stride_h.setter
    def stride_h(self, stride_h: int):
        self._stride_h = stride_h

    @property
    def stride_w(self):
        """The kernel's integer valued stride along the width dimension."""
        return self._stride_w

    @stride_w.setter
    def stride_w(self, stride_w: int):
        self._stride_w = stride_w

    @property
    def output_noise_var(self):
        """Variance of output gaussian noise injection."""
        return self._output_noise_std ** 2

    @output_noise_var.setter
    def output_noise_var(self, output_noise_var: float):
        self._output_noise_std = np.sqrt(output_noise_var)

    @staticmethod
    def _warn_float32(name, dtype):
        if dtype != np.float32:
            warnings.warn(f"{name} had dtype={dtype} and was cast to np.float32.")

    def __call__(self, x_in: np.ndarray):
        """Convolve the provided input tensor with the layer's kernel and stride config.

        Args:
            x_in (np.ndarray): An input tensor of shape (batch_size, num_in_chs, height, width).

        Returns:
            y_out (np.ndarray): An output tensor of shape (batch_size, num_out_chs, Th, Tw), where
                Th and Tw are the number of [Kh, Kw,Cin] tiles computed along the height and width dimensions,
                respectively. These are determined at run time as a function of the kernel, stride, and input
                dimensions.
        """

        N, Cin_x, H, W = x_in.shape
        Cout, Cin, Kh, Kw = self.kernel.shape

        assert (
            Cin_x == Cin
        ), f"x_in.shape[1]={Cin_x} does not have the expected # of input channels={Cin}"

        self._warn_float32("x_in", x_in.dtype)
        x_in = x_in.astype("float32")

        Th = int(
            np.floor((H - Kh + self.stride_h) / self.stride_h)
        )  # number of tiles in H dimension
        Tw = int(
            np.floor((W - Kw + self.stride_w) / self.stride_w)
        )  # number of tiles in W dimension

        y_out = np.zeros([N, Cout, Th, Tw], dtype=np.float32)
        # scratch tensors for 2D tensor multiplication
        x_scratch = np.zeros([N * Th * Tw, Kh * Kw * Cin], dtype=np.float32)
        k_scratch = np.zeros([Kh * Kw * Cin, Cout], dtype=np.float32)
        y_scratch = np.zeros([N * Tw * Th, Cout], dtype=np.float32)

        self._fun(
            N,
            Cin,
            H,
            W,
            Kh,
            Kw,
            self.stride_h,
            self.stride_w,
            Cout,
            Th,
            Tw,
            x_in,
            self.kernel,
            y_out,
            x_scratch,
            k_scratch,
            y_scratch,
        )

        if self._output_noise_std:
            noise = self._output_noise_std * np.random.randn(*y_out.shape).astype(
                np.float32
            )

            return y_out + noise
        return y_out

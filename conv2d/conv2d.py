import ctypes
import warnings
import numpy as np

lib = ctypes.cdll.LoadLibrary("./c/conv2d.so")


class Conv2D(object):
    def __init__(self, kernel: np.ndarray = None, stride_h: int = 1, stride_w: int = 1):
        """Instaniate a callable multichannel 2D convolution layer.

        Args:
            kernel (np.ndarray, optional): A kernel of shape (num_out_chs, num_in_chs, 
                kernel_height, kernel_width). Defaults to None.
            stride_h (int, optional): kernel stride along the height dimension. Defaults to 1.
            stride_w (int, optional): kernel stride along the width dimension. Defaults to 1.
        """
        self._kernel = kernel
        self._stride_h = stride_h
        self._stride_w = stride_w

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
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: np.ndarray):
        assert (
            kernel.ndim == 4
        ), "kernel must be a 4D tensor of shape [Cout, Cin, Kw, Kh]."

        self._warn_float32("kernel", kernel.dtype)
        self._kernel = kernel.astype("float32")

    @property
    def stride_h(self):
        return self._stride_h

    @stride_h.setter
    def stride_h(self, stride_h: int):
        self._stride_h = stride_h

    @property
    def stride_w(self):
        return self._stride_w

    @stride_w.setter
    def stride_w(self, stride_w: int):
        self._stride_w = stride_w

    @staticmethod
    def _warn_float32(name, dtype):
        if dtype != np.float32:
            warnings.warn(f"{name} had dtype={dtype} and was cast to np.float32.")

    def __call__(self, x_in: np.ndarray):
        stride = 1

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

        return y_out, x_scratch, k_scratch, y_scratch

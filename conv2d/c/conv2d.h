void conv2d(int N, int Cin, int H, int W, int Kh, int Kw, int stride_h, int stride_w, int Cout, int Th, int Tw,
            float input[N][Cin][H][W], float kernel[Cout][Cin][Kh][Kw], float output[N][Cout][Th][Tw],
            float input_2d[N * Tw * Th][Kh * Kw * Cin],
            float kernel_2d[Kh * Kw * Cin][Cout],
            float output_2d[N * Tw * Th][Cout]);
void output_matrix_to_4D(int N, int Cout, int Th, int Tw,
                         float output[N][Cout][Th][Tw],
                         float output_2d[N * Tw * Th][Cout]);
void matmul_generic(int Ra, int Ca, int Cb, float A[Ra][Ca], float B[Ca][Cb], float Y[Ra][Cb]);
void input_to_conv_matrix(int N, int H, int W, int Th, int Tw, int Kh, int Kw, int Cin, int stride_h, int stride_w,
                          float input[N][Cin][H][W], float input_2d[N * Tw * Th][Kh * Kw * Cin]);
void kernel_to_matrix(int Cout, int Cin, int Kh, int Kw,
                      float kernel[Cout][Cin][Kh][Kw],
                      float kernel_2d[Kh * Kw * Cin][Cout]);
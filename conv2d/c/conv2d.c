#include <stdio.h>
#include <stdlib.h>
#include "conv2d.h"

void conv2d(int N, int Cin, int H, int W, int Kh, int Kw, int stride_h, int stride_w, int Cout, int Th, int Tw,
            float input[N][Cin][H][W], float kernel[Cout][Cin][Kh][Kw], float output[N][Cout][Th][Tw],
            float input_2d[N * Tw * Th][Kh * Kw * Cin],
            float kernel_2d[Kh * Kw * Cin][Cout],
            float output_2d[N * Tw * Th][Cout])
{
    
    input_to_conv_matrix(N, H, W, Th, Tw, Kh, Kw, Cin, stride_h, stride_w, input, input_2d);

    kernel_to_matrix(Cout, Cin, Kh, Kw, kernel, kernel_2d);

    // substitute an optimized matmul routine here in the future for convolution op
    matmul_generic(N * Tw * Th, Kh * Kw * Cin, Cout, input_2d, kernel_2d, output_2d);

    output_matrix_to_4D(N, Cout, Th, Tw, output, output_2d);
}

void output_matrix_to_4D(int N, int Cout, int Th, int Tw,
                         float output[N][Cout][Th][Tw],
                         float output_2d[N * Tw * Th][Cout])
{
    // copy shape 2D output to 4D tensor
    int row;

    for (int co = 0; co < Cout; co++)
    {
        row = 0;
        for (int n = 0; n < N; n++)
        {
            for (int th = 0; th < Th; th++)
            {
                for (int tw = 0; tw < Tw; tw++)
                {
                    output[n][co][th][tw] = output_2d[row][co];
                    row++;
                }
            }
        }
    }
}

void kernel_to_matrix(int Cout, int Cin, int Kh, int Kw,
                      float kernel[Cout][Cin][Kh][Kw],
                      float kernel_2d[Kh * Kw * Cin][Cout])
{
    int row_h;
    // copy kernel to 2D form
    for (int co = 0; co < Cout; co++)
    {
        row_h = 0;
        for (int kh = 0; kh < Kh; kh++)
        {
            for (int kw = 0; kw < Kw; kw++)
            {
                for (int ci = 0; ci < Cin; ci++)
                {
                    kernel_2d[row_h++][co] = kernel[co][ci][kh][kw];
                }
            }
        }
    }
}

void input_to_conv_matrix(int N, int H, int W, int Th, int Tw, int Kh, int Kw, int Cin, int stride_h, int stride_w,
                          float input[N][Cin][H][W], float input_2d[N * Tw * Th][Kh * Kw * Cin])
{
    // copy input tensor to 2D for matmul
    int Th_offset, Tw_offset, row_x, col_x;
    row_x = 0;
    for (int n = 0; n < N; n++)
    {
        for (int th = 0; th < Th; th++)
        {
            Th_offset = th * stride_h; // starting h index of tile
            for (int tw = 0; tw < Tw; tw++)
            {
                Tw_offset = tw * stride_w; // starting w index of tile
                col_x = 0;
                // loop over full area and depth of current tile for row_x
                for (int kh = Th_offset; kh < Kh + Th_offset; kh++)
                {
                    for (int kw = Tw_offset; kw < Kw + Tw_offset; kw++)
                    {
                        for (int c = 0; c < Cin; c++)
                        {
                            input_2d[row_x][col_x++] = input[n][c][kh][kw];
                        }
                    }
                }
                row_x++; // a new row for every example-tile tuple, or for every (n,th,tw)
            }
        }
    }
}

void matmul_generic(int Ra, int Ca, int Cb, float A[Ra][Ca], float B[Ca][Cb], float Y[Ra][Cb])
{
    float accum = 0;

    for (int r = 0; r < Ra; r++)
    {
        for (int cb = 0; cb < Cb; cb++)
        {
            for (int ca = 0; ca < Ca; ca++)
            {
                accum += A[r][ca] * B[ca][cb];
            }

            Y[r][cb] = accum;
            accum = 0;
        }
    }
}
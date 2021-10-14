import numpy as np

from Compiler.mpc_math import log_fx
from Compiler.mpc_math import cos
from Compiler.mpc_math import sin
from Compiler.mpc_math import sqrt
from Compiler.types import *
from Compiler.library import *

class Layers:

    def __init__(self):
        self.layers = []


    def add_layer(self, l):
        self.layers.append(l)


    def forward(self, input):

        processed_input = input

        for l in self.layers:
            print_ln("entering layer")
            #print(processed_input)
            processed_input = l.compute(processed_input)
            # print_ln("%s", processed_input.reveal_nested())
            if l.flatten_after:
                processed_input = flatten(processed_input)

        return processed_input

    def get_final_dim(self):
        return self.layers[-1].output_shape


class Layer:

    def __init__(self, input_shape, output_shape, flatten_after=False):
        self.flatten_after = flatten_after
        self.input_shape = input_shape
        self.output_shape = output_shape


class Dense(Layer):

    def __init__(self, input_shape, output_shape, w, b, activation, flatten_after=False):
        super(Dense, self).__init__(input_shape, output_shape, flatten_after)
        # TODO should be shape, but Arrays have no shape...
        self.w_shape = len(w)
        self.w = w

        self.b_shape = len(b)
        self.b = b

        self.activation = activation

    def compute(self, input):
        print(input)

        # TODO currently assumes 1d input/output
        w = self.w
        b = self.b

        output = sfix.Array(self.output_shape)

        @for_range_parallel(self.w_shape, self.w_shape)
        def _(i):
            output[i] = self.activation(dot_1d(input, self.w[i]) + self.b[i])

        print("dense")

        print(output)

        return output


class MaxPooling1D(Layer):

    def __init__(self, input_shape, output_shape, width, filter_dim, flatten_after=False):
        super(MaxPooling1D, self).__init__(input_shape, output_shape, flatten_after)
        self.width = width
        self.filter_dim = filter_dim
        # TODO: padding, stride

    def compute(self, input):

        width = self.width
        filter_dim = self.filter_dim
        output_width = len(input[0]) // width
        left_out_elements = len(input[0]) % width

        # assert filter_dim, output_width == self.output_shape

        output = sfix.Tensor((filter_dim, output_width))
        @for_range_opt((filter_dim, output_width - 1))
        def _(i, j):
            # TODO currently, for Tensors where the width does not divide the input dim properly,
            #  we ignore values fix this
            val = sfix.Array(width)
            @for_range_opt(width)
            def _(k):
                val[k] = input[i][j * width + k]

            output[i][j] = max(val)

        @for_range_opt(filter_dim)
        def _(i):
            # TODO currently, for Tensors where the width does not divide the input dim properly,
            #  we ignore values fix this
            val = sfix.Array(width)

            @for_range_opt(width + left_out_elements)
            def _(k):
                val[k] = input[i][(output_width - 1) * width + k]

            output[i][(output_width - 1)] = max(val)

        # print("maxpool")
        print(output)

        return output


class Conv1D(Layer):

    def __init__(self, input_shape, output_shape, kernels, kernel_bias, activation, stride=None, flatten_after=False):
        super(Conv1D, self).__init__(input_shape, output_shape, flatten_after)
        self.activation = activation
        self.kernel_bias = kernel_bias
        self.kernels = kernels  # multi dimensioned because of multiple filters
        self.filters = len(kernels)  # size
        self.kernel_w = len(kernels[0][0])  # prev filters_dim or input height
        self.kernel_h = len(kernels[0])

        # TODO: padding, stride


    def compute(self, input):
        # print(input)

        kernels = self.kernels
        kernels_bias = self.kernel_bias
        k_width = self.kernel_w
        # print(k_width)
        output_width = len(input[0]) - k_width + 1

        # assert self.filters, output_width == self.output_shape

        output = sfix.Tensor((self.filters, output_width))
        # print("first time")
        # print(output)

        @for_range_opt((self.filters, output_width))
        def _(i, j):
            val = sfix.Matrix(self.kernel_h, self.kernel_w)
            @for_range_opt(self.kernel_h)
            def _(k):
                @for_range_opt(self.kernel_w)
                def _(e):
                    val[k][e] = input[k][e + j]  # optimize by doing things in-place?
            # print(kernels[j])
            output[i][j] = self.activation(dot_2d(val, kernels[i]) + kernels_bias[i])

        # print("conv")

        print(output)

        return output


# TODO optimize
def max(x):
    max_value = sfix.Array(1)
    max_value[0] = x[0]
    @for_range(len(x) - 1)
    def _(i):
        cmp = max_value[0] > x[i + 1]
        max_value[0] = cmp * max_value[0] + (1 - cmp) * x[i + 1]

    return max_value[0]


# TODO only works with 2d to 1
def flatten(x):
    w = len(x)
    h = len(x[0])

    new_array = sfix.Array(w * h)

    @for_range_opt((w, h))
    def _(i,j):
        new_array[i + j * w] = x[i][j]

    return new_array


def dot_1d(x,y):

    return sum(x * y)

    # res = sfix.Array(1)
    # res[0] = sfix(0)
    #
    # @for_range(len(x))
    # def _(i):
    #     res[0] += x[i] * y[i]
    #
    # return res[0]


def dot_2d(x,y):
    res = sfix.Array(1)
    res[0] = sfix(0)

    # print(x[0])
    # print(y[0])

    assert len(x) == len(y)
    assert len(x[0]) == len(y[0])

    # c = sfix.Array(len(x[0]))

    # WARNING: Consider removing parallelization if the results are looking incorrect
    @for_range_parallel(len(x), len(x))
    def _(i):
        c = sum(x[i] * y[i])
        res[0] += c

    return res[0]

    # @for_range(len(x))
    # def _(i):
    #     @for_range(len(x[0]))
    #     def _(j):
    #         prod = x[i][j] * y[i][j]
    #         res[0] += prod
    #
    # return res[0]










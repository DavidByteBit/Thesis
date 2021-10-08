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
            #print(processed_input)
            processed_input = l.compute(processed_input)
            if l.flatten_after:
                processed_input = flatten(processed_input)

        return processed_input


class Layer:

    def __init__(self, input_shape, output_shape, flatten_after=False):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.flatten_after = flatten_after

    def get_shapes(self):
        return self.input_shape, self.output_shape


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

        # TODO currently assumes 1d input/output
        w = self.w
        b = self.b

        output = sfix.Array(self.output_shape)

        @for_range_opt(self.w_shape)
        def _(i):
            output[i] = self.activation(np.dot(input, self.w[i]) + self.b[i])

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

        input_T = transpose(input, (self.input_shape[1], self.input_shape[0]))

        width = self.width
        filter_dim = self.filter_dim
        output_width = len(input) // width

        # assert output_width, filter_dim == self.output_shape

        output = sfix.Tensor((output_width, filter_dim))

        @for_range_opt([filter_dim, output_width//2])
        def _(i, j):
            # TODO currently, for Tensors where the width does not divide the input dim properly,
            #  we ignore values fix this

            val = sfix.Array(width)
            @for_range_opt(width)
            def _(k):
                val[k] = input[i][i * width + k]

            output[j][i] = max(val)

        print("maxpool")
        print(output)

        return output


class Conv1D(Layer):

    def __init__(self, input_shape, output_shape, kernels, kernel_bias, stride=None, flatten_after=False):
        super(Conv1D, self).__init__(input_shape, output_shape, flatten_after)
        self.kernel_bias = kernel_bias
        self.kernels = kernels  # multi dimensioned because of multiple filters
        self.filters = len(kernels)  # size
        self.kernel_w = len(kernels[0][0])  # prev filters_dim or input height
        self.kernel_h = len(kernels[0])

        # TODO: padding, stride


    def compute(self, input):
        kernels = self.kernels
        kernels_bias = self.kernel_bias
        k_width = self.kernel_w
        # print(k_width)
        output_width = len(input) - k_width + 1
        output = sfix.Tensor((output_width, self.filters))
        # print("first time")
        # print(output)

        @for_range_opt((output_width, self.filters))
        def _(i, j):
            val = sfix.Matrix(k_width, self.kernel_w)
            @for_range_opt(k_width)
            def _(k):
                @for_range_opt(self.kernel_h)
                def _(e):
                    val[k][e] = input[i + k][e]
            print(kernels[j])
            output[i] = dot_2d(val, kernels[j]) + kernels_bias[j]

        print("conv")

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
        new_array[i * h + j] = x[i][j]


def transpose(x, shape):
    x_T = sfix.Tensor(shape)

    @for_range(shape[0])
    def _(i):
        @for_range(shape[1])
        def _(j):
            x_T[i][j] = x[j][i]


def dot_2d(x,y):
    res = sfix.Array(1)
    res[0] = sfix(0)

    @for_range(len(x))
    def _(i):
        @for_range(len(x[0]))
        def _(j):
            res[0] += x[i][j] * y[i][j]

    return res[0]

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


def relu(x):
    return max(0.0, x)








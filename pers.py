from Compiler import mpc_math
from Compiler import ml

from Compiler.types import *
from Compiler.library import *

def dot_product(a, b):
    assert (len(a) == len(b))

    return sum(a * b)


def Euclid(x):
    ret_x = sint(0)

    for k in range(len(x)):
        ret_x += x[k] * x[k]

    ret_x = mpc_math.sqrt(ret_x)

    return ret_x


def personalization(layers, source, target, label_space):

    print_ln("checkpoint 1")

    source_size = len(source[0])
    target_size = len(target[0])

    window_size = len(source[0][0])

    feat_size = len(source[0][0][0])

    data_size = target_size + source_size

    output_dim = layers.get_final_dim
    print_ln("checkpoint 2")
    # Data and labels run parallel to each other
    data = MultiArray([data_size, window_size, feat_size], sfix)
    labels = sint.Array(data_size)
    print_ln("checkpoint 3")
    # Line 1
    @for_range(source_size)
    def _(i):
        @for_range(window_size)
        def _(j):
            @for_range(feat_size)
            def _(k):
                data[i][j][k] = source[0][i][j][k]
        labels[i] = source[1][i]

    print_ln("checkpoint 4")
    @for_range(target_size)
    def _(i):
        @for_range(window_size)
        def _(j):
            @for_range(feat_size)
            def _(k):
                data[i + source_size][j][k] = target[0][i][j][k]
        labels[i + source_size] = target[1][i]

    print_ln("checkpoint 5")
    weight_matrix = sfix.Matrix(len(label_space), feat_size)
    print_ln("checkpoint 6")

    @for_range(len(label_space))  # Line 2
    def _(j):
        num = sint.Array(output_dim)  # Length may need to be dynamic.
        dem = sint.Array(1)
        dem[0] = sint(0)
        print_ln("checkpoint 7")
        @for_range(data_size)  # Line 3
        def _(i):
            print_ln("checkpoint 8")
            eq_res = (sint(j) == labels[i])  # Line 4
            feat_res = layers.forward(data[i])  # Line 5
            print_ln("checkpoint 9")
            scalar = sint.Array(output_dim)
            @for_range(output_dim)
            def _(k):
                scalar[k] = eq_res

            print_ln("checkpoint 10")
            num_intermediate = scalar * feat_res  # Line 6

            dem[0] += eq_res  # Line 7
            @for_range(output_dim)
            def _(k):
                num[k] += num_intermediate[k]  # line 8

        dem_extended = sint.Array(len(num))
        for k in range(len(num)):  # Line 9
            dem_extended[k] = dem

        W_intermediate_1 = sfix.Array(len(num))
        for k in range(len(num)):  # Line 10
            W_intermediate_1[k] = num[k] / dem_extended[k]
        W_intermediate_2 = Euclid(W_intermediate_1)  # Line 11

        for k in range(len(num)):  # Line 12
            weight_matrix[j][k] = W_intermediate_1[k] / W_intermediate_2

    return weight_matrix  # Line 13


def infer(layers, weight_matrix, unlabled_data):
    data_feature = layers.forward(unlabled_data)  # Line 1

    rankings = sfix.Array(len(data_feature))

    @for_range_opt(len(data_feature))  # Line 2
    def _(j):
        rankings[j] = dot_product(weight_matrix[j], data_feature)  # Line 3

    return ml.argmax(rankings) # Line 4,5


#####################################################################################

# # CONSTANTS TO MAKE SURE THINGS WORK
# def feature_extractor(x):
#     a = Array(len(x), sint)
#     return a
#
#
# source = (Matrix(3, 2, sfix), Array(3, sint))
# source[0][0][0] = 0
# source[0][1][0] = 1
# source[0][0][1] = 2
# source[0][1][1] = 3
# source[0][2][0] = 4
# source[0][2][1] = 5
#
# source[1][0] = sint(0)
# source[1][1] = sint(1)
# source[1][2] = sint(0)
#
# target = (Matrix(2, 2, sfix), Array(2, sint))
# target[0][0][0] = 0
# target[0][1][0] = 1
# target[0][0][1] = 2
# target[0][1][1] = 3
#
# target[1][0] = sint(1)
# target[1][1] = sint(0)
#
# target_unlabled = Array(2, sfix)
# target_unlabled[0] = 0
# target_unlabled[1] = 1
#
# label_space = [sint(0), sint(1)]
#
# W = personalization(feature_extractor, source, target, label_space)
#
# predicted_label = infer(feature_extractor, W, target_unlabled)
#
# print_ln(str(predicted_label.reveal()))

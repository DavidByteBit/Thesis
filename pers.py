from Compiler import mpc_math
from Compiler import ml


def dot_product(a, b):
    assert (len(a) == len(b))

    return sum(a * b)


def Euclid(x):
    ret_x = sint(0)

    for k in range(len(x)):
        ret_x += x[k] * x[k]

    ret_x = mpc_math.sqrt(ret_x)

    return ret_x


def personalization(feature_extractor, source, target, label_space):
    cols = 2

    source_size = len(source[0])
    target_size = len(target[0])
    data_size = target_size + source_size

    # Data and labels run parallel to each other
    data = Matrix(data_size, cols, sfix)
    labels = Array(data_size, sint)

    # Line 1
    @for_range_opt(source_size)
    def _(i):
        for j in range(cols):
            data[i][j] = source[0][i][j]
        labels[i] = source[1][i]

    @for_range_opt(target_size)
    def _(i):
        for j in range(cols):
            data[i + source_size][j] = target[0][i][j]
        labels[i + source_size] = target[1][i]

    weight_matrix = Matrix(len(label_space), cols, sfix)

    @for_range_opt(len(label_space))  # Line 2
    def _(j):
        num = Array(cols, sint)  # Length may need to be dynamic.
        dem = sint(0)
        for i in range(data_size):  # Line 3
            eq_res = (sint(j) == labels[i])  # Line 4
            feat_res = feature_extractor(data[i])  # Line 5

            scalar = Array(len(feat_res), sint)
            for k in range(len(feat_res)):
                scalar[k] = eq_res

            num_intermediate = scalar * feat_res  # Line 6

            dem += eq_res  # Line 7
            for k in range(len(num)):
                num[k] += num_intermediate[k]  # line 8

        dem_extended = Array(len(num), sint)
        for k in range(len(num)):  # Line 9
            dem_extended[k] = dem

        W_intermediate_1 = Array(len(num), sfix)
        for k in range(len(num)):  # Line 10
            W_intermediate_1[k] = num[k] / dem_extended[k]
        W_intermediate_2 = Euclid(W_intermediate_1)  # Line 11

        for k in range(len(num)):  # Line 12
            weight_matrix[j][k] = W_intermediate_1[k] / W_intermediate_2

    return weight_matrix  # Line 13


def infer(feature_extractor, weight_matrix, unlabled_data):
    data_feature = feature_extractor(unlabled_data)  # Line 1

    rankings = Array(len(data_feature), sfix)

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

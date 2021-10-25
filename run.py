import yaml
import subprocess
import sys
import os
import time
import random
import json
import numpy as np
from .CNN.run_cnn import train_CNN
from .CNN.nets import nets
from .data_formatting import spdz_format_cnn
from clear_code import pers
from keras.models import Model

from .networking import client, server

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import path


# # TODO: Find a better place for this... Or just make it a constant... decide
# runtime_results_file = "runtime_results.txt"


# party: "0"
# path_to_public_data_dir: "/home/sikhapentyala/FairSPDZ/data/fold0"
# path_to_private_data: "/home/sikhapentyala/MP-SPDZ/Player-Data/Input-P1-0"
# path_to_this_repo: "/home/sikhapentyala/FairSPDZ"
# path_to_top_of_mpspdz: "/home/sikhapentyala/MP-SPDZ"
# target_id: "auto"
# compile: "true"
# compiler: "-R 64 -Y run"
# VM: "semi2k-party.x -N 2 -p 1 run"
# online: "true"
# my_private_ip: "10.128.0.4"


def run(setting_map_path):
    print("parsing settings map")
    # retrieves the settings of the yaml file the user passed in
    settings_map = _parse_settings(setting_map_path)

    print("loading data")
    # load data from source. Source data is separated by participants
    source_data, target_data = _load_data(settings_map)

    print("pre-processing data")
    # normalize and window the data (no longer separated by participants)
    source_data, target_data = _pre_process_data(settings_map, source_data, target_data)

    print("splitting data of target user into known/unknown subsets for our k-shot classifier")
    # randomly select 'k' instances from the target data for our k-shot classifier
    target_test_data, target_kshot_data = _partition_data(settings_map, target_data)
    _, source_kshot_data = _partition_data(settings_map, source_data)

    print("training (or collecting) CNN")
    # Train CNN where source data is the training data, and target data is the test data: NOTE - also stores CNN
    # in a format that will make it easy to export for in-the-clear use, and MP-SPDZ use
    cnn_acc_res = _train(settings_map, source_data, target_test_data)

    print("personalizing model")
    # personalize model/classify
    pers_result = _personalize_classifier(settings_map, source_kshot_data, target_test_data, target_kshot_data)

    print("cnn acc: " + str(cnn_acc_res))
    print("pers acc: " + str(pers_result))

    print("storing ITC results")
    # Stores our in-the-clear results
    _store_itc_results(settings_map, cnn_acc_res, pers_result)

    print("storing params in MP-SPDZ files")
    # store local params in private files
    _store_secure_params(settings_map, source_kshot_data, target_kshot_data, target_test_data)

    print("distributing metadata")
    # send online params (custom networking)
    metadata = _distribute_Data(settings_map)

    print("editing secure code")
    print(metadata)
    # prep MP-SPDZ code
    _edit_source_code(settings_map, metadata, source_kshot_data)

    print("transferring files to MP-SPDZ library")
    # Write our secure mpc files to the MP-SPDZ library
    _populate_spdz_files(settings_map)

    print("compiling secure code")
    # compile MP-SPDZ code
    _compile_spdz(settings_map)

    print("running secure code... This may take a while")
    # run MP-SPDZ code
    _run_mpSPDZ(settings_map)

    print("validating results")
    # validate results
    _validate_results(settings_map)

    print("Determining the accuracy of the MP-SPDZ protocol")
    # Take the predicted labels of the spdz protocol and comapre them against the ground truth
    mpc_accuracy = _compute_spdz_accuracy(settings_map, target_test_data)


def _compute_spdz_accuracy(settings_map, target_test_data):
    class_path = settings_map["path_to_this_repo"] + "/storage/results/mpc/classifications.save"

    classifications = ""

    with open(class_path, 'r') as stream:
        # should be one line
        for line in stream:
            classifications += line

    classifications = json.loads(classifications.replace("\n", "").replace("\'", ""))
    print(len(classifications))
    correct = 0

    for i in range(len(classifications)):
        print(np.argmax(target_test_data[1][i]))
        print(int(classifications[i]))
        correct += int(int(classifications[i]) == np.argmax(target_test_data[1][i]))

    accuracy = float(correct) / float(len(classifications))

    print(accuracy)

    return accuracy



def _validate_results(settings_map):
    tolerance = float(settings_map["validation_threshold"]) / 100.0
    path_to_this_repo = settings_map["path_to_this_repo"]

    import json

    itc_wm_path = path_to_this_repo + "/storage/results/itc/weight_matrix.save"

    itc_wm = ""
    with open(itc_wm_path, "r") as stream:
        for line in stream:
            itc_wm += line.replace("\'", "").replace("\"", "")

    itc_wm = json.loads(itc_wm)

    itc_fp_path = path_to_this_repo + "/storage/results/itc/forward_pass.save"

    itc_fp = ""
    with open(itc_fp_path, "r") as stream:
        for line in stream:
            itc_fp += line.replace("\'", "").replace("\"", "")

    itc_fp = json.loads(itc_fp)

    print(itc_wm)
    print()
    print(itc_fp)
    print()

    mpc_fp_path = path_to_this_repo + "/storage/results/mpc/results.save"
    mpc_results = []
    with open(mpc_fp_path, 'r') as stream:
        for line in stream:
            mpc_results.append(line)

    mpc_results = "".join(mpc_results).replace("\n", "").split("@end")

    print(mpc_results)
    print()

    mpc_wm = str(mpc_results[-1]).replace("\n", "").replace("\'", "")
    mpc_fp = str(mpc_results[:-1]).replace("\n", "").replace("\'", "")
    print(mpc_wm)
    print()
    print(mpc_fp)
    print()

    mpc_wm = json.loads(mpc_wm)
    mpc_fp = json.loads(mpc_fp)

    for i in range(len(itc_fp)):
        valid = __compare_within_range(itc_fp[i], mpc_fp[i], tolerance)
        if not valid:
            print("WARNING, NON-VALID RESULT FOR {a}".format(a=i))

    for i in range(len(itc_wm)):
        valid = __compare_within_range(itc_wm[i], mpc_wm[i], tolerance)
        if not valid:
            print("WARNING, NON-VALID RESULT FOR {a}".format(a=i))


def __compare_within_range(a, b, tolerance):
    valid = True

    assert len(a) == len(b)

    for i in range(len(a)):
        c = a[i]
        d = b[i]

        a_percent = c * tolerance
        a_min = c - a_percent
        a_max = c + a_percent

        b_percent = d * tolerance
        b_min = d - b_percent
        b_max = d + b_percent

        if (b_max - a_min < 0) and (b_min - a_max < 0):
            valid = False
            break


    return valid


def _run_mpSPDZ(settings_map):
    runner = settings_map["VM"]
    is_online = settings_map["online"].lower() == "true"
    path_to_spdz = settings_map['path_to_top_of_mpspdz']
    path_to_this_repo = settings_map["path_to_this_repo"]

    intermediate_results_file = path_to_this_repo + "/tmp.save"

    if settings_map["party"] == "0":
        with open(intermediate_results_file, 'w') as stream:
            stream.write("")
        if is_online:
            run_cmd = "cd {a} && ./{b} -pn {c} -h {d} >> {e}".format(a=path_to_spdz, b=runner,
                                                                    c=settings_map["model_holders_port"],
                                                                    d=settings_map["model_holders_ip"],
                                                                    e=intermediate_results_file
                                                                    )
        else:
            run_cmd = "cd {a} && ./{b} >> {e}".format(a=path_to_spdz, b=runner,
                                                      e=intermediate_results_file)

    else:
        if is_online:
            run_cmd = "cd {a} && ./{b} -pn {c} -h {d}".format(a=path_to_spdz, b=runner,
                                                              c=settings_map["model_holders_port"],
                                                              d=settings_map["model_holders_ip"],
                                                              )
        else:
            run_cmd = "cd {a} && ./{b}".format(a=path_to_spdz, b=runner)

    # print("Starting secure program with command: {a}".format(a=run_cmd))

    subprocess.check_call(run_cmd, shell=True)

    save_file_intermediate = settings_map["path_to_this_repo"] + "/storage/results/mpc/results.save"
    save_file_classifications = settings_map["path_to_this_repo"] + "/storage/results/mpc/classifications.save"

    save_results = ""

    with open(intermediate_results_file, 'r') as stream:
        for line in stream:
            print("WHAT IS BELOW?")
            print(line)
            save_results += line

    save_results = save_results.split("@results")

    with open(save_file_intermediate, 'w') as stream:
        stream.write(save_results[0])

    with open(save_file_classifications, 'w') as stream:
        stream.write(save_results[1])


def _compile_spdz(settings_map):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0" and settings_map["online"].lower() != "true":
        return

    # Compile .mpc program
    c = settings_map["compiler"]
    online = settings_map["online"]

    subprocess.check_call("cp {a}/mpc_code/run.mpc {b}/Programs/Source/run.mpc".
                          format(a=settings_map['path_to_this_repo'], b=settings_map["path_to_top_of_mpspdz"]),
                          shell=True)

    # subprocess.check_call("python3 {a}/compile.py {b} > tmp.txt".format(a=settings_map["path_to_top_of_mpspdz"], b=c),
    #                       shell=True)

    subprocess.check_call("python3 {a}/compile.py {b}".format(a=settings_map["path_to_top_of_mpspdz"], b=c),
                          shell=True)

    # if not online.lower() == "true":
    #     subprocess.check_call("rm tmp.txt", shell=True)


def _populate_spdz_files(settings_map):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0" and settings_map["online"].lower() != "true":
        return

    def getListOfFiles(dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)

            # run.mpc should not go to the Compiler directory
            if "run.mpc" in fullPath or "init" in fullPath:
                continue

            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append((fullPath, entry))

        return allFiles

    dir = settings_map["path_to_this_repo"] + "/mpc_code"

    allFiles = getListOfFiles(dir)

    # puts all mpc files into the MP-SPDZ compiler - this excludes run.mpc
    for path_data in allFiles:
        subprocess.check_call("cp {a} {b}/Compiler/{c}".
                              format(a=path_data[0], b=settings_map["path_to_top_of_mpspdz"], c=path_data[1]),
                              shell=True)

    # Now take care of run.mpc
    subprocess.check_call("cp {a}/mpc_code/run.mpc {b}/Programs/Source/run.mpc".
                          format(a=settings_map['path_to_this_repo'], b=settings_map["path_to_top_of_mpspdz"]),
                          shell=True)


def _edit_source_code(settings_map, all_metadata, data):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0" and settings_map["online"].lower() != "true":
        return

    repo_file_path = settings_map["path_to_this_repo"] + "/mpc_code/run.mpc"

    kshot = settings_map["kshot"]
    shapes = all_metadata
    n_timesteps = data[0].shape[1]
    n_features = data[0].shape[2]
    n_outputs = data[1].shape[1]

    file = []
    found_delim = False
    start_of_delim = 0

    i = 0
    with open(repo_file_path, 'r') as stream:
        for line in stream:
            if not found_delim and "@args" in line:
                start_of_delim = i
                found_delim = True
            i += 1
            file.append(line)

    # TODO: Should not be 50 in general
    compile_args = __format_args(test_data_len=3, kshot=kshot, window_size=n_timesteps, shapes=shapes, n_features=n_features,
                                 n_outputs=n_outputs)

    file[start_of_delim + 1] = "settings_map = {n}\n".format(n=compile_args)
    # print(file[start_of_delim + 1])

    # file as a string
    file = ''.join([s for s in file])

    # print(file)

    # print(file)

    with open(repo_file_path, 'w') as stream:
        stream.write(file)


def __format_args(**kwargs):
    res = "{"

    # # shapes is a special case (list of strings), so we can populate that explicitly
    # shapes = kwargs.pop("shapes")
    # res += "\'{key}\': {value},".format(key="metrics", value=shapes)

    for key in kwargs:
        res += "\'{key}\': \'{value}\',".format(key=key, value=kwargs[key])

    # Omit last comma
    res = res[:-1] + "}"

    return res


def _distribute_Data(settings_map):
    is_model_owner = bool(settings_map["party"] == "0")

    metadata = None

    if is_model_owner:
        metadata = __read_shapes(settings_map)

    all_metadata = None

    if is_model_owner:
        all_metadata = __distribute_as_client(settings_map, metadata)
    else:
        all_metadata = __distribute_as_host(settings_map, metadata)

    all_metadata = str(all_metadata)
    all_metadata = all_metadata.replace("\'", "").replace("\"", "")

    # print("all metadata: {a}".format(a=all_metadata))

    return str(all_metadata)


# ./storage/spdz_compatible/save_model.txt
def __read_shapes(settings_map):
    path_to_this_repo = settings_map["path_to_this_repo"]
    shape_path = path_to_this_repo + "/storage/spdz_compatible/spdz_shapes.save"

    metadata = []

    with open(shape_path, 'r') as f:
        for line in f:
            metadata.append(line.replace("(", "[").replace(")", "]").replace("\n", "")
                            .replace(" ", "").replace(",]", "]"))

    metadata = "[" + ",".join(metadata) + "]"
    print("HOWDY")
    print(metadata)
    return metadata


def __distribute_as_host(settings_map, metadata=None):
    # TODO: Need an exit cond. if not online

    if settings_map["online"].lower() != "true":
        return metadata

    data = server.run(settings_map, introduce=False)  # receive data

    return data.split("@seperate")


def __distribute_as_client(settings_map, metadata):
    # print(metadata)

    if settings_map["online"].lower() == "false":
        return metadata

    client.run(settings_map, metadata, introduce=False)

    return metadata


def _store_secure_params(settings_map, kshot_source_data, khshot_target_data, target_test_data):
    # TODO: These tasks should, ideally, be split up between the parties
    if settings_map["party"] != "0":
        return

    # loads params into intermediate files to be sent to MP-SPDZ files
    spdz_format_cnn.load_payload(settings_map)
    model_params_path = "./storage/spdz_compatible/spdz_cnn.save"

    def flatten(S):
        if not S:
            return S
        if isinstance(S[0], list):
            return flatten(S[0]) + flatten(S[1:])
        return S[:1] + flatten(S[1:])

    all_data = []

    with open(model_params_path, 'r') as stream:
        # Note, should only be one, really long line
        for line in stream:
            all_data.append(line)

    # TODO: Make sure that it's being flattened like I think it is..
    all_data.append(str([float(el) for el in kshot_source_data[0].flatten('C')]))
    all_data.append(str([int(np.argmax(el)) for el in kshot_source_data[1].tolist()]))
    all_data.append(str([float(el) for el in khshot_target_data[0].flatten('C')]))
    all_data.append(str([int(np.argmax(el)) for el in khshot_target_data[1].tolist()]))
    # TODO: should not be 50 in general
    all_data.append(str([float(el) for el in target_test_data[0][:3].flatten('C')]))
    # all_data.append(str([int(np.argmax(el)) for el in target_test_data[1].tolist()]))

    ' '.join(all_data)

    all_data = str(all_data).replace("]", '').replace("[", '').replace(",", '').replace("\'", "")

    with open(settings_map["path_to_private_data"], 'w') as stream:
        stream.write(all_data)


def _store_itc_results(settings_map, cnn_acc_res, pers_result):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0":
        return

    results_filepath = "./storage/results/itc/accuracy.csv"

    result = "-----@start-----\n"
    result += "cnn_acc_res = " + str(cnn_acc_res) + "\n"
    result += "pers_result = " + str(pers_result) + "\n"
    result += "target_id = " + settings_map["target_id"] + "\n"
    result += "net = " + settings_map["net"] + "\n"
    result += "epochs = " + settings_map["epochs"] + "\n"
    result += "kshot = " + settings_map["kshot"] + "\n"
    result += "-----@end-----\n"

    with open(results_filepath, 'a+') as f:
        f.write(result)


def __load_cnn(settings_map, data):
    n_timesteps = data[0].shape[1]
    n_features = data[0].shape[2]
    n_outputs = data[1].shape[1]

    net_string = settings_map["net"]
    model = nets().models[net_string](n_timesteps, n_features, n_outputs)

    cnn_to_load_path = settings_map["cnn_path"]

    model.load_weights(cnn_to_load_path)

    return model


def _personalize_classifier(settings_map, source_data, target_test_data, target_kshot_data):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0":
        return

    n_outputs = source_data[1].shape[1]

    model = __load_cnn(settings_map, source_data)

    model_feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    label_space = [i for i in range(n_outputs)]

    personalizer = pers.personalizer(label_space=label_space, feature_extractor=model_feature_extractor)
    personalizer.initialize_weight_matrix(settings_map, source_data, target_kshot_data)

    __save_weight_matrix(settings_map, personalizer)

    results = []
    correct_results = []

    for i in range(len(target_test_data[0])):
        data = target_test_data[0][i]
        new_data = np.expand_dims(data, axis=0)
        results.append(personalizer.classify(new_data))
        correct_results.append(np.dot(label_space, target_test_data[1][i]))

    # results = personalizer.classify(target_test_data[0])
    # correct_results = []
    #
    # for i in range(len(target_test_data[0])):
    #     correct_results.append(np.dot(label_space, target_test_data[1][i]))
    #
    # print(results)
    # print(correct_results)

    return float(sum([int(results[i] == correct_results[i]) for i in range(len(results))])) / len(target_test_data[0])


def __save_weight_matrix(settings_map, personalizer):
    weight_path = settings_map["path_to_this_repo"] + "/storage/results/itc/weight_matrix.save"

    matrix = str([['{:.7f}'.format(b) for b in a] for a in personalizer.weight_matrix.tolist()])

    print(matrix)

    with open(weight_path, 'w') as f:
        f.write(matrix)


def _partition_data(settings_map, data):
    features = data[0]
    labels = data[1]

    kshot = int(settings_map["kshot"])

    # Since data is already ohe'd, it is more efficient just to use a dict where the keys are ohe'd vectors
    data_sorted_by_label = {}

    rows_of_data = len(data[0])

    testing_features = []
    testing_labels = []
    kshot_features = []
    kshot_labels = []

    for i in range(rows_of_data):
        sample = features[i]
        # tuple makes it hashable
        ohe_label = tuple(labels[i].tolist())
        if ohe_label not in data_sorted_by_label.keys():
            data_sorted_by_label[ohe_label] = []
        data_sorted_by_label[ohe_label].append(sample)

    for key in data_sorted_by_label.keys():
        rows_of_subset = len(data_sorted_by_label[key])

        # In this context, the holdout refers to the values that should be saved for our k-shot classifier
        holdout_indices = np.random.choice(rows_of_subset, size=kshot, replace=False)
        remaining_indices = [i for i in range(rows_of_subset) if i not in holdout_indices]

        features_of_subset = np.array(data_sorted_by_label[key])

        key_as_np = np.array(list(key))

        # Note to self, probably not the cleanest way of doing this, but it works
        testing_labels_subset = np.array([key_as_np for i in remaining_indices])
        kshot_labels_subset = np.array([key_as_np for i in holdout_indices])

        kshot_features.extend(features_of_subset[np.array(holdout_indices)])
        kshot_labels.extend(kshot_labels_subset)
        testing_features.extend(features_of_subset[np.array(remaining_indices)])
        testing_labels.extend(testing_labels_subset)

    testing_features = np.array(testing_features)
    testing_labels = np.array(testing_labels)
    kshot_features = np.array(kshot_features)
    kshot_labels = np.array(kshot_labels)

    # print(testing_features.shape)
    # print(testing_labels.shape)
    # print(kshot_features.shape)
    # print(kshot_labels.shape)

    test = []
    holdout = []

    test.append(testing_features)
    test.append(testing_labels)
    holdout.append(kshot_features)
    holdout.append(kshot_labels)

    return test, holdout


def _train(settings_map, source_data, target_test_data):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0":
        return

    accuracy = None

    # if we are not training, then simply classify target_data using the pre-built model
    if settings_map["train_cnn"].lower() != "true":
        # TODO: Upload correct weights...
        model = __load_cnn(settings_map, source_data)

        results = model.evaluate(target_test_data[0], target_test_data[1], verbose=0)
        accuracy = results[1]
    else:
        epochs = int(settings_map["epochs"])

        # Note that training the CNN using this function will save model parameters for later use by in-the-clear code
        accuracy = train_CNN(source_data[0], source_data[1],
                             target_test_data[0], target_test_data[1]).run_experiment(settings_map, repeats=1,
                                                                                      epochs=epochs)

    # Store the model in a different file format than it is currently saved for MP-SPDZ
    __store_cnn_SPDZ_format(settings_map, source_data)

    return accuracy


def __store_cnn_SPDZ_format(settings_map, data):
    model = __load_cnn(settings_map, data)

    # print(model.summary())

    layers_w = model.get_weights()
    layer_str = ""
    layer_wstr = ""

    for layer in layers_w:
        layer_str += str(layer.tolist()) + "\n@end\n"
        layer_wstr += str(layer.shape) + "\n"
        # print(layer.shape)

    with open("./storage/spdz_compatible/save_model.txt", 'w') as f:
        # f.write(layer_wstr)
        # f.write("@end")
        f.write(layer_str)


def __ohe(labels):
    # Assumption: labels range from 0 to n (sequentially), resulting in n + 1 total labels
    different_labels = max(labels).astype(int) + 1
    return np.array([[int(i == num.astype(int)) for i in range(different_labels)] for num in labels])


def _pre_process_data(settings_map, source_data, target_data):
    source_data_norm, source_labels, mean, std = __normalize(source_data)
    target_data_norm, target_labels, _, _ = __normalize([target_data], mean, std)
    return __window_data(settings_map, source_data_norm, source_labels), __window_data(settings_map, target_data_norm,
                                                                                       target_labels)


def __window_data(settings_map, data, labels):
    window_size = int(settings_map["time_slice_len"])

    windowed_data = [[], []]

    row_quantity = len(data)

    for i in range(row_quantity // window_size):
        start_index = i * window_size
        end_index = (i + 1) * window_size

        # Otherwise we'd be out of bounds
        if end_index < row_quantity:
            start_label = labels[start_index]
            end_label = labels[end_index]
            # If the labels match, then there is enough data to make a full window of data
            if start_label == end_label:
                windowed_data[0].append(data[start_index:end_index])
                windowed_data[1].append(start_label)

    windowed_data[0] = np.array(windowed_data[0])
    # one hot encodes each label, making this vector into a matrix
    windowed_data[1] = __ohe(windowed_data[1])

    return windowed_data


def __normalize(data, mean=None, std=None):
    """
    :param data: Data from participants.
    :param mean: mean used to calculate norm. If None, calculates mean
    :param std: std used to calculate norm. If None, calculates std
    :return: Normalized data stacked as single matrix, and the labels.
             Also returns the mean and std used to normalize the data.
    """
    stacked_data = data[0]

    for i in range(len(data) - 1):
        stacked_data = np.vstack((stacked_data, data[i + 1]))

    labels = stacked_data[:, -1]  # for last column
    stacked_data = stacked_data[:, :-1]  # for all but last column

    calculate_stats = False
    if mean is None and std is None:
        calculate_stats = True

    if calculate_stats:
        mean = stacked_data.mean(axis=0)
        std = stacked_data.std(axis=0)

    col_wise_stacked_data = stacked_data.T
    col_wise_stacked_data_normalized = []

    for i in range(len(col_wise_stacked_data)):
        col = col_wise_stacked_data[i]
        col_wise_stacked_data_normalized.append((col - mean[i]) / std[i])

    col_wise_stacked_data_normalized = np.array(col_wise_stacked_data_normalized)

    return col_wise_stacked_data_normalized.T, labels, mean, std


def _load_data(settings_map):
    path_to_data = settings_map["path_to_public_data_dir"]
    target_id = int(settings_map["target_id"])

    participants = 0
    participant_files = []

    for filename in os.listdir(path_to_data):
        participants += 1
        p_id = ""
        # id substring needs to be located at the end of the filename for this to work
        for i in range(len(filename)):
            # -5 because we assume files have a .csv extension
            ch = filename[-5 - i]
            if ch.isnumeric():
                p_id = ch + p_id
            else:
                break
        participant_files.append((filename, int(p_id)))

    source_data = []
    target_data = None

    # does this work for tuples?
    for file, p_id in participant_files:
        file_path = path_to_data + "/" + file
        if p_id != target_id:
            source_data.append(np.genfromtxt(file_path, delimiter=','))
        else:
            target_data = np.genfromtxt(file_path, delimiter=',')

    return source_data, target_data


def _parse_settings(setting_map_path):
    settings_map = None

    with open(setting_map_path, 'r') as stream:
        try:
            settings_map = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return settings_map

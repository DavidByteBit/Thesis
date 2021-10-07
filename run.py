import yaml
import subprocess
import sys
import os
import time

from os import path

# # TODO: Find a better place for this... Or just make it a constant... decide
# runtime_results_file = "runtime_results.txt"

def run(setting_map_path):
    # Path to settings file

    # print("parsing settings")
    settings_map = _parse_settings(setting_map_path)

    # print("validating settings")
    # validate_settings(settings_map)
    #
    # # print("processing data and retrieving its metadata")
    # metadata = _getMetaData(settings_map)
    #
    # print("distributing data")
    # all_metadata = _distribute_Data(settings_map, metadata)

    # print("Compiling secure program -- This may take several minutes\n\n")
    _setup_compilation(settings_map, all_metadata)

    # print("Compilation successful, running secure code")
    _run_mpSPDZ(settings_map)

    # _print_results(settings_map)


# def _print_results(setting_map):
#     start = False
#     res = []
#
#     with open("{a}/{b}".format(a=setting_map["path_to_this_repo"], b=runtime_results_file), 'r') as file:
#         for line in file:
#
#             if line == "@end":
#                 break
#
#             if start:
#                 res.append(line)
#             elif line == "@results":
#                 start = True
#
#     res = ''.join([s for s in res])
#
#     print(res)




def _setup_compilation(settings_map, all_metadata):
    if settings_map["online"].lower() == "false":
        if settings_map["type_of_data"] == "model":
            __edit_source_code(settings_map, all_metadata)
            __compile_spdz(settings_map)

        # This 'else' is fairly sloppy. Since terminal A is compiling, we need a way for terminal B
        # to wait for it. Only issue is that they run in different processes, so the process running
        # code on terminal B can't check the state of the data on terminal A. Thus, we can use the
        # existence of a file as a flag for terminal B. TODO: See if there is a better way to accomplish this...
        else:
            time.sleep(2)
            while path.exists("tmp.txt"):
                time.sleep(0.1)

    else:
        __edit_source_code(settings_map, all_metadata)
        __compile_spdz(settings_map)


# TODO: This currently only tested for LR
def __edit_source_code(settings_map, all_metadata):
    mpc_file_path = settings_map["path_to_this_repo"] + "/mpc_code/run.mpc"

    # 'command line arguments' for our .mpc file
    num_of_parties = str(settings_map["num_of_parties"])
    model_type = settings_map["model_type"]
    model_owner_id = settings_map["party_id_of_model_holder"]
    protected_col = settings_map["protected_col"]
    protected_col_vals = settings_map["protected_col_vals"]
    metrics = settings_map["metrics"]
    recipients = settings_map["recipients"]

    # make sure metrics is ready to be read as a json
    metrics = metrics.replace("[", "").replace("]", "").split(",")

    file = []
    found_delim = False
    start_of_delim = 0

    i = 0
    with open(mpc_file_path, 'r') as stream:
        for line in stream:

            if not found_delim and "@args" in line:
                start_of_delim = i
                found_delim = True
            i += 1

            file.append(line)

    compile_args = __format_args(num_of_parties=num_of_parties, model_type=model_type, model_owner_id=model_owner_id,
                                 all_metadata=all_metadata, protected_col=protected_col,
                                 protected_col_vals=protected_col_vals, metrics=metrics, recipients=recipients)

    file[start_of_delim + 1] = "settings_map = {n}\n".format(n=compile_args)

    # file as a string
    file = ''.join([s for s in file])

    # print(file)

    with open(mpc_file_path, 'w') as stream:
        stream.write(file)


def __format_args(**kwargs):
    res = "{"

    # metrics is a special case (list of strings), so we can populate that explicitly
    metrics = kwargs.pop("metrics")
    res += "\'{key}\': {value},".format(key="metrics", value=metrics)

    for key in kwargs:
        res += "\'{key}\': \'{value}\',".format(key=key, value=kwargs[key])

    # Omit last comma
    res = res[:-1] + "}"

    return res


def _run_mpSPDZ(settings_map):
    runner = settings_map["VM"]
    is_online = settings_map["online"].lower() == "true"
    path_to_spdz = settings_map['path_to_top_of_mpspdz']

    if is_online:
        run_cmd = "cd {a} && ./{b} -pn {c} -h {d}".format(a=path_to_spdz, b=runner,
                                                          c=settings_map["model_holders_port"],
                                                          d=settings_map["model_holders_ip"],
                                                          )
    else:
        run_cmd = "cd {a} && ./{b}".format(a=path_to_spdz, b=runner)

    # print("Starting secure program with command: {a}".format(a=run_cmd))

    subprocess.check_call(run_cmd, shell=True)


def __compile_spdz(settings_map):
    # Compile .mpc program
    c = settings_map["compiler"]
    online = settings_map["online"]

    subprocess.check_call("cp {a}/mpc_code/run.mpc {b}/Programs/Source/run.mpc".
                          format(a=settings_map['path_to_this_repo'], b=settings_map["path_to_top_of_mpspdz"]),
                          shell=True)

    # Take the directory tree in the "mpc_code" folder, flatten out the files,
    # and direct it to the Compiler directory in the spdz directory (skips run.mpc)
    __populate_spdz_files(settings_map)

    subprocess.check_call("python3 {a}/compile.py {b} > tmp.txt".format(a=settings_map["path_to_top_of_mpspdz"], b=c),
                          shell=True)

    if not online.lower() == "true":
        subprocess.check_call("rm tmp.txt", shell=True)


def __populate_spdz_files(settings_map):
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

    for path_data in allFiles:
        subprocess.check_call("cp {a} {b}/Compiler/{c}".
                              format(a=path_data[0], b=settings_map["path_to_top_of_mpspdz"], c=path_data[1]),
                              shell=True)


# def _distribute_Data(settings_map, metadata):
#     is_model_owner = bool(settings_map["type_of_data"] == "model")
#
#     all_metadata = None
#
#     # asynchronous execution to distribute data
#     if is_model_owner:
#         all_metadata = __distribute_as_host(settings_map, metadata)
#     else:
#         all_metadata = __distribute_as_client(settings_map, metadata)
#
#     all_metadata = str(all_metadata)
#     all_metadata = all_metadata.replace("\'", "").replace("\"", "")
#
#     # print("all metadata: {a}".format(a=all_metadata))
#
#     return all_metadata


# def __distribute_as_host(settings_map, metadata):
#     parties = int(settings_map["num_of_parties"])
#     party_id = int(settings_map["party"])
#
#     others_ip = {}
#     all_metadata = []
#
#     all_metadata.insert(party_id, metadata)
#
#     # print("setting up server")
#     for i in range(parties - 1):
#         data, other_parties_ip = server.run(settings_map, introduce=True)  # receive data
#
#         other_parties_id = int(data[0])
#         others_metadata = data[1:]
#
#         all_metadata.insert(other_parties_id, others_metadata)
#
#         others_ip[other_parties_id] = other_parties_ip
#
#     all_metadata = "@seperate".join(all_metadata)
#
#     # If we are not online, we do not need to make a connection with the other parties
#     if settings_map["online"].lower() == "false":
#         return str(all_metadata.split("@seperate")).replace("\'", "").replace("\"", "")
#
#     for party in others_ip:
#         # print(party)
#         # print(others_ip[party])
#         # server.run(settings_map, all_metadata)
#         client.run(settings_map, all_metadata, host_ip=others_ip[party], share_party_id=False)
#
#     return all_metadata.split("@seperate")
#
#
# def __distribute_as_client(settings_map, metadata):
#     client.run(settings_map, metadata, introduce=True)
#
#     if settings_map["online"].lower() == "false":
#         return
#
#     all_metadata = server.run(settings_map)[0].split("@seperate")
#     return all_metadata
#
#
# def _getMetaData(settings_map):
#     own_model = bool(settings_map["party_id_of_model_holder"] == settings_map["party"])
#
#     metadata = None
#
#     if own_model:
#         metadata = processModel.logistic_regression(settings_map)
#     else:
#         metadata = __write_data(settings_map)
#
#     return str(metadata)


def __write_data(settings_map):
    public_data_path = settings_map["path_to_public_data"]
    feature_path = public_data_path + "/x.csv"
    label_path = public_data_path + "/y.csv"

    private_data_path = settings_map["path_to_private_data"]

    data = []

    rows = 0
    cols = 0

    grab_features = True

    # Grab audit features
    with open(feature_path, 'r') as stream:
        for line in stream:

            rows += 1

            line = line.replace("\n", "").split(",")

            data.extend(line)

            # TODO: Kind of sloppy, should optimize
            if grab_features:
                cols = len(line)
                grab_features = False

    # Grab audit labels
    with open(label_path, 'r') as stream:
        for line in stream:
            line = line.replace("\n", "")
            data.extend(line)

    # Write data to private file
    with open(private_data_path, 'w') as stream:
        s = " ".join(data)
        stream.write(s)

    return [str(rows), str(cols)]


def _parse_settings(setting_map_path):
    settings_map = None

    with open(setting_map_path, 'r') as stream:
        try:
            settings_map = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # __validate_settings(settings_map)

    return settings_map


# TODO: This is no longer relevant, needs to be updated
def __validate_settings(settings_map):
    error_found = False

    # Validate paths
    if not path.exists(settings_map["path_to_public_data"]):
        error_msg = "Path to public data not found\n" \
                    "In your settings file, check \'path_to_public_data\' to confirm if it is correct\n"

        error_found = True

        print(error_msg)

    if not path.exists(settings_map["path_to_private_data"]):
        error_msg = "\tPath to private data ({n}) not found\n" \
                    "\tIn your settings file, check \'path_to_private_data\' " \
                    "to confirm if it is correct\n".format(n=settings_map["path_to_private_data"])

        error_found = True

        print(error_msg)

    if not path.exists(settings_map["path_to_this_repo"]):
        error_msg = "Path to this repo not found\n" \
                    "In your settings file, check \'path_to_this_repo\' to confirm if it is correct\n"

        error_found = True

        print(error_msg)

    elif not path.exists(settings_map["path_to_top_of_mpspdz"]):
        error_msg = "Path to MP-SPDZ folder not found\n" \
                    "In your settings file, check \'path_to_top_of_mpspdz\' to confirm if it is correct\n"

        error_found = True

        print(error_msg)

    else:
        # Check to see if the correct virtual machine has been compiled
        if not path.exists(settings_map["path_to_top_of_mpspdz"] + "/semi2k-party.x"):
            user_in = input("The virtual machine \'semi2k-party.x\' has not been compiled.\n"
                            "In order to run this code, it must be compiled.\n"
                            "Would you like for us to create it? (y/n):").lower()

            if user_in == "y" or user_in == "yes":
                subprocess.call(settings_map['path_to_this_repo'] + "/bashScripts/compileVM.sh %s" %
                                settings_map["path_to_top_of_mpspdz"],
                                shell=True)

    # Validate variables
    if settings_map["type_of_data"] != "audit" and settings_map["type_of_data"] != "model":
        error_msg = "The type of data is not valid. Only valid options are \'audit\' and \'model\' \n"

        error_found = True

        print(error_msg)

    if settings_map["compile"] != "true" and settings_map["compile"] != "false":
        error_msg = "Compile value not valid. Should be \'true\' or \'false\' \n"

        error_found = True

        print(error_msg)

    # TODO: verify if models and metrics are correct

    if error_found:
        raise Exception("\nThere was an error with the settings file. Please look above to determine what the error "
                        "was\n")

import cplex
import stim
import functools
import operator
import json
import numpy as np
from itertools import product

p = 0.001
num_shots = 1000000

parameters_set = {
    "488": {
        5: {
            "num_qubit": 17,
            "num_face": 8,
            "num_depth": 3,
            "trape_list": [3, 6],
            "left_trape_list": [6],
            "indices_8body_face":[1],
            "g_idx_list": [[1, 2, 6, 7],[2, 3, 7, 8, 11, 12, 14, 15],[3, 4, 8, 9],[4, 5, 9, 13],[6, 7, 10, 11],[8, 9, 12, 13],[10, 11, 14, 16],[14, 15, 16, 17],],
            "g_idx_list_revised":[[0, 1, 5, 6], [1, 2, 6, 7, 10, 11, 13, 14], [2, 3, 7, 8], [3, 4, 8, 12], [5, 6, 9, 10], [7, 8, 11, 12], [9, 10, 13, 15], [13, 14, 15, 16]],
            "cnot_order":[[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,3],[3,3,1,2],[3,3,1,2],[2,1,2,3],[3,3,1,2]]
        },
        7: {
            "num_qubit": 31,
            "num_face": 15,
            "num_depth": 3,
            "trape_list": [0, 12, 13],
            "left_trape_list":[0, 13],
            "indices_8body_face":[2, 4, 10],
            "g_idx_list": [[1, 2, 8, 14],[2, 3, 8, 9],[3, 4, 9, 10, 15, 16, 20, 21],[4, 5, 10, 11],[5, 6, 11, 12, 17, 18, 22, 23],[6, 7, 12, 13],[8, 9, 14, 15],[10, 11, 16, 17],[12, 13, 18, 19],[20, 21, 24, 25],[16, 17, 21, 22, 25, 26, 28, 29],[22, 23, 26, 27],[18, 19, 23, 27],[24, 25, 28, 30],[28, 29, 30, 31],],
            "g_idx_list_revised":[[0, 1, 7, 13], [1, 2, 7, 8], [2, 3, 8, 9, 14, 15, 19, 20], [3, 4, 9, 10], [4, 5, 10, 11, 16, 17, 21, 22], [5, 6, 11, 12], [7, 8, 13, 14], [9, 10, 15, 16], [11, 12, 17, 18], [19, 20, 23, 24], [15, 16, 20, 21, 24, 25, 27, 28], [21, 22, 25, 26], [17, 18, 22, 26], [23, 24, 27, 29], [27, 28, 29, 30]],
            "cnot_order":[[2,1,2,3],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[3,3,1,2],[3,3,1,2],[3,3,1,2],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,3],[2,1,2,3],[3,3,1,2]]
        },
        9: {
            "num_qubit": 49,
            "num_face": 24,
            "num_depth": 3,
            "trape_list": [7, 12, 21, 22],
            "left_trape_list": [12, 22],
            "indices_8body_face":[1, 3, 5, 14, 16, 19],
            "g_idx_list": [[1,2,10,11],[2,3,11,12,19,20,26,27],[3,4,12,13],[4,5,13,14,21,22,28,29],[5,6,14,15,],[6,7,15,16,23,24,30,31],[7,8,16,17],[8,9,17,25],[10,11,18,19],[12,13,20,21],[14,15,22,23],[16,17,24,25],[18,19,26,32],[26,27,32,33],[20,21,27,28,33,34,38,39],[28,29,34,35],[22,23,29,30,35,36,40,41],[30,31,36,37],[38,39,42,43],[34,35,39,40,43,44,46,47],[40,41,44,45],[36,37,41,45],[42,43,46,48],[46,47,48,49]],
            "g_idx_list_revised":[[0, 1, 9, 10], [1, 2, 10, 11, 18, 19, 25, 26], [2, 3, 11, 12], [3, 4, 12, 13, 20, 21, 27, 28], [4, 5, 13, 14], [5, 6, 14, 15, 22, 23, 29, 30], [6, 7, 15, 16], [7, 8, 16, 24], [9, 10, 17, 18], [11, 12, 19, 20], [13, 14, 21, 22], [15, 16, 23, 24], [17, 18, 25, 31], [25, 26, 31, 32], [19, 20, 26, 27, 32, 33, 37, 38], [27, 28, 33, 34], [21, 22, 28, 29, 34, 35, 39, 40], [29, 30, 35, 36], [37, 38, 41, 42], [33, 34, 38, 39, 42, 43, 45, 46], [39, 40, 43, 44], [35, 36, 40, 44], [41, 42, 45, 47], [45, 46, 47, 48]],
            "cnot_order":[[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,3],[3,3,1,2],[3,3,1,2],[3,3,1,2],[3,3,1,2],[2,1,2,3],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[3,3,1,2],[2,1,1,2,3,3,1,2],[3,3,1,2],[2,1,1,3],[2,1,2,3],[3,3,1,2]]
        },
    }
}

def assign_variables(data_set_name, d_value):
    global d, num_qubit, num_face, num_depth, trape_list,left_trape_list,indices_8body_face,g_idx_list, g_idx_list_revised, cnot_order
    data = parameters_set[data_set_name][d_value]
    # Assigning the chosen d_value to the global variable d
    d = d_value
    num_qubit = data["num_qubit"]
    num_face = data["num_face"]
    num_depth = data["num_depth"]
    trape_list = data["trape_list"]
    left_trape_list = data["left_trape_list"]
    indices_8body_face= data["indices_8body_face"]
    g_idx_list = data["g_idx_list"]
    g_idx_list_revised = data["g_idx_list_revised"]
    cnot_order = data["cnot_order"]

#######Choosing code########
assign_variables("488", 5)
##########################

with open("FWO_488_data.json", "r") as file:
    data_store = json.load(file)
with open("FWO_488_synd.json", "r") as file:
    data_store_synd = json.load(file)

s = [[0 for _ in range(num_face)] for _ in range(d)]
def filter_qubits_based_on_ancilla(indices, ancilla_list, unused_qubits):

    include_numbers = []
    for idx in indices:
        include_numbers.extend(ancilla_list[idx])

    return [qubit for qubit in unused_qubits if qubit not in include_numbers]

def find_indices(i, g_idx_list):
    target = i + 1
    indices = []
    for idx, sublist in enumerate(g_idx_list):
        if target in sublist:
            indices.append(idx)

    return [index for index in indices]

def generate_flag_results_auto(ancilla_index_list, flag_meas, flag_meas_2qubit_cat):
    flag_results_x = []
    idx_1q = 0
    idx_2q = 0
    for sublist in ancilla_index_list:
        if len(sublist) == 2:
            flag_results_x.append(flag_meas_2qubit_cat[idx_2q])
            idx_2q += 1
        elif len(sublist) == 4:
            flag_results_x.append(flag_meas[idx_1q])
            idx_1q += 1

    return flag_results_x

def time_setting(time_value):
    if time_value==0:
        return 0
    else:
        return 1

def convert_bool_to_int(item):
    if isinstance(item, bool):
        return int(item)
    elif isinstance(item, list):
        return [convert_bool_to_int(sub_item) for sub_item in item]
    else:
        return item

def convert_to_string(input_value):
    if isinstance(input_value, (int, float)):
        return str(input_value)
    elif isinstance(input_value, list) and all(isinstance(i, (int, float)) for i in input_value):
        return ''.join(map(str, input_value))
    else:
        raise ValueError("Unsupported input type.")

def compute_plane_estimate(ans, d, num_qubit):
    slices = [ans[i * num_qubit: (i+1) * num_qubit] for i in range(d)]
    plane_estimate = [
        functools.reduce(operator.xor, data_point)
        for data_point in zip(*slices)
    ]

    return plane_estimate

str_data_initialize_rx = "RX " + " ".join([str(i) for i in range(num_qubit)])
str_data_initialize_r = "R " + " ".join([str(i) for i in range(num_qubit)])

str_logical_mpp_x = "MPP " + "*".join(["X" + str(i) for i in range(num_qubit)])
str_logical_mpp_z = "MPP " + "*".join(["Z" + str(i) for i in range(num_qubit)])

def str_stabilizer_mpp_x(g_idx_list_revised):
    output = ""
    for lst in g_idx_list_revised:
        mpp = "MPP "
        for i in range(len(lst)):
            if i == len(lst) - 1:
                mpp += "X" + str(lst[i])
            else:
                mpp += "X" + str(lst[i]) + "*"
        output += mpp + "\n"
    return output.strip()

def str_stabilizer_mpp_z(g_idx_list_revised):
    output = ""
    for lst in g_idx_list_revised:
        mpp = "MPP "
        for i in range(len(lst)):
            if i == len(lst) - 1:
                mpp += "Z" + str(lst[i])
            else:
                mpp += "Z" + str(lst[i]) + "*"
        output += mpp + "\n"
    return output.strip()

def reorder_list_len4_elements(input_list):
    output = []
    for i in range(0, len(input_list), 4):
        chunk = input_list[i:i + 4]
        if len(chunk) == 4:
            reordered_chunk = [chunk[1], chunk[0], chunk[2], chunk[3]]
            output.extend(reordered_chunk)

    return output

def reorder_list_len4_elements_after(input_list):
    output = []
    for i in range(0, len(input_list), 4):
        chunk = input_list[i:i + 4]
        if len(chunk) == 4:
            reordered_chunk = [chunk[1], chunk[3], chunk[2], chunk[0]]
            output.extend(reordered_chunk)

    return output

def reorder_list_len4_elements_z_meas(input_list):
    output = []
    for i in range(0, len(input_list), 4):
        chunk = input_list[i:i + 4]
        if len(chunk) == 4:
            reordered_chunk = [chunk[0], chunk[1], chunk[3], chunk[2]]
            output.extend(reordered_chunk)

    return output

def reorder_list_len4_elements_z_meas_after(input_list):
    output = []
    for i in range(0, len(input_list), 4):
        chunk = input_list[i:i + 4]
        if len(chunk) == 4:
            reordered_chunk = [chunk[3], chunk[1], chunk[0], chunk[2]]
            output.extend(reordered_chunk)

    return output

def extract_specific_indices_len4_elements(input_list):
    output = []
    for i in range(0, len(input_list), 4):
        chunk = input_list[i:i + 4]
        if len(chunk) == 4:
            extracted_elements = [chunk[1], chunk[2]]
            output.extend(extracted_elements)

    return output

def generate_ancilla_index_list(num_qubit, g_idx_list_revised):
    ancilla_index_list = []
    for lst in g_idx_list_revised:
        if len(lst) == 4:
            ancilla_index_list.append([num_qubit, num_qubit + 1])
            num_qubit += 2
        elif len(lst) == 8:
            ancilla_index_list.append([num_qubit, num_qubit + 1, num_qubit + 2, num_qubit + 3])
            num_qubit += 4
    return ancilla_index_list

ancilla_index_list = generate_ancilla_index_list(num_qubit, g_idx_list_revised)

num_flag_2 = sum(1 for sublist in ancilla_index_list if len(sublist) == 2)
num_flag_4 = sum(1 for sublist in ancilla_index_list if len(sublist) == 4)
two_length_indices = [i for i, sublist in enumerate(ancilla_index_list) if len(sublist) == 2]
four_length_indices = [i for i, sublist in enumerate(ancilla_index_list) if len(sublist) == 4]
start_index_2qubit = -(num_face + num_flag_2)

#Generate strings for constructing the circuits
##############For X stabilizer measurement circuit################
def str_cat_ancilla_prep_state_x_meas_error_include(ancilla_index_list, p="{p}"):
    # RX string
    rx_indices = []
    for sublist in ancilla_index_list:
        if len(sublist) == 2:
            rx_indices.append(sublist[0])
        elif len(sublist) == 4:
            rx_indices.append(sublist[1])
    rx_str = "RX " + " ".join(map(str, rx_indices))

    # Z_ERROR string
    z_error_str = "Z_ERROR(" + p + ") " + " ".join(map(str, rx_indices))

    # R string
    all_ancilla = [num for sublist in ancilla_index_list for num in sublist]
    unused_ancilla = list(set(all_ancilla) - set(rx_indices))
    r_str = "R " + " ".join(map(str, sorted(unused_ancilla)))

    # X_ERROR string
    x_error_str = "X_ERROR(" + p + ") " + " ".join(map(str, sorted(unused_ancilla)))
    idel_str = "DEPOLARIZE1(" + p + ") " + " ".join(map(str, list(range(0, num_qubit))))

    return "\n".join([rx_str, z_error_str, r_str, x_error_str, idel_str])

def str_cat_ancilla_prep_cnot_x_meas_error_include(
    ancilla_index_list, num_qubit, p="{p}"
):
    output = []

    len2_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 2 for num in sublist
    ]

    len4_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 4 for num in sublist
    ]

    output.append("CX " + " ".join(map(str, len2_elements)))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, len2_elements)))
    output.append("CX " + " ".join(map(str, extract_specific_indices_len4_elements(len4_elements))))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, extract_specific_indices_len4_elements(len4_elements))))
    depolarize1_numbers = list(range(0, num_qubit))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, depolarize1_numbers)))
    output.append("CX " + " ".join(map(str, reorder_list_len4_elements(len4_elements))))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, reorder_list_len4_elements(len4_elements))))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, depolarize1_numbers)))

    return "\n".join(output)

def str_cat_cnot_x_meas_error_include(
    num_qubit, cnot_order, g_idx_list_revised, ancilla_index_list, p, trape_list,indices_8body_face
):
    output = ""

    #Counting all numbers in ancilla_index_list
    num_total_ancilla = sum(len(sublist) for sublist in ancilla_index_list)

    for val in range(1, num_depth + 1):
        cx_str = "CX"
        used_indices = []

        for index, (cnot, revised) in enumerate(zip(cnot_order, g_idx_list_revised)):
            if val in cnot:
                indices_of_val = [i for i, v in enumerate(cnot) if v == val]
                for idx in indices_of_val:

                    if index in trape_list:
                        if idx in [0, 1]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][0]
                        else:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][1]
                    elif index in indices_8body_face:
                        if idx in [4,6]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][0]
                        elif idx in [5,7]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][1]
                        elif idx in [0,2]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][2]
                        elif idx in [1, 3]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][3]
                    else:
                        if idx in [0, 2]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][0]
                        else:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][1]
                    cx_str += f" {cat_index} {revised[idx]}"
                    used_indices.extend([cat_index, revised[idx]])

        depolarize2 = (
            "\nDEPOLARIZE2(" + str(p) + ") " + " ".join(map(str, used_indices))
        )

        all_qubits = set(range(num_qubit))
        ancilla_qubits = set(range(num_qubit, num_qubit + num_total_ancilla))
        unused_qubits = list((all_qubits.union(ancilla_qubits)) - set(used_indices))

        depolarize1 = (
            "\nDEPOLARIZE1(" + str(p) + ") " + " ".join(map(str, sorted(unused_qubits)))
        )
        output += cx_str + depolarize2 + depolarize1 + "\nTICK\n"

    return output

def str_cat_ancilla_decoding_x_meas_error_include(
    ancilla_index_list, num_qubit, p="{p}"
):
    output = []

    len2_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 2 for num in sublist
    ]
    len4_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 4 for num in sublist
    ]

    output.append("CX " + " ".join(map(str, reorder_list_len4_elements_after(len4_elements))))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, reorder_list_len4_elements_after(len4_elements))))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))))
    output.append("CX " + " ".join(map(str, extract_specific_indices_len4_elements(len4_elements))))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, extract_specific_indices_len4_elements(len4_elements))))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))))
    output.append("CX " + " ".join(map(str, len2_elements)))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, len2_elements)))

    return "\n".join(output)

def str_cat_ancilla_meas_x_meas_error_include(ancilla_index_list, num_qubit, p="{p}"):
    output = []

    len4_elements= [
        sublist[i] for sublist in ancilla_index_list if len(sublist) == 4
    for i in [0, 2, 3] ]

    len2_elements_second = [
        sublist[1] for sublist in ancilla_index_list if len(sublist) == 2
    ]
    output.append("M(" + p + ") " + " ".join(map(str, len4_elements)))
    output.append("M(" + p + ") " + " ".join(map(str, len2_elements_second)))

    combined_elements = []
    for sublist in ancilla_index_list:
        if len(sublist) == 2:
            combined_elements.append(sublist[0])
        else:
            combined_elements.append(sublist[1])

    output.append("MX(" + p + ") " + " ".join(map(str, combined_elements)))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))))

    return "\n".join(output)


##############For Z stabilizer measurement circuit################

def str_cat_ancilla_prep_state_z_meas_error_include(ancilla_index_list, p="{p}"):
    rx_indices = []
    for sublist in ancilla_index_list:
        if len(sublist) == 2:
            rx_indices.append(sublist[0])
        elif len(sublist) == 4:
            rx_indices.append(sublist[1])
    r_str = "R " + " ".join(map(str, rx_indices))

    # X_ERROR string
    x_error_str_1 = "X_ERROR(" + p + ") " + " ".join(map(str, rx_indices))

    # RX string
    all_ancilla = [num for sublist in ancilla_index_list for num in sublist]
    unused_ancilla = list(set(all_ancilla) - set(rx_indices))
    rx_str_unused = "RX " + " ".join(map(str, sorted(unused_ancilla)))

    # Z_ERROR string
    z_error_str = "Z_ERROR(" + p + ") " + " ".join(map(str, sorted(unused_ancilla)))

    # DEPOLARIZE1
    idel_str = "DEPOLARIZE1(" + p + ") " + " ".join(map(str, list(range(0, num_qubit))))

    return "\n".join([r_str, x_error_str_1, rx_str_unused, z_error_str, idel_str])

def str_cat_ancilla_prep_cnot_z_meas_error_include(
    ancilla_index_list, num_qubit, p="{p}"
):
    output = []

    len2_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 2 for num in sublist
    ]
    len2_elements = [len2_elements[i ^ 1] for i in range(len(len2_elements))]

    len4_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 4 for num in sublist
    ]
    extract_reverse = [extract_specific_indices_len4_elements(len4_elements)[i ^ 1] for i in range(len(extract_specific_indices_len4_elements(len4_elements)))]

    output.append("CX " + " ".join(map(str, len2_elements)))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, len2_elements)))
    output.append("CX " + " ".join(map(str, extract_reverse)))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, extract_reverse)))
    depolarize1_numbers = list(range(0, num_qubit))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, depolarize1_numbers)))
    output.append("CX " + " ".join(map(str, reorder_list_len4_elements_z_meas(len4_elements))))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, reorder_list_len4_elements_z_meas(len4_elements))))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, depolarize1_numbers)))

    return "\n".join(output)

def str_cat_cnot_z_meas_error_include(
    num_qubit, cnot_order, g_idx_list_revised, ancilla_index_list, p, trape_list,indices_8body_face
):
    output = ""

    #Counting all numbers in ancilla_index_listト
    num_total_ancilla = sum(len(sublist) for sublist in ancilla_index_list)

    for val in range(1, num_depth + 1):
        cx_str = "CX"
        used_indices = []

        for index, (cnot, revised) in enumerate(zip(cnot_order, g_idx_list_revised)):
            if val in cnot:
                indices_of_val = [i for i, v in enumerate(cnot) if v == val]
                for idx in indices_of_val:

                    if index in trape_list:
                        if idx in [0, 1]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][0]
                        else:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][1]

                    elif index in indices_8body_face:
                        if idx in [4,6]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][0]
                        elif idx in [5,7]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][1]
                        elif idx in [0,2]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][2]
                        elif idx in [1, 3]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][3]
                    else:
                        if idx in [0, 2]:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][0]
                        else:
                            cat_index = ancilla_index_list[
                                g_idx_list_revised.index(revised)
                            ][1]

                    used_indices.extend([cat_index, revised[idx]])

        used_indices_swapped = [used_indices[i ^ 1] for i in range(len(used_indices))]
        cx_str += " " + " ".join(map(str, used_indices_swapped))
        depolarize2 = (
            "\nDEPOLARIZE2(" + str(p) + ") " + " ".join(map(str, used_indices_swapped))
        )

        all_qubits = set(range(num_qubit))
        ancilla_qubits = set(range(num_qubit, num_qubit + num_total_ancilla))
        unused_qubits = list((all_qubits.union(ancilla_qubits)) - set(used_indices))

        depolarize1 = (
            "\nDEPOLARIZE1(" + str(p) + ") " + " ".join(map(str, sorted(unused_qubits)))
        )
        output += cx_str + depolarize2 + depolarize1 + "\nTICK\n"

    return output

def str_cat_ancilla_decoding_z_meas_error_include(
    ancilla_index_list, num_qubit, p="{p}"
):
    output = []
    len2_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 2 for num in sublist
    ]
    len2_elements = [len2_elements[i ^ 1] for i in range(len(len2_elements))]

    len4_elements = [
        num for sublist in ancilla_index_list if len(sublist) == 4 for num in sublist
    ]
    extract_reverse = [extract_specific_indices_len4_elements(len4_elements)[i ^ 1] for i in range(len(extract_specific_indices_len4_elements(len4_elements)))]

    output.append("CX " + " ".join(map(str, reorder_list_len4_elements_z_meas_after(len4_elements))))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, reorder_list_len4_elements_z_meas_after(len4_elements))))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))))
    output.append("CX " + " ".join(map(str, extract_reverse)))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, extract_reverse)))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))))
    output.append("CX " + " ".join(map(str, len2_elements)))
    output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, len2_elements)))

    return "\n".join(output)

def str_cat_ancilla_meas_z_meas_error_include(ancilla_index_list, num_qubit, p="{p}"):
    output = []

    len4_elements= [
        sublist[i] for sublist in ancilla_index_list if len(sublist) == 4
    for i in [0, 2, 3] ]

    len2_elements_second = [
        sublist[1] for sublist in ancilla_index_list if len(sublist) == 2
    ]
    output.append("MX(" + p + ") " + " ".join(map(str, len4_elements)))
    output.append("MX(" + p + ") " + " ".join(map(str, len2_elements_second)))

    combined_elements = []
    for sublist in ancilla_index_list:
        if len(sublist) == 2:
            combined_elements.append(sublist[0])
        else:
            combined_elements.append(sublist[1])

    output.append("M(" + p + ") " + " ".join(map(str, combined_elements)))
    output.append("DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))))

    return "\n".join(output)

str_data_meas_mx = f"MX " + " ".join([str(i) for i in range(num_qubit)])
str_data_meas_m = f"M " + " ".join([str(i) for i in range(num_qubit)])

#Construct the circuits
mpp_circuit = stim.Circuit(
    f"""
    {str_data_initialize_rx}
    {str_stabilizer_mpp_x(g_idx_list_revised)}
    {str_stabilizer_mpp_z(g_idx_list_revised)}
    {str_logical_mpp_x}
"""
)

stb_x_circuit = stim.Circuit(
    f"""
    {str_cat_ancilla_prep_state_x_meas_error_include(ancilla_index_list, p=f"{p}")}
    {str_cat_ancilla_prep_cnot_x_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
    {str_cat_cnot_x_meas_error_include(num_qubit, cnot_order, g_idx_list_revised, ancilla_index_list, p, trape_list,indices_8body_face)}
    {str_cat_ancilla_decoding_x_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
    {str_cat_ancilla_meas_x_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
"""
)

stb_z_circuit = stim.Circuit(
    f"""
    {str_cat_ancilla_prep_state_z_meas_error_include(ancilla_index_list, p=f"{p}")}
    {str_cat_ancilla_prep_cnot_z_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
    {str_cat_cnot_z_meas_error_include(num_qubit, cnot_order, g_idx_list_revised, ancilla_index_list, p, trape_list,indices_8body_face)}
    {str_cat_ancilla_decoding_z_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
    {str_cat_ancilla_meas_z_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
"""
)

cnt_lgcerror = 0
for _ in range(num_shots):
    flag_result_dround=[]
    flag_result_dround_x=[]
    simulator = stim.TableauSimulator()
    mpp_x_initial = [0 for _ in range(num_face)]
    mpp_z_initial = [0 for _ in range(num_face)]
    not_diff_s = [[0 for _ in range(num_face)] for _ in range(d)]
    simulator.do(mpp_circuit)
    for i in range(num_face):
        mpp_x_initial[i] = simulator.current_measurement_record()[
            -(num_face * 2 + 1) + i
        ]
    for i in range(num_face):
        mpp_z_initial[i] = simulator.current_measurement_record()[-(num_face + 1) + i]
    for i_d in range(d):
        simulator.do(stb_x_circuit)
        for n in range(num_face):
            not_diff_s[i_d][n] = (
                simulator.current_measurement_record()[-num_face + n] + mpp_x_initial[n]
            ) % 2

        # Flag values for 2-qubit flag gadget
        flag_meas_2qubit_cat = [simulator.current_measurement_record()[start_index_2qubit + i] for i in range(num_flag_2)]

        # Flag values for 4-qubit flag gadget
        flag_meas = []
        start_index_flag = -(num_face + num_flag_2 + 3*num_flag_4)
        for i in range(num_flag_4):
            sublist = [
                simulator.current_measurement_record()[start_index_flag + 3*i],
                simulator.current_measurement_record()[start_index_flag + 3*i + 1],
                simulator.current_measurement_record()[start_index_flag + 3*i + 2],
            ]
            flag_meas.append(sublist)

        flag_results_x = generate_flag_results_auto(ancilla_index_list, flag_meas, flag_meas_2qubit_cat)
        flag_results_x = convert_bool_to_int(flag_results_x)
        flag_result_dround_x.append(flag_results_x)

        # Deflagging procedure for 2-qubit flag gadget
        for idx, flag_value in zip(two_length_indices, flag_meas_2qubit_cat):
            if flag_value == 1:
                if idx in trape_list:
                    operation_index = g_idx_list_revised[idx][0], g_idx_list_revised[idx][1]
                else:
                    operation_index = g_idx_list_revised[idx][0], g_idx_list_revised[idx][2]
                command = f"X_ERROR(1) {' '.join(map(str, operation_index))}"
                simulator.do(stim.Circuit(command))

        # Deflagging procedure for 4-qubit flag gadget
        for idx, flag_value in zip(four_length_indices, flag_meas):
            if all(x == 1 for x in flag_value):
                operation_index = g_idx_list_revised[idx][4], g_idx_list_revised[idx][5],g_idx_list_revised[idx][6], g_idx_list_revised[idx][7]
                command = f"X_ERROR(1) {' '.join(map(str, operation_index))}"
                simulator.do(stim.Circuit(command))

        simulator.do(stb_z_circuit)
        # Flag values for 2-qubit flag gadget
        flag_meas_2qubit_cat = [simulator.current_measurement_record()[start_index_2qubit + i] for i in range(num_flag_2)]

        # Flag values for 4-qubit flag gadget
        flag_meas = []
        start_index_flag = -(num_face + num_flag_2 + 3*num_flag_4)
        for i in range(num_flag_4):
            sublist = [
                simulator.current_measurement_record()[start_index_flag + 3*i],
                simulator.current_measurement_record()[start_index_flag + 3*i + 1],
                simulator.current_measurement_record()[start_index_flag + 3*i + 2],
            ]
            flag_meas.append(sublist)
        flag_results = generate_flag_results_auto(ancilla_index_list, flag_meas, flag_meas_2qubit_cat)
        flag_results = convert_bool_to_int(flag_results)
        flag_result_dround.append(flag_results)

        # Deflagging procedure for 2-qubit flag gadget
        for idx, flag_value in zip(two_length_indices, flag_meas_2qubit_cat):
            if flag_value == 1:
                if idx in trape_list:
                    operation_index = g_idx_list_revised[idx][0], g_idx_list_revised[idx][1]
                else:
                    operation_index = g_idx_list_revised[idx][0], g_idx_list_revised[idx][2]
                command = f"Z_ERROR(1) {' '.join(map(str, operation_index))}"
                simulator.do(stim.Circuit(command))

        # Deflagging procedure for 4-qubit flag gadget
        for idx, flag_value in zip(four_length_indices, flag_meas):
            if all(x == 1 for x in flag_value):
                operation_index = g_idx_list_revised[idx][4], g_idx_list_revised[idx][5],g_idx_list_revised[idx][6], g_idx_list_revised[idx][7]
                command = f"Z_ERROR(1) {' '.join(map(str, operation_index))}"
                simulator.do(stim.Circuit(command))

    # Run the rest of the circuit.
    simulator.do(stim.Circuit(f"""{str_data_meas_mx}"""))
    plane_true = [
        simulator.current_measurement_record()[-num_qubit + i] for i in range(num_qubit)
    ]
    plane_true = [int(b) for b in plane_true]
    s = [not_diff_s[0]]

    for i in range(1, len(not_diff_s)):
        new_row = [
            (not_diff_s[i - 1][j] + not_diff_s[i][j]) % 2
            for j in range(len(not_diff_s[0]))
        ]
        s.append(new_row)

    #Integer programming decoder
    lp = cplex.Cplex()
    lp.set_problem_type(lp.problem_type.LP)
    lp.objective.set_sense(lp.objective.sense.minimize)
    lp.set_problem_name("distance_lp")
    lp.parameters.threads.set(1)
    lp_2 = cplex.Cplex()
    lp_2.set_problem_type(lp_2.problem_type.LP)
    lp_2.objective.set_sense(lp_2.objective.sense.minimize)
    lp_2.set_problem_name("distance_2lp")
    lp_2.parameters.threads.set(1)
    lp.set_log_stream(None)
    lp.set_results_stream(None)
    lp_2.set_log_stream(None)
    lp_2.set_results_stream(None)

    var = [
        "x" + str(i) + str(j) for j in range(1, d + 1) for i in range(1, num_qubit + 1)
    ] + ["r" + str(j) + str(i) for i in range(1, d + 1) for j in range(1, num_face + 1)]

    var_w = ["w" + str(i) for i in range(1, num_face * d + 1)]

    b=[]
    for t in range(d):
        if t==0:
            for data_qubit_index in range(num_qubit):
                p_adjust=data_store[str(p)][str(t)][str(data_qubit_index)]
                b.append(-np.log(p_adjust/(1-p_adjust)))

        else:
            for data_qubit_index in range(num_qubit):
                hoge = "".join([convert_to_string(flag_result_dround[t-1][i]) for i in find_indices(data_qubit_index, g_idx_list)])
                p_adjust=data_store[str(p)][str(1)][str(data_qubit_index)][hoge]
                if p_adjust==0:
                    b.append(10*5)
                elif p_adjust==1:
                    b.append(-10*5)
                else:
                    b.append(-np.log(p_adjust/(1-p_adjust)))

    for t in range(d):
        for synd_qubit_index in range(num_face):
            hoge = "".join([convert_to_string(flag_result_dround_x[t][synd_qubit_index])])
            p_adjus=data_store_synd[str(p)][str(time_setting(t))][str(synd_qubit_index)][hoge]
            if p_adjus==0:
                b.append(10*5)
            elif p_adjus==1:
                b.append(-10*5)
            else:
                b.append(-np.log(p_adjus/(1-p_adjus)))

    lp.variables.add(obj=b, names=var, types=["B"] * (num_qubit * d + num_face * d))
    lp.variables.add(names=var_w, types=["I"] * num_face * d)
    lin_exp_former = []
    cnt_w = 1
    for m in range(num_face):
        lin_exp_former.append(
            [
                ["x" + str(i) + "1" for i in g_idx_list[m]]
                + ["r" + str(m + 1) + str(1), "w" + str(cnt_w)],
                [1 for _ in range(len(g_idx_list[m]) + 1)] + [-2],
            ]
        )
        cnt_w += 1
        for j in range(2, d):
            lin_exp_former.append(
                [
                    ["x" + str(i) + str(j) for i in g_idx_list[m]]
                    + [
                        "r" + str(m + 1) + str(j),
                        "r" + str(m + 1) + str(j - 1),
                        "w" + str(cnt_w),
                    ],
                    [1 for _ in range(len(g_idx_list[m]) + 2)] + [-2],
                ]
            )
            cnt_w += 1
        lin_exp_former.append(
            [
                ["x" + str(i) + str(d) for i in g_idx_list[m]]
                + [
                    "r" + str(m + 1) + str(d - 1),
                    "r" + str(m + 1) + str(d),
                    "w" + str(cnt_w),
                ],
                [1 for _ in range(len(g_idx_list[m]) + 2)] + [-2],
            ]
        )
        cnt_w += 1
    # Syndrome constraint
    lp.linear_constraints.add(
        names=["C_" + str(i) for i in range(num_face * d)],
        lin_expr=lin_exp_former,
        senses=["E"] * num_face * d,
        rhs=[s[t][i] for i in range(num_face) for t in range(d)],
    )

    lp.solve()
    ans = lp.solution.get_values(var)
    for i in range(len(ans)):
        ans[i] = int(ans[i] + 0.5)

    plane_estimate = compute_plane_estimate(ans, d, num_qubit)
    plane_correction = [x ^ y for (x, y) in zip(plane_true, plane_estimate)]

    #Ideal error correction
    s_after = [
        (sum(plane_correction[j] for j in g_idx_list_revised[i]) % 2)
        for i in range(num_face)
    ]
    var = ["x" + str(i) for i in range(num_qubit)]
    var_w = ["w" + str(i) for i in range(num_face)]

    b = [1] * num_qubit

    lp_2.variables.add(obj=b, names=var, types=["B"] * num_qubit)
    lp_2.variables.add(names=var_w, types=["I"] * num_face)

    # Syndrome constraint
    lp_2.linear_constraints.add(
        names=["C" + str(i) for i in range(1, num_face + 1)],
        lin_expr=[
            [
                ["x" + str(idx) for idx in g_idx_list_revised[i]] + ["w" + str(i)],
                [1 for _ in g_idx_list_revised[i]] + [-2],
            ]
            for i in range(len(g_idx_list_revised))
        ],
        senses=["E"] * num_face,
        rhs=s_after,
    )

    lp_2.solve()
    ans = lp_2.solution.get_values()
    for i in range(len(ans)):
        ans[i] = int(ans[i] + 0.5)
    plane_correction_after = [
        x ^ y for (x, y) in zip(ans[:num_qubit], plane_correction)
    ]
    if sum(plane_correction_after) % 2 == 1:
        cnt_lgcerror += 1

print(cnt_lgcerror/num_shots)

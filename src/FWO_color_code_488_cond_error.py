import cplex
import stim
import functools
import operator
import json
import numpy as np
from itertools import product

num_shots = 1000000
p_values = np.arange(0.0001, 0.0041, 0.0001)

parameters_set = {
    "488": {
        5: {
            "num_qubit": 17,
            "num_face": 8,
            "num_depth": 3,
            "trape_list": [3, 6],
            "left_trape_list": [6],
            "indices_8body_face":[1],
            "g_idx_list": [[1, 2, 6, 7], [2, 3, 7, 8, 11, 12, 14, 15], [3, 4, 8, 9], [4, 5, 9, 13], [6, 7, 10, 11], [8, 9, 12, 13], [10, 11, 14, 16], [14, 15, 16, 17]],
            "g_idx_list_revised":[[0, 1, 5, 6], [1, 2, 6, 7, 10, 11, 13, 14], [2, 3, 7, 8], [3, 4, 8, 12], [5, 6, 9, 10], [7, 8, 11, 12], [9, 10, 13, 15], [13, 14, 15, 16]],
            "cnot_order":[[3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 3], [3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 2, 3], [3, 3, 1, 2]]
        },
        7: {
            "num_qubit": 31,
            "num_face": 15,
            "num_depth": 3,
            "trape_list": [0, 12, 13],
            "left_trape_list":[0, 13],
            "indices_8body_face":[2, 4, 10],
            "g_idx_list": [[1, 2, 8, 14], [2, 3, 8, 9], [3, 4, 9, 10, 15, 16, 20, 21], [4, 5, 10, 11], [5, 6, 11, 12, 17, 18, 22, 23], [6, 7, 12, 13], [8, 9, 14, 15], [10, 11, 16, 17], [12, 13, 18, 19], [20, 21, 24, 25], [16, 17, 21, 22, 25, 26, 28, 29], [22, 23, 26, 27], [18, 19, 23, 27], [24, 25, 28, 30], [28, 29, 30, 31]],
            "g_idx_list_revised":[[0, 1, 7, 13], [1, 2, 7, 8], [2, 3, 8, 9, 14, 15, 19, 20], [3, 4, 9, 10], [4, 5, 10, 11, 16, 17, 21, 22], [5, 6, 11, 12], [7, 8, 13, 14], [9, 10, 15, 16], [11, 12, 17, 18], [19, 20, 23, 24], [15, 16, 20, 21, 24, 25, 27, 28], [21, 22, 25, 26], [17, 18, 22, 26], [23, 24, 27, 29], [27, 28, 29, 30]],
            "cnot_order":[[2, 1, 2, 3], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [3, 3, 1, 2], [3, 3, 1, 2], [3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 3], [2, 1, 2, 3], [3, 3, 1, 2]]
        },
        9: {
            "num_qubit": 49,
            "num_face": 24,
            "num_depth": 3,
            "trape_list": [7, 12, 21, 22],
            "left_trape_list": [12, 22],
            "indices_8body_face":[1, 3, 5, 14, 16, 19],
            "g_idx_list": [[1, 2, 10, 11],[2, 3, 11, 12, 19, 20, 26, 27], [3, 4, 12, 13], [4, 5, 13, 14, 21, 22, 28, 29], [5, 6, 14, 15], [6, 7, 15, 16, 23, 24, 30, 31], [7, 8, 16, 17], [8, 9, 17, 25], [10, 11, 18, 19], [12, 13, 20, 21], [14, 15, 22, 23], [16, 17, 24, 25], [18, 19, 26, 32], [26, 27, 32, 33], [20, 21, 27, 28, 33, 34, 38, 39], [28, 29, 34, 35], [22, 23, 29, 30, 35, 36, 40, 41], [30, 31, 36, 37], [38, 39, 42, 43], [34, 35, 39, 40, 43, 44, 46, 47], [40, 41, 44, 45], [36, 37, 41, 45], [42, 43, 46, 48], [46, 47, 48, 49]],
            "g_idx_list_revised":[[0, 1, 9, 10], [1, 2, 10, 11, 18, 19, 25, 26], [2, 3, 11, 12], [3, 4, 12, 13, 20, 21, 27, 28], [4, 5, 13, 14], [5, 6, 14, 15, 22, 23, 29, 30], [6, 7, 15, 16], [7, 8, 16, 24], [9, 10, 17, 18], [11, 12, 19, 20], [13, 14, 21, 22], [15, 16, 23, 24], [17, 18, 25, 31], [25, 26, 31, 32], [19, 20, 26, 27, 32, 33, 37, 38], [27, 28, 33, 34], [21, 22, 28, 29, 34, 35, 39, 40], [29, 30, 35, 36], [37, 38, 41, 42], [33, 34, 38, 39, 42, 43, 45, 46], [39, 40, 43, 44], [35, 36, 40, 44], [41, 42, 45, 47], [45, 46, 47, 48]],
            "cnot_order":[[3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 3], [3, 3, 1, 2], [3, 3, 1, 2], [3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 2, 3], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 2, 3, 3, 1, 2], [3, 3, 1, 2], [2, 1, 1, 3], [2, 1, 2, 3], [3, 3, 1, 2]]
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

def calcu_synd_from_data(data_err, g_idx_list_revised):
    s_list = []
    for idx_list in g_idx_list_revised:
        s = sum(data_err[i] for i in idx_list) % 2
        s_list.append(s)

    return s_list

def convert_bool_to_int(item):
    if isinstance(item, bool):
        return int(item)
    elif isinstance(item, list):
        return [convert_bool_to_int(sub_item) for sub_item in item]
    else:
        return item

def split_list_into_chunks(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def reshape_list(a, b):
    if not isinstance(a, list):
        return b.pop(0)
    else:
        return [reshape_list(item, b) for item in a]

def transform_list(lst):
    if not lst:
        return []

    grouped = []
    sub_list = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i-1]:
            sub_list.append(lst[i])
        else:
            if len(sub_list) == 3:
                grouped.append(sub_list)
            else:
                grouped.extend(sub_list)
            sub_list = [lst[i]]

    if len(sub_list) == 3:
        grouped.append(sub_list)
    else:
        grouped.extend(sub_list)

    return grouped

def find_indices(i, g_idx_list):
    target = i + 1
    indices = []
    for idx, sublist in enumerate(g_idx_list):
        if target in sublist:
            indices.append(idx)
    return [index for index in indices]

def find_indices_triple_1(i, g_idx_list):
    target = i + 1
    indices = []
    for idx, sublist in enumerate(g_idx_list):
        if target in sublist:
            if idx in indices_8body_face:
                for _ in range(3):
                    indices.append(idx)
            else:
                indices.append(idx)
    return indices

def generate_bit_combinations(length):
    return [format(i, f"0{length}b") for i in range(2**length)]

def initialize_data_store(num_qubit, g_idx_list,p_values):
    data_store = {}
    for p in p_values:
        data_store[p] = {}
        for t in range(2):
            data_store[p][t] = {}

            for i in range(num_qubit):
                bit_length = len(find_indices_triple_1(i, g_idx_list))
                combinations = generate_bit_combinations(bit_length)
                if t == 0:
                    data_store[p][t][i] = 0
                else:
                    data_store[p][t][i] = {comb: 0 for comb in combinations}

    return data_store

#The number of times data errors occur for all sets of flag values
data_store = initialize_data_store(num_qubit, g_idx_list,p_values)
#The number of times sets of flag values are triggered
data_store_count = initialize_data_store(num_qubit, g_idx_list,p_values)

def initialize_synd_store(num_face,p_values):
    synd_store = {}
    for p in p_values:
        synd_store[p] = {}
        for t in range(2):
            synd_store[p][t] = {}

            for i in range(num_face):
                bit_length  = 3 if i in indices_8body_face else 1
                combinations = generate_bit_combinations(bit_length)
                synd_store[p][t][i] = {comb: 0 for comb in combinations}

    return synd_store

#The number of times measurement errors occur for all sets of flag values
synd_store = initialize_synd_store(num_face,p_values)
#The number of times sets of flag values are triggered
synd_store_count = initialize_synd_store(num_face,p_values)

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

for p in p_values:
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

        output.append("CX " + " ".join(map(str, extract_specific_indices_len4_elements(len4_elements))))
        output.append("DEPOLARIZE2(" + p + ") " + " ".join(map(str, extract_specific_indices_len4_elements(len4_elements))))


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

        return "\n".join(output)

    ##############For Z stabilizer measurement circuit################

    def str_cat_ancilla_prep_state_z_meas_error_include(ancilla_index_list, p="{p}"):
        # R string
        rx_indices = []
        for sublist in ancilla_index_list:
            if len(sublist) == 2:
                rx_indices.append(sublist[0])
            elif len(sublist) == 4:
                rx_indices.append(sublist[1])
        r_str = "RX " + " ".join(map(str, rx_indices))

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

        return "\n".join(["DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))),"DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))),"DEPOLARIZE1(" + p + ") " + " ".join(map(str, range(num_qubit))),r_str, x_error_str_1, rx_str_unused, z_error_str, idel_str])

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
    str_data_initialize_rx = "RX " + " ".join([str(i) for i in range(num_qubit)])
    all_samples = []
    flag_results_list = []

    #Construct the circuits
    mpp_circuit=stim.Circuit(f"""
        {str_data_initialize_rx}
    """)
    stb_x_circuit_time1 = stim.Circuit(
        f"""
        {str_cat_ancilla_prep_state_x_meas_error_include(ancilla_index_list, p=f"{p}")}
        {str_cat_ancilla_prep_cnot_x_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
        {str_cat_cnot_x_meas_error_include(num_qubit, cnot_order, g_idx_list_revised, ancilla_index_list, p, trape_list,indices_8body_face)}
        {str_cat_ancilla_decoding_x_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
        {str_cat_ancilla_meas_x_meas_error_include(ancilla_index_list, num_qubit, p=f"{p}")}
    """
    )
    stb_x_circuit_time2 = stim.Circuit(
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

    all_flag_result_x = []
    all_s=[]
    all_plane_true=[]
    for sampl in range(num_shots):
        flag_result_dround=[]
        simulator = stim.TableauSimulator()
        mpp_x_initial = [0 for _ in range(num_face)]
        mpp_z_initial = [0 for _ in range(num_face)]
        not_diff_s = [[0 for _ in range(num_face)] for _ in range(1)]
        simulator.do(mpp_circuit)

        for i_d in range(1):
            simulator.do(stb_x_circuit_time1)
            for n in range(num_face):
                not_diff_s[i_d][n] = (
                    simulator.current_measurement_record()[-num_face + n]
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

        # Run the rest of the circuit.
        simulator.do(stim.Circuit(f"""{str_data_meas_mx}"""))
        plane_true = [
            simulator.current_measurement_record()[-num_qubit + i] for i in range(num_qubit)
        ]
        plane_true = [int(b) for b in plane_true]
        all_plane_true.append(plane_true)

        s = [convert_bool_to_int(not_diff_s[0])]
        all_flag_result_x.append(flag_results_x)
        all_s.append(s)

    all_flag_result_time2 = []
    all_flag_result_x_time2 = []
    all_s_time2=[]
    all_plane_true_time2=[]
    for sampl in range(num_shots):
        flag_result_dround=[]
        simulator = stim.TableauSimulator()
        mpp_x_initial = [0 for _ in range(num_face)]
        mpp_z_initial = [0 for _ in range(num_face)]
        not_diff_s = [[0 for _ in range(num_face)] for _ in range(1)]
        simulator.do(mpp_circuit)

        for i_d_2 in range(1):
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
            flag_results_z = generate_flag_results_auto(ancilla_index_list, flag_meas, flag_meas_2qubit_cat)
            flag_results_z = convert_bool_to_int(flag_results_z)

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

            simulator.do(stb_x_circuit_time2)

            for n in range(num_face):
                not_diff_s[i_d_2][n] = (
                    simulator.current_measurement_record()[-num_face + n]
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
            flag_results_x_time2 = generate_flag_results_auto(ancilla_index_list, flag_meas, flag_meas_2qubit_cat)
            flag_results_x_time2 = convert_bool_to_int(flag_results_x_time2)

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

        # Run the rest of the circuit.
        simulator.do(stim.Circuit(f"""{str_data_meas_mx}"""))
        plane_true = [
            simulator.current_measurement_record()[-num_qubit + i] for i in range(num_qubit)
        ]
        plane_true = [int(b) for b in plane_true]
        all_plane_true_time2.append(plane_true)

        s = [convert_bool_to_int(not_diff_s[0])]
        all_flag_result_x_time2.append(flag_results_x_time2)
        all_flag_result_time2.append(flag_results_z)
        all_s_time2.append(s)

    for sampl in range(num_shots):
        data_ans = all_plane_true[sampl]
        syndrome = all_s[sampl][0]
        data_ans_time2 = all_plane_true_time2[sampl]
        syndrome_time2 = all_s_time2[sampl][0]
        for data_qubit_index_initial in range(num_qubit):
            data_store_count[p][0][data_qubit_index_initial]+=1
            data_store[p][0][data_qubit_index_initial]+=data_ans[data_qubit_index_initial]

        flag_results = all_flag_result_time2[sampl]
        flag_results_x = all_flag_result_x[sampl]
        flag_results_x_time2 = all_flag_result_x_time2[sampl]

        #Count the number of times data errors occur for all sets of flag values
        #Count the number of times sets of flag values are triggered
        for data_qubit_index in range(num_qubit):
            patterns = list(product([0, 1], repeat=len(find_indices_triple_1(data_qubit_index, g_idx_list))))
            hoge_transform=transform_list(find_indices_triple_1(data_qubit_index, g_idx_list))

            for pattern in patterns:
                string_value = "".join(str(x) for x in pattern)
                pattern = list(pattern)
                pattern=reshape_list(hoge_transform,pattern)
                if all(flag_results[find_indices(data_qubit_index, g_idx_list)[i]] == pattern[i] for i in range(len(find_indices(data_qubit_index, g_idx_list)))):
                    data_store_count[p][1][data_qubit_index][string_value]+=1
                    data_store[p][1][data_qubit_index][string_value]+=data_ans_time2[data_qubit_index]

        #Count the number of times measurement errors occur for all sets of flag values
        #Count the number of times sets of flag values are triggered
        for synd_qubit_index in range(num_face):
            repeat = 3 if synd_qubit_index in indices_8body_face else 1
            patterns = list(product([0, 1], repeat=repeat))

            for pattern in patterns:
                string_value = "".join(str(x) for x in pattern)
                pattern = list(pattern)
                pattern=pattern[0] if len(pattern)==1 else pattern
                if flag_results_x[synd_qubit_index] == pattern:
                    synd_store_count[p][0][synd_qubit_index][string_value]+=1
                    if syndrome[synd_qubit_index] != calcu_synd_from_data(data_ans,g_idx_list_revised)[synd_qubit_index]:
                        synd_store[p][0][synd_qubit_index][string_value]+=1

                if flag_results_x_time2[synd_qubit_index] == pattern:
                    synd_store_count[p][1][synd_qubit_index][string_value]+=1
                    if syndrome_time2[synd_qubit_index] != calcu_synd_from_data(data_ans_time2,g_idx_list_revised)[synd_qubit_index]:
                        synd_store[p][1][synd_qubit_index][string_value]+=1

print(data_store)
print(data_store_count)
print(synd_store)
print(synd_store_count)
#Calculate data_store/data_store_count and synd_store/synd_store_count after that to calculate conditional error probabilities.

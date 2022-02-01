from MySkmultiflow.data import DataStream
from MySkmultiflow.trees import HoeffdingTreeClassifier
from statistics import mean, stdev
from pympler import asizeof
import random
import math
import csv


def check_dataframe(target_index, dataframe_base, dataframe_1_1, dataframe_1_2, dataframe_1_4, dataframe_1_8):
    print("Checking")
    count = 0

    original_label = dataframe_base[target_index]
    label_1_1 = dataframe_1_1[target_index]
    label_1_2 = dataframe_1_2[target_index]
    label_1_4 = dataframe_1_4[target_index]
    label_1_8 = dataframe_1_8[target_index]

    for i in range(len(original_label)):
        if original_label[i] != label_1_1[i]:
            count = count + 1
    print(count)
    count = 0
    for i in range(len(original_label)):
        if original_label[i] != label_1_2[i]:
            count = count + 1
    print(count)
    count = 0
    for i in range(len(original_label)):
        if original_label[i] != label_1_4[i]:
            count = count + 1
    print(count)
    count = 0
    for i in range(len(original_label)):
        if original_label[i] != label_1_8[i]:
            count = count + 1
    print(count)


def generate_dataframe_reverse(dataframe, target_index, random_seed):
    label = list(dataframe[target_index])
    count_0 = label.count(0)
    count_1 = label.count(1)

    index_0 = []
    index_1 = []
    for i in range(len(label)):
        if label[i] == 0:
            index_0.append(i)
        else:
            index_1.append(i)

    change_count_0_1_1 = int(count_0 * (1 / 1))
    change_count_1_1_1 = int(count_1 * (1 / 1))
    change_count_0_1_2 = int(count_0 * (1 / 2))
    change_count_1_1_2 = int(count_1 * (1 / 2))
    change_count_0_1_4 = int(count_0 * (1 / 4))
    change_count_1_1_4 = int(count_1 * (1 / 4))
    change_count_0_1_8 = int(count_0 * (1 / 8))
    change_count_1_1_8 = int(count_1 * (1 / 8))

    dataframe_base = dataframe.copy()
    dataframe_1_1 = dataframe.copy()
    dataframe_1_2 = dataframe.copy()
    dataframe_1_4 = dataframe.copy()
    dataframe_1_8 = dataframe.copy()

    random.seed(random_seed)
    change_index_0_1_1 = random.sample(index_0, change_count_0_1_1)
    change_index_1_1_1 = random.sample(index_1, change_count_1_1_1)
    change_index_0_1_2 = random.sample(index_0, change_count_0_1_2)
    change_index_1_1_2 = random.sample(index_1, change_count_1_1_2)
    change_index_0_1_4 = random.sample(index_0, change_count_0_1_4)
    change_index_1_1_4 = random.sample(index_1, change_count_1_1_4)
    change_index_0_1_8 = random.sample(index_0, change_count_0_1_8)
    change_index_1_1_8 = random.sample(index_1, change_count_1_1_8)

    original_label = dataframe_1_1[target_index]
    label_1_1 = []
    label_1_2 = []
    label_1_4 = []
    label_1_8 = []

    for j in range(len(original_label)):
        if j in change_index_0_1_1 or j in change_index_1_1_1:
            if original_label[j] == 1:
                label_1_1.append(0)
            else:
                label_1_1.append(1)
        else:
            label_1_1.append(original_label[j])

        if j in change_index_0_1_2 or j in change_index_1_1_2:
            if original_label[j] == 1:
                label_1_2.append(0)
            else:
                label_1_2.append(1)
        else:
            label_1_2.append(original_label[j])

        if j in change_index_0_1_4 or j in change_index_1_1_4:
            if original_label[j] == 1:
                label_1_4.append(0)
            else:
                label_1_4.append(1)
        else:
            label_1_4.append(original_label[j])

        if j in change_index_0_1_8 or j in change_index_1_1_8:
            if original_label[j] == 1:
                label_1_8.append(0)
            else:
                label_1_8.append(1)
        else:
            label_1_8.append(original_label[j])

    dataframe_1_1[target_index] = label_1_1
    dataframe_1_2[target_index] = label_1_2
    dataframe_1_4[target_index] = label_1_4
    dataframe_1_8[target_index] = label_1_8

    #     Check_DataFrame(target_index, dataframe_base, dataframe_1_1, dataframe_1_2, dataframe_1_4, dataframe_1_8)

    dataframe_base = dataframe_base.astype(int)
    dataframe_1_1 = dataframe_1_1.astype(int)
    dataframe_1_2 = dataframe_1_2.astype(int)
    dataframe_1_4 = dataframe_1_4.astype(int)
    dataframe_1_8 = dataframe_1_8.astype(int)

    return dataframe_base, dataframe_1_1, dataframe_1_2, dataframe_1_4, dataframe_1_8


def generate_dataframe_feature(dataframe, target_index, feature_shift_list, random_seed, drift_type):
    print("generate dataframe for feature drift " + str(drift_type))
    label = dataframe[target_index]

    label = list(dataframe[target_index])
    count_0 = label.count(0)
    count_1 = label.count(1)

    index_0 = []
    index_1 = []
    for i in range(len(label)):
        if label[i] == 0:
            index_0.append(i)
        else:
            index_1.append(i)

    dataframes = []
    dataframe_base = dataframe.copy()
    dataframes.append(dataframe_base)

    change_ratios = [1, 2, 4, 8]

    for change_ratio in change_ratios:
        old_values = {}
        new_values = {}
        for feature in feature_shift_list:
            old_values[feature] = dataframe[feature]
            new_values[feature] = []

        change_count_0 = int(count_0 * (1 / change_ratio))
        change_count_1 = int(count_1 * (1 / change_ratio))

        dataframe_change = dataframe.copy()
        random.seed(random_seed)

        change_index_0 = random.sample(index_0, change_count_0)
        change_index_1 = random.sample(index_1, change_count_1)

        if drift_type == 0:
            change_index = change_index_0
        elif drift_type == 1:
            change_index = change_index_1
        elif drift_type == 2:
            change_index = change_index_0
            change_index.extend(change_index_1)

        Change_Count = 0
        for i in range(len(label)):
            if i in change_index:
                #                 print("Generate Feature Drift")
                Change_Count = Change_Count + 1

                for k in range(len(feature_shift_list)):
                    new_key = feature_shift_list[k]

                    if k == 0:
                        old_key = feature_shift_list[len(feature_shift_list) - 1]
                    else:
                        old_key = feature_shift_list[k - 1]
                    new_values[new_key].append(old_values[old_key][i])
            #                     print("new_" + str(new_key) + ".append(" + "old_" + str(old_key) +")")
            else:
                for k in range(len(feature_shift_list)):
                    new_key = feature_shift_list[k]
                    new_values[new_key].append(old_values[new_key][i])

        for k in range(len(feature_shift_list)):
            key = feature_shift_list[k]
            old = dataframe_change[key]
            new = new_values[key]
            c = 0
            for i in range(len(old)):
                if old[i] != new[i]:
                    c += 1
            # print("Feature " + str(key) + " # Shifted :" + str(c))

            dataframe_change[key] = new_values[key]

        dataframes.append(dataframe_change)
        # print(Change_Count)

    return dataframes[0], dataframes[1], dataframes[2], dataframes[3], dataframes[4]


def get_vectors(dataframe_base, dataframe_compared, target_index, cat_features):
    stream_base = DataStream(dataframe_base, target_idx=target_index, cat_features=cat_features)
    stream_compared = DataStream(dataframe_compared, target_idx=target_index, cat_features=cat_features)

    HT_Base = HoeffdingTreeClassifier(binary_split=True, no_preprune=True)
    HT_Base.partial_fit(stream_base.X, stream_base.y)
    constrain_dict = HT_Base.constrain_dict
    feature_list_base = HT_Base.feature_list.copy()
    HT_compared = HoeffdingTreeClassifier(binary_split=True, no_preprune=True, constrain_dict=constrain_dict, feature_list=feature_list_base)
    HT_compared.partial_fit(stream_compared.X, stream_compared.y)
    model_size_1 = asizeof.asizeof(HT_Base)
    model_size_2 = asizeof.asizeof(HT_compared)
    print("Model Size 1: " + str(model_size_1))
    print("Model Size 2: " + str(model_size_2))

    feature_list_overall = HT_compared.feature_list
    feature_dict = get_feature_encoded_dict(feature_list_overall)
    feature_dict_size = asizeof.asizeof(feature_dict)
    print("Feature Dictionary Size: " + str(feature_dict_size))
    
    rules_base = tree_to_rules(HT_Base)
    rules_compared = tree_to_rules(HT_compared)
    vectors_base = rules_to_vectors(rules_base, feature_dict)
    vectors_compared = rules_to_vectors(rules_compared, feature_dict)
    
    vector_size_1 = asizeof.asizeof(vectors_base)
    vector_size_2 = asizeof.asizeof(vectors_compared)
    print("Vector Size 1: " + str(vector_size_1))
    print("Vector Size 2: " + str(vector_size_2))
    
    total_size = model_size_1 + model_size_2 + feature_dict_size + vector_size_1 + vector_size_2
    print("Total Size: " + str(total_size))

    return vectors_base, vectors_compared

def get_feature_encoded_dict(feature_list):
    feature_encoded_dict = {}
    i = 0
    for feature in feature_list:
        f_0 = str(feature) + "_0"
        f_1 = str(feature) + "_1"
        feature_encoded_dict[f_0] = 2 * i
        feature_encoded_dict[f_1] = 2 * i + 1
        i = i + 1

    return feature_encoded_dict


def tree_to_rules(tree):
    model_rules = tree.get_model_rules()
    new_model_rules = []
    for model_rule in model_rules:
        predicate_set = model_rule.predicate_set
        new_model_rule = []
        for predicate in predicate_set:
            feature = predicate.att_idx
            operator = predicate.operator

            if operator == "<=":
                r = str(feature) + "_0"
            else:
                r = str(feature) + "_1"
            new_model_rule.append(r)
        class_distribution = ""
        keys = model_rule._observed_class_distribution.keys()
        if 0 in keys and 1 in keys:
            new_model_rule.append(str(int(model_rule._observed_class_distribution[0])) + "," + str(
                int(model_rule._observed_class_distribution[1])))
        elif 0 in keys:
            new_model_rule.append(str(int(model_rule._observed_class_distribution[0])) + ",0")
        elif 1 in keys:
            new_model_rule.append("0," + str(int(model_rule._observed_class_distribution[1])))
        new_model_rules.append(new_model_rule)
    return new_model_rules


def rules_to_vectors(rules, feature_encoded_dict):
    vectors = []
    for rule in rules:
        vector = []
        classification_rule = rule[:-1]
        classification_result = rule[-1]

        for key in feature_encoded_dict:
            if key in classification_rule:
                vector.append(1)
            else:
                vector.append(0)
        vector.append(classification_result)
        vectors.append(vector)
    return vectors


def group_vectors(vectors):
    vec_dict = {}
    for v in vectors:
        path = v[:-1]
        result = v[-1]

        key = str(result[0]) + "_" + str(result[1])
        if key not in vec_dict:
            paths = []
            paths.append(path)
            vec_dict[key] = paths
        else:
            paths = vec_dict[key]
            paths.append(path)
            vec_dict[key] = paths

    return vec_dict


def get_otu_expression(vec_A, vec_B):
    a = 0
    b = 0
    c = 0
    d = 0

    for i in range(len(vec_A)):
        if vec_A[i] == 1 and vec_B[i] == 1:
            a = a + 1
        if vec_A[i] == 0 and vec_B[i] == 1:
            b = b + 1
        if vec_A[i] == 1 and vec_B[i] == 0:
            c = c + 1
        if vec_A[i] == 0 and vec_B[i] == 0:
            d = d + 1

    return a, b, c, d

def write_to_file(filename, measurements_result, Measurements):
    output = []
    measurement_index = 0

    with open(filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Measurement', 'Base_to_1', 'Base_to_2', 'Base_to_4', 'Bast_to_8', 'Base_to_Base'])

        for Key in measurements_result:
            measurement_mean = mean(measurements_result[Key])
            measurement_stdev = stdev(measurements_result[Key])
            output.append((measurement_mean, measurement_stdev))
            if len(output) == 5:
                measurement = Measurements[measurement_index]
                o_0 = str(round(output[0][0], 4)) + " $\pm$ " + str(round(output[0][1], 4))
                o_1 = str(round(output[1][0], 4)) + " $\pm$ " + str(round(output[1][1], 4))
                o_2 = str(round(output[2][0], 4)) + " $\pm$ " + str(round(output[2][1], 4))
                o_3 = str(round(output[3][0], 4)) + " $\pm$ " + str(round(output[3][1], 4))
                o_4 = str(round(output[4][0], 4)) + " $\pm$ " + str(round(output[4][1], 4))

                writer.writerow([measurement, o_1, o_2, o_3, o_4, o_0])
                output = []
                measurement_index += 1


def Calculate_Measurement(v_a, v_b, measurement):
    a, b, c, d = get_otu_expression(v_a, v_b)
    n = a + b + c + d

    laplace_correction = 0.000000001

    chi_square = (n * math.pow((a * d - b * c), 2)) / ((a + b) * (a + c) * (c + d) * (b + d))
    rho = abs((a * d - b * c)) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    sigma = max(a, b) + max(c, d) + max(a, c) + max(b, d)
    sigma_prime = max(a + c, b + d) + max(a + b, c + d)
    
    r = -1

    if measurement == 1:
        r = a / (a + b + c)
    if measurement == 2:
        r = 2 * a / (2 * a + b + c)
    if measurement == 3:
        r = 2 * a / (2 * a + b + c)
    if measurement == 4:
        r = 3 * a / (3 * a + b + c)
    if measurement == 5:
        r = 2 * a / ((a + b) + (a + c))
    if measurement == 6:
        r = a / (a + 2 * b + 2 * c)
    if measurement == 7:
        r = (a + d) / (a + b + c + d)
    if measurement == 8:
        r = 2 * (a + d) / (2 * a + b + c + 2 * d)
    if measurement == 9:
        r = (a + d) / (a + 2 * (b + c) + d)
    if measurement == 10:
        r = (a + 0.5 * d) / (a + b + c + d)
    if measurement == 11:
        r = (a + d) / (a + 0.5 * (b + c) + d)
    if measurement == 12:
        r = a
    if measurement == 13:
        r = a + d
    if measurement == 14:
        r = a / (a + b + c + d)
    if measurement == 15:
        r = b + c
    if measurement == 16:
        r = math.sqrt(b + c)
    if measurement == 17:
        r = math.sqrt(math.pow((b + c), 2))
    if measurement == 18:
        r = b + c
    if measurement == 19:
        r = b + c
    if measurement == 20:
        r = (b + c) / (a + b + c + d)
    if measurement == 21:
        r = b + c
    if measurement == 22:
        r = b + c
    if measurement == 23:
        r = (b + c) / (4 * (a + b + c + d))
    if measurement == 24:
        r = math.pow((b + c), 2) / math.pow((a + b + c + d), 2)
    if measurement == 25:
        r = (n * (b + c) - math.pow((b - c), 2)) / math.pow((a + b + c + d), 2)
    if measurement == 26:
        r = (4 * b * c) / math.pow((a + b + c + d), 2)
    if measurement == 27:
        r = (b + c) / (2 * a + b + c)
    if measurement == 28:
        r = (b + c) / (2 * a + b + c)
    if measurement == 29:
        r = 2 * math.sqrt(1 - a / math.sqrt((a + b) * (a + c)))
    if measurement == 30:
        r = math.sqrt(2 * (1 - a / math.sqrt((a + b) * (a + c))))
    if measurement == 31:
        r = a / math.sqrt(math.pow((a + b) * (a + c), 2))
    if measurement == 32:
        if a == 0:
            a = laplace_correction
        r = math.log(a) - math.log(n) - math.log((a + b) / n) - math.log((a + c) / n)
    if measurement == 33:
        r = a / math.sqrt((a + b) * (a + c))
    if measurement == 34:
        r = (n * a) / ((a + b) * (a + c))
    if measurement == 35:
        r = (n * math.pow((a - 0.5), 2)) / ((a + b) * (a + c))
    if measurement == 36:
        r = (a * a) / ((a + b) * (a + c))
    if measurement == 37:
        r = (a * a - b * c) / ((a + b) * (a + c))
    if measurement == 38:
        r = a / math.sqrt((a + b) * (a + c))
    if measurement == 39:
        r = (math.pow(a, 2) - b * c) / ((a + b) * (a + c))
    if measurement == 40:
        r = (n * a - (a + b) * (a + c)) / (n * a + (a + b) * (a + c))
    if measurement == 41:
        r = ((a / 2) * (2 * a + b + c)) / ((a + b) * (a + c))
    if measurement == 42:
        r = (a / 2) * (1 / (a + b) + 1 / (a + c))
    if measurement == 43:
        r = (a / (a + b)) + (a / (a + c))
    if measurement == 44:
        r = (a * d - b * c) / (math.sqrt(n * (a + b) * (a + c)))
    if measurement == 45:
        r = a / min((a + b), (a + c))
    if measurement == 46:
        r = a / max((a + b), (a + c))
    if measurement == 47:
        r = a / math.sqrt((a + b) * (a + c)) - max((a + b), (a + c)) / 2
    if measurement == 48:
        r = (n * a - (a + b) * (a + c)) / (n * min((a + b), (a + c)) - (a + b) * (a + c))
    if measurement == 49:
        r = (a / (a + b) + a / (a + c) + d / (b + d) + d / (b + d)) / 4
    if measurement == 50:
        r = (a + d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    if measurement == 51:
        r = chi_square
    if measurement == 52:
        r = math.sqrt(chi_square / (n + chi_square))
    if measurement == 53:
        r = math.sqrt(rho / (n + rho))
    if measurement == 54:
        r = (a * d - b * c) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    if measurement == 55:
        r = math.cos((math.pi * math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c)))
    if measurement == 56:
        denominator = b + c
        if denominator == 0:
            denominator = laplace_correction
        r = (a + d) / denominator
    if measurement == 57:
        r = a * d / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    if measurement == 58:
        denominator = math.sqrt(abs(math.pow((a * d - b * c), 2) - (a + b) * (a + c) * (b + d) * (c + d)))
        if denominator == 0:
            denominator = laplace_correction
        r = (math.sqrt(2) * (a * d - b * c)) / denominator
    if measurement == 59:
        r = math.log(10) * n * math.pow((abs(a * c - b * c) - n / 2), 2) / ((a + b) * (a + c) * (b + d) * (c + d))
    if measurement == 60:
        r = a * d / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    if measurement == 61:
        r = (a * d - b * c) / (a * d + b * c)
    if measurement == 62:
        r = 2 * b * c / (a * d + b * c)
    if measurement == 63:
        r = (math.sqrt(a * d) - math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c))
    if measurement == 64:
        denominator = b + c
        if denominator == 0:
            denominator = laplace_correction
        r = a / denominator
    if measurement == 65:
        r = a / (a + b + a + c - a)
    if measurement == 66:
        r = (a * d - b * c) / math.pow((a + b + c + d), 2)
    if measurement == 67:
        r = ((a + d) - (b + c)) / (a + b + c + d)
    if measurement == 68:
        r = (4 * (a * d - b * c)) / (math.pow((a + d), 2) + math.pow((b + c), 2))
    if measurement == 69:
        r = (sigma - sigma_prime) / (2 * n - sigma_prime)
    if measurement == 70:
        r = (sigma - sigma_prime) / (2 * n)
    if measurement == 71:
        r = (math.sqrt(a * d) + a) / (math.sqrt(a * d) + a + b + c)
    if measurement == 72:
        r = (math.sqrt(a * d) + a - (b + c)) / (math.sqrt(a * d) + a + b + c)
    if measurement == 73:
        denominator = (a * b + 2 * b * c + c * d)
        if denominator == 0:
            denominator = laplace_correction
        r = (a * b + b * c) / denominator
    if measurement == 74:
        r = (math.pow(n, 2) * (n * a - (a + b) * (a + c))) / ((a + b) * (a + c) * (b + d) * (c + d))
    if measurement == 75:
        denominator = (c * (a + b))
        if denominator == 0:
            denominator = laplace_correction
        r = (a * (c + d)) / denominator
    if measurement == 76:
        denominator = (c * (a + b))
        if denominator == 0:
            denominator = laplace_correction
        r = abs((a * (c + d)) / denominator)
    return r


def Similarity_Between_Trees(vec_A, vec_B, measurement):
    group_vec_A = group_vectors(vec_A)
    group_vec_B = group_vectors(vec_B)

    vec_similarities = {}
    total_vecs = 0

    for key in group_vec_A:
        A_vectors = group_vec_A[key]
        if key in group_vec_B:
            B_vectors = group_vec_B[key]

            similarities = []

            for v_a in A_vectors:
                for v_b in B_vectors:
                    similarity = Calculate_Measurement(v_a, v_b, measurement)
                    similarities.append(similarity)

            weight = len(A_vectors) + len(B_vectors)
            similarity = max(similarities)
        else:
            weight = len(A_vectors)
            similarity = 0
        vec_similarities[key] = [similarity, weight]
        total_vecs += weight

    for key in group_vec_B:
        if key not in group_vec_A:
            B_vectors = group_vec_B[key]
            weight = len(B_vectors)
            similarity = 0
            vec_similarities[key] = [similarity, weight]
            total_vecs += weight

    final_similarity = 0
    for key in vec_similarities:
        final_similarity = final_similarity + vec_similarities[key][0] * (vec_similarities[key][1] / total_vecs)

    return final_similarity


def Distance_Between_Trees(vec_A, vec_B, measurement):
    group_vec_A = group_vectors(vec_A)
    group_vec_B = group_vectors(vec_B)

    vec_distances = {}
    total_vecs = 0

    for key in group_vec_A:
        A_vectors = group_vec_A[key]
        if key in group_vec_B:
            B_vectors = group_vec_B[key]

            distances = []

            for v_a in A_vectors:
                for v_b in B_vectors:
                    distance = Calculate_Measurement(v_a, v_b, measurement)
                    distances.append(distance)

            weight = len(A_vectors) + len(B_vectors)
            distance = min(distances)
        else:
            weight = len(A_vectors)
            distance = 1
        vec_distances[key] = [distance, weight]
        total_vecs += weight

    for key in group_vec_B:
        if key not in group_vec_A:
            B_vectors = group_vec_B[key]
            weight = len(B_vectors)
            distance = 1
            vec_distances[key] = [distance, weight]
            total_vecs += weight

    final_distance = 0
    for key in vec_distances:
        final_distance = final_distance + vec_distances[key][0] * (vec_distances[key][1] / total_vecs)

    return final_distance



def get_measurement_results(Measurement_Indication, Measurements, Measurement_Results, vec_base, vec_base_1, vec_base_2, vec_base_4, vec_base_8, vec_1, vec_2, vec_4, vec_8):
    if Measurement_Indication == "Similarity":
        print("Calculate Similarity")
        for measurement in Measurements:
            Similarity_base_to_base = Similarity_Between_Trees(vec_base, vec_base, measurement)
            Key = str(measurement) + "_base_to_base"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Similarity_base_to_base)

            Similarity_base_to_1 = Similarity_Between_Trees(vec_base_1, vec_1, measurement)
            Key = str(measurement) + "_base_to_1"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Similarity_base_to_1)

            Similarity_base_to_2 = Similarity_Between_Trees(vec_base_2, vec_2, measurement)
            Key = str(measurement) + "_base_to_2"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Similarity_base_to_2)

            Similarity_base_to_4 = Similarity_Between_Trees(vec_base_4, vec_4, measurement)
            Key = str(measurement) + "_base_to_4"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Similarity_base_to_4)

            Similarity_base_to_8 = Similarity_Between_Trees(vec_base_8, vec_8, measurement)
            Key = str(measurement) + "_base_to_8"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Similarity_base_to_8)

    elif Measurement_Indication == "Distance":
        print("Calculate Distance")
        for measurement in Measurements:
            Distance_base_to_base = Distance_Between_Trees(vec_base, vec_base, measurement)
            Key = str(measurement) + "_base_to_base"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Distance_base_to_base)

            Distance_base_to_1 = Distance_Between_Trees(vec_base_1, vec_1, measurement)
            Key = str(measurement) + "_base_to_1"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Distance_base_to_1)

            Distance_base_to_2 = Distance_Between_Trees(vec_base_2, vec_2, measurement)
            Key = str(measurement) + "_base_to_2"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Distance_base_to_2)

            Distance_base_to_4 = Distance_Between_Trees(vec_base_4, vec_4, measurement)
            Key = str(measurement) + "_base_to_4"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Distance_base_to_4)

            Distance_base_to_8 = Distance_Between_Trees(vec_base_8, vec_8, measurement)
            Key = str(measurement) + "_base_to_8"
            if Key not in Measurement_Results:
                Measurement_Results[Key] = []
            Measurement_Results[Key].append(Distance_base_to_8)

    return Measurement_Results

    # for measurement in Measurements:
    #     Distance_base_to_base = Distance_Between_Trees(vec_base, vec_base, measurement)
    #     Key = measurement + "_base_to_base"
    #     if Key not in Measurement_Results:
    #         Measurement_Results[Key] = []
    #     Measurement_Results[Key].append(Distance_base_to_base)
    #
    #     Distance_base_to_1 = Distance_Between_Trees(vec_base_1, vec_1, measurement)
    #     Key = measurement + "_base_to_1"
    #     if Key not in Measurement_Results:
    #         Measurement_Results[Key] = []
    #     Measurement_Results[Key].append(Distance_base_to_1)
    #
    #     Distance_base_to_2 = Distance_Between_Trees(vec_base_2, vec_2, measurement)
    #     Key = measurement + "_base_to_2"
    #     if Key not in Measurement_Results:
    #         Measurement_Results[Key] = []
    #     Measurement_Results[Key].append(Distance_base_to_2)
    #
    #     Distance_base_to_4 = Distance_Between_Trees(vec_base_4, vec_4, measurement)
    #     Key = measurement + "_base_to_4"
    #     if Key not in Measurement_Results:
    #         Measurement_Results[Key] = []
    #     Measurement_Results[Key].append(Distance_base_to_4)
    #
    #     Distance_base_to_8 = Distance_Between_Trees(vec_base_8, vec_8, measurement)
    #     Key = measurement + "_base_to_8"
    #     if Key not in Measurement_Results:
    #         Measurement_Results[Key] = []
    #     Measurement_Results[Key].append(Distance_base_to_8)

    # return Measurement_Results






# def Distance_Between_Trees(vec_A, vec_B, measurement):
#     group_vec_A = group_vectors(vec_A)
#     # print(group_vec_A)
#     # for k in group_vec_A:
#     #     print("Key: " + str(k) + " Vectors: " + str(group_vec_A[k]))
#     # print("-----")
#
#     group_vec_B = group_vectors(vec_B)
#     # print(group_vec_B)
#     # for k in group_vec_B:
#     #     print("Key: " + str(k) + " Vectors: " + str(group_vec_B[k]))
#
#     A_keys = list(group_vec_A.keys())
#     B_keys = list(group_vec_B.keys())
#
#     vec_distances = {}
#     total_paths = 0
#     for key in A_keys:
#         if key in B_keys:
#             print("Key: " + str(key))
#
#             A_paths = group_vec_A[key]
#             B_paths = group_vec_B[key]
#             distances = []
#
#             length_count = len(A_paths) + len(B_paths)
#             for p_a in A_paths:
#                 for p_b in B_paths:
#                     print("Path A: " + str(p_a))
#                     print("Path B: " + str(p_b))
#
#                     a, b, c, d = get_otu_expression(p_a, p_b)
#                     n = a + b + c + d
#                     chi = (n * (a * d - b * c) * (a * d - b * c)) / ((a + b) * (a + c) * (c + d) * (b + d))
#                     rho = (a * d - b * c) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
#                     sigma = max(a, b) + max(c, d) + max(a, c) + max(b, d)
#                     sigma_prime = max(a + c, b + d) + max(a + b, c + d)
#
#                     if measurement == "S_JACCARD":
#                         distance = a / (a + b + c)
#                     if measurement == "S_DICE":
#                         distance = (2 * a) / (2 * a + b + c)
#                     if measurement == "S_3W_JACCARD":
#                         distance = (3 * a) / (3 * a + b + c)
#                     if measurement == "S_SOKAL&SNEATH-I":
#                         distance = a / (a + 2 * b + 2 * c)
#                     if measurement == "S_SOKAL&MICHENER":
#                         distance = (a + d) / (a + b + c + d)
#                     if measurement == "S_SOKAL&SNEATH-II":
#                         distance = 2 * (a + d) / (2 * a + b + c + 2 * d)
#                     if measurement == "S_ROGET&TANIMOTO":
#                         distance = (a + d) / (a + 2 * (b + c) + d)
#                     if measurement == "S_FAITH":
#                         distance = (a + 0.5 * d) / (a + b + c + d)
#                     if measurement == "S_GOWER&LEGENDRE":
#                         distance = (a + d) / (a + 0.5 * (b + c) + d)
#                     if measurement == "S_INTERSECTION":
#                         distance = a
#                     if measurement == "S_INNERPRODUCT":
#                         distance = a + d
#                     if measurement == "S_RUSSELL&RAO":
#                         distance = a / (a + b + c + d)
#                     if measurement == "D_HAMMING":
#                         distance = b + c
#                     if measurement == "D_EUCLID":
#                         distance = math.sqrt(b + c)
#                     if measurement == "D_VARI":
#                         distance = (b + c) / (4 * (a + b + c + d))
#                     if measurement == "D_MEAN_MANHATTAN":
#                         distance = (b + c) / (a + b + c + d)
#                     if measurement == "D_SIZEDIFFERENCE":
#                         distance = ((b + c) * (b + c)) / ((a + b + c + d) * (a + b + c + d))
#                     if measurement == "D_SHAPEDIFFERENCE":
#                         distance = (n * (b + c) - (b - c) * (b - c)) / (a + b + c + d) * (a + b + c + d)
#                     if measurement == "D_PATTERNDIFFERENCE":
#                         distance = (4 * b * c) / ((a + b + c + d) * (a + b + c + d))
#                     if measurement == "D_LANCE&WILLIAMS":
#                         distance = (b + c) / (2 * a + b + c)
#                     if measurement == "D_HELLINGER":
#                         distance = 2 * math.sqrt(1 - a / math.sqrt((a + b) * (a + c)))
#                     if measurement == "D_CHORD":
#                         distance = math.sqrt(2 * (1 - a / math.sqrt((a + b) * (a + c))))
#                     if measurement == "S_COSINE":
#                         distance = a / math.sqrt((a + b) * (a + c))
#                     if measurement == "S_FORBESI":
#                         distance = (n * a) / ((a + b) * (a + c))
#                     if measurement == "S_FOSSUM":
#                         distance = (n * (a - 0.5) * (a - 0.5)) / ((a + b) * (a + c))
#                     if measurement == "S_SORGENFREI":
#                         distance = (a * a) / ((a + b) * (a + c))
#                     if measurement == "S_MOUNTFORD":
#                         distance = (a * a - b * c) / ((a + b) * (a + c))
#                     if measurement == "S_TARWID":
#                         distance = (n * a - (a + b) * (a + c)) / (n * a + (a + b) * (a + c))
#                     if measurement == "S_KULCZYNSKI-II":
#                         distance = ((a / 2) * (2 * a + b + c)) / ((a + b) * (a + c))
#                     if measurement == "S_DRIVER&KROEBER":
#                         distance = (a / 2) * (1 / (a + b) + 1 / (a + c))
#                     if measurement == "S_JOHNSON":
#                         distance = (a / (a + b)) + (a / (a + c))
#                     if measurement == "S_DENNIS":
#                         distance = (a * d - b * c) / (math.sqrt(n * (a + b) * (a + c)))
#                     if measurement == "S_SIMPSON":
#                         distance = a / min((a + b), (a + c))
#                     if measurement == "S_BRAUN&BANQUET":
#                         distance = a / max((a + b), (a + c))
#                     if measurement == "S_FAGER&McGOWAN":
#                         distance = a / math.sqrt((a + b) * (a + c)) - max((a + b), (a + c)) / 2
#                     if measurement == "S_FORBES-II":
#                         distance = (n * a - (a + b) * (a + c)) / (n * min((a + b), (a + c)) - (a + b) * (a + c))
#                     if measurement == "S_SOKAL&SNEATH-IV":
#                         distance = (a / (a + b) + a / (a + c) + d / (b + d) + d / (b + d)) / 4
#                     if measurement == "S_GOWER":
#                         distance = (a + d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
#                     if measurement == "S_PEARSON-I":
#                         distance = chi * chi
#                     if measurement == "S_PEARSON-II":
#                         distance = math.sqrt((chi * chi) / (n + chi * chi))
#                     if measurement == "S_PEARSON&HERON-I":
#                         distance = (a * d - b * c) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
#                     if measurement == "S_PEARSON&HERON-II":
#                         distance = math.cos(math.pi * math.sqrt(b * c) / (math.sqrt(a * d) + math.sqrt(b * c)))
#                     if measurement == "S_SOKAL&SNEATH-V":
#                         distance = a * d / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
#                     if measurement == "S_COLE":
#                         if (math.pow((abs(a * d - b * c) - n / 2), 2)) == 0:
#                             continue
#                         distance = math.log10(
#                             n * math.pow((abs(a * d - b * c) - n / 2), 2) / (a + b) * (a + c) * (b + d) * (c + d))
#                     if measurement == "S_YULEQ":
#                         distance = (a * d - b * c) / (a * d + b * c)
#                     if measurement == "D_YULEQ":
#                         distance = 2 * b * c / (a * d + b * c)
#                     if measurement == "S_YULEW":
#                         distance = (math.sqrt(a * d) - math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c))
#                     if measurement == "S_DISPERSON":
#                         distance = (a * d - b * c) / math.pow((a + b + c + d), 2)
#                     if measurement == "S_HAMANN":
#                         distance = ((a + d) - (b + c)) / (a + b + c + d)
#                     if measurement == "S_MICHAEL":
#                         distance = (4 * (a * d - b * c)) / (math.pow((a + d), 2) + math.pow((b + c), 2))
#                     if measurement == "S_GOODMAN&KRUSKAL":
#                         distance = (sigma - sigma_prime) / (2 * n - sigma_prime)
#                     if measurement == "S_ANDERBERG":
#                         distance = (sigma - sigma_prime) / (2 * n)
#                     if measurement == "S_BARONI-URBANI&BUSER-I":
#                         distance = (math.sqrt(a * d) + a) / (math.sqrt(a * d) + a + b + c)
#                     if measurement == "S_BARONI-URBANI&BUSER-II":
#                         distance = (math.sqrt(a * d) + a - (b + c)) / (math.sqrt(a * d) + a + b + c)
#                     if measurement == "S_EYRAUD":
#                         distance = (n * n * (n * a - (a + b) * (a + c))) / ((a + b) * (a + c) * (b + d) * (c + d))
#                     if measurement == "S_TARANTULA":
#                         distance = (a * (c + d)) / (c * (a + b))
#
#                     distances.append(distance)
#                     # print("Similarity: " + str(distance))
#                     print("Distance: " + str(distance))
#             # else:
#             vec_distances[key] = [min(distances), length_count]
#             # vec_distances[key] = [max(distances), length_count]
#             # print("Max Similarity: " + str(max(distances)))
#             print("Min Distance: " + str(min(distances)))
#             total_paths = total_paths + length_count
#
#         else:
#             A_paths = group_vec_A[key]
#             vec_distances[key] = [1, len(A_paths)]
#             total_paths = total_paths + len(A_paths)
#     for key in B_keys:
#         if key not in A_keys:
#             B_paths = group_vec_B[key]
#             vec_distances[key] = [1, len(B_paths)]
#             total_paths = total_paths + len(B_paths)
#
#     distance = 0
#     for key in vec_distances:
#         print(str(vec_distances[key][0]) + "* " + str(vec_distances[key][1]))
#         distance = distance + vec_distances[key][0] * (vec_distances[key][1] / total_paths)
#     return distance
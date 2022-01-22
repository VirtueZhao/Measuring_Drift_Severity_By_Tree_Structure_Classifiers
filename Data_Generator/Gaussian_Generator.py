#%%%
import random
import numpy as np
from Library.lib import write_to_file
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#%%%
def generate_gaussian_data(label, class_0_mean, class_0_cov, class_1_mean, class_1_cov):
    if label == 0:
        data = np.random.multivariate_normal(class_0_mean, class_0_cov, 1).T
    else:
        data = np.random.multivariate_normal(class_1_mean, class_1_cov, 1).T
    data = [data[0][0], data[1][0], label]
    return data


def generate_gaussian_2d_stream(class_0_means, class_0_cov, class_1_means, class_1_cov, instance_per_concept, width, file_name):
    drift_position = [instance_per_concept, instance_per_concept * 2]
    instance_limit = instance_per_concept * 3
    stream = []

    for i in range(instance_limit):
        current_concept = int(i / instance_per_concept)
        label = random.randint(0, 1)
        if current_concept == 0:
            # print("Generating Case_Study_Results", i + 1, "From Concept", current_concept)
            data = generate_gaussian_data(label, class_0_means[current_concept], class_0_cov, class_1_means[current_concept], class_1_cov)
        else:
            drift_p = drift_position[current_concept - 1]
            if drift_p <= i < drift_p + width:
                prob_threshold = (i + 1 - drift_p) / width
                prob = random.uniform(0, 1)
                # print("Generating Case_Study_Results", i + 1, "From Concept", current_concept - 1, " to Concept", current_concept,
                      # "with prob:", prob_threshold)
                # print("Probability: ", prob)
                if prob <= prob_threshold:
                    # print("Generating Case_Study_Results From Concept", current_concept)
                    data = generate_gaussian_data(label, class_0_means[current_concept], class_0_cov, class_1_means[current_concept], class_1_cov)
                else:
                    # print("Generating Case_Study_Results From Concept", current_concept-1)
                    data = generate_gaussian_data(label, class_0_means[current_concept-1], class_0_cov, class_1_means[current_concept-1], class_1_cov)
                # data = []
            else:
                # print("Generating Case_Study_Results", i + 1, "From Concept", current_concept)
                data = generate_gaussian_data(label, class_0_means[current_concept], class_0_cov, class_1_means[current_concept], class_1_cov)

        stream.append(data)

    drift_point = [instance_per_concept, instance_per_concept * 2]

    sample_pass = 0
    sample_num = 10

    HT_WithoutDD = HoeffdingTreeClassifier()
    HT_WithDD = HoeffdingTreeClassifier()
    true_labels = []
    pred_withoutDD = []
    pred_withDD = []
    accuracy_withoutDD = []
    accuracy_withDD = []

    for data in stream:
        x = np.array([data[0:-1]])
        y = np.array([data[-1]])

        true_labels.append(y[0])
        pred_withoutDD.append(HT_WithoutDD.predict(x)[0])
        pred_withDD.append(HT_WithDD.predict(x)[0])

        HT_WithoutDD.partial_fit(x, y)
        HT_WithDD.partial_fit(x, y)

        if (sample_pass + 1) % sample_num == 0:
            accuracy_withoutDD.append(accuracy_score(true_labels, pred_withoutDD))
            accuracy_withDD.append(accuracy_score(true_labels, pred_withDD))
        if sample_pass in drift_point:
            HT_WithDD = HoeffdingTreeClassifier()
            HT_WithDD.partial_fit(x, y)

        sample_pass += 1

    plt.plot(accuracy_withoutDD, label=['Without DD'])
    plt.plot(accuracy_withDD, label=['With DD'])
    plt.legend()
    plt.show()
    print(np.count_nonzero(true_labels))
    write_to_file(file_name, stream)
#%%%
def main():
    target_0_cov = [[1, 0], [0, 2]]
    target_1_cov = [[1, 0], [0, 2]]
    target_0_means = [[2, 3], [3, 5], [4, 7]]
    target_1_means = [[4, 5], [7, 9], [10, 13]]

    source_0_cov = [[1, 0], [0, 2]]
    source_1_cov = [[1, 0], [0, 2]]
    source_0_means = [[3, 4], [7, 9], [10, 13]]
    source_1_means = [[5, 6], [11, 13], [16, 19]]

    # instance_per_concept = 5000
    # width = 1
    # file_name = "Gaussian_Large_Abrupt_Target.csv"
    # generate_gaussian_2d_stream(target_0_means, target_0_cov, target_1_means, target_1_cov, instance_per_concept, width, file_name)
    # width = instance_per_concept / 10
    # file_name = "Gaussian_Large_Gradual_Target.csv"
    # generate_gaussian_2d_stream(target_0_means, target_0_cov, target_1_means, target_1_cov, instance_per_concept, width, file_name)
    # instance_per_concept = 1000
    # width = 1
    # file_name = "Gaussian_Small_Abrupt_Target.csv"
    # generate_gaussian_2d_stream(target_0_means, target_0_cov, target_1_means, target_1_cov, instance_per_concept, width, file_name)
    # width = instance_per_concept / 10
    # file_name = "Gaussian_Small_Gradual_Target.csv"
    # generate_gaussian_2d_stream(target_0_means, target_0_cov, target_1_means, target_1_cov, instance_per_concept, width, file_name)

    # instance_per_concept = 5000
    # width = 1
    # file_name = "Gaussian_Large_Abrupt_Source.csv"
    # generate_gaussian_2d_stream(source_0_means,source_0_cov,source_1_means, source_1_cov, instance_per_concept, width, file_name)
    # width = instance_per_concept / 10
    # file_name = "Gaussian_Large_Gradual_Source.csv"
    # generate_gaussian_2d_stream(source_0_means,source_0_cov,source_1_means, source_1_cov, instance_per_concept, width, file_name)
    instance_per_concept = 1000
    # width = 1
    # file_name = "Gaussian_Small_Abrupt_Source.csv"
    # generate_gaussian_2d_stream(source_0_means,source_0_cov,source_1_means, source_1_cov, instance_per_concept, width, file_name)
    width = instance_per_concept / 10
    file_name = "Gaussian_Small_Gradual_Source.csv"
    generate_gaussian_2d_stream(source_0_means,source_0_cov,source_1_means, source_1_cov, instance_per_concept, width, file_name)


if __name__ == "__main__":
    main()
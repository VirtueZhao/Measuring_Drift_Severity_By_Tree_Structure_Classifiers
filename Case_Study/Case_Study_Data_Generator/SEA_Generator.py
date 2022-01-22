import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skmultiflow.data import SEAGenerator
from skmultiflow.data import ConceptDriftStream
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from sklearn.metrics import accuracy_score
#%%%
def generate_data(concept_size, drift_type):

    if concept_size == 'Large':
        example_num = 5000
    else:
        example_num = 1000
    if drift_type == 'Abrupt':
        drift_width = 1
    else:
        drift_width = example_num * 0.1
    print("Generating SEA - " + concept_size +'(' + str(example_num) + ')' + "-" + drift_type + '(' + str(drift_width) + ')')
    data_generator = ConceptDriftStream(
        stream=SEAGenerator(classification_function=1, random_state=1, balance_classes=True, noise_percentage=0),
        drift_stream=ConceptDriftStream(
            stream=SEAGenerator(classification_function=3, random_state=3, balance_classes=True, noise_percentage=0),
            drift_stream=SEAGenerator(classification_function=2, random_state=2, balance_classes=True, noise_percentage=0),
            random_state=2, position=example_num, width=drift_width
        ),
    random_state=0, position=example_num, width=drift_width)

    stream = data_generator.next_sample(example_num * 3)
    drift_point = [example_num, example_num * 2]

    stream_x_all = stream[0]
    stream_y_all = stream[1]
    sample_pass = 0
    sample_num = 5

    HT_WithoutDD = HoeffdingAdaptiveTreeClassifier(random_state=42)
    HT_WithDD = HoeffdingAdaptiveTreeClassifier(random_state=42)
    HT_TDM = HoeffdingAdaptiveTreeClassifier(random_state=42)

    true_labels = []
    pred_withoutDD = []
    pred_withDD = []
    pred_TDM = []
    accuracy_withoutDD = []
    accuracy_withDD = []
    accuracy_TDM = []

    for i in range(len(stream_x_all)):
        data = np.hstack([stream_x_all[i], stream_y_all[i]])
        x = np.array([data[0:-1]])
        y = np.array([data[-1]])

        true_labels.append(y[0])
        pred_withoutDD.append(HT_WithoutDD.predict(x)[0])
        pred_withDD.append(HT_WithDD.predict(x)[0])
        pred_TDM.append(HT_TDM.predict(x)[0])

        HT_WithoutDD.partial_fit(x, y)
        HT_WithDD.partial_fit(x, y)
        HT_TDM.partial_fit(x, y)

        if (sample_pass + 1) % sample_num == 0:
            accuracy_withoutDD.append(accuracy_score(true_labels, pred_withoutDD))
            accuracy_withDD.append(accuracy_score(true_labels, pred_withDD))
            accuracy_TDM.append(accuracy_score(true_labels, pred_TDM))
        if sample_pass in drift_point:
            HT_WithDD = HoeffdingAdaptiveTreeClassifier(random_state=42)
            HT_WithDD.partial_fit(x, y)
            if sample_pass == drift_point[1]:
                HT_TDM = HoeffdingAdaptiveTreeClassifier(random_state=42)
                HT_TDM.partial_fit(x, y)

        sample_pass += 1

    sns.set()
    plt.plot(accuracy_withoutDD[20:], label='Without DD')
    plt.plot(accuracy_withDD[20:], label='With DD')
    plt.plot(accuracy_TDM[20:], label='TDM')
    plt.legend()
    plt.show()

    DD_file = "Case_Study/Case_Study_Results/SEA_" + concept_size + "_" + drift_type + "_DD.csv"
    TDM_file = "Case_Study/Case_Study_Results/SEA_" + concept_size + "_" + drift_type + "_TDM.csv"
    np.savetxt(DD_file, np.array(accuracy_withDD))
    np.savetxt(TDM_file, np.array(accuracy_TDM))



concept_sizes = ['Large', 'Small']
drift_types = ['Abrupt', 'Gradual']

for concept_size in concept_sizes:
    for drift_type in drift_types:
        generate_data(concept_size, drift_type)
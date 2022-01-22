import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import ConceptDriftStream
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from sklearn.metrics import accuracy_score
#%%%
columns = ['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan', 'class']

target_large_abrupt_generator = ConceptDriftStream(
    stream=AGRAWALGenerator(classification_function=1, random_state=1, balance_classes=True, perturbation=0),
    drift_stream=(ConceptDriftStream(
        stream=AGRAWALGenerator(classification_function=3, random_state=2, balance_classes=True, perturbation=0),
        drift_stream=AGRAWALGenerator(classification_function=4, random_state=3, balance_classes=True, perturbation=0),
        random_state=2, position=5000, width=1
    )),
    random_state=0, position=5000, width=1)

stream = target_large_abrupt_generator.next_sample(15000)
file_name = 'Case_Study_Results/AGRAWAL_Large_Abrupt_Target.csv'


drift_point = [5000, 10000]
stream_x_all = stream[0]
stream_y_all = stream[1]
sample_pass = 0
sample_num = 10

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

output_stream = []
for i in range(len(stream_x_all)):
    data = np.hstack([stream_x_all[i], stream_y_all[i]])
    output_stream.append(data)
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
plt.plot(accuracy_withoutDD[30:], label=['Without DD'])
plt.plot(accuracy_withDD[30:], label=['With DD'])
plt.plot(accuracy_TDM[30:], label='TDM')
plt.legend()
plt.show()
print(np.count_nonzero(stream_y_all))
#%%%
np.savetxt("Case_Study_Results/AGRAWAL_DD.csv", np.array(accuracy_withDD))
np.savetxt("Case_Study_Results/AGRAWAL_TDM.csv", np.array(accuracy_TDM))
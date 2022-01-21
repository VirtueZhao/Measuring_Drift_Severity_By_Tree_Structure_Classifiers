import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import ConceptDriftStream
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import accuracy_score
#%%%
# target_large_abrupt_generator = ConceptDriftStream(
#     stream=HyperplaneGenerator(random_state=10, n_features=20, n_drift_features=10, noise_percentage=0, sigma_percentage=0),
#     drift_stream=(ConceptDriftStream(
#         stream=HyperplaneGenerator(random_state=12, n_features=20, n_drift_features=12, noise_percentage=0, sigma_percentage=0),
#         drift_stream=HyperplaneGenerator(random_state=14, n_features=20, n_drift_features=14, noise_percentage=0, sigma_percentage=0),
#         random_state=2, position=5000, width=1
#     )),
#     random_state=0, position=5000, width=1)

# target_large_gradual_generator = ConceptDriftStream(
#     stream=HyperplaneGenerator(random_state=10, n_features=20, n_drift_features=10, noise_percentage=0, sigma_percentage=0),
#     drift_stream=(ConceptDriftStream(
#         stream=HyperplaneGenerator(random_state=12, n_features=20, n_drift_features=12, noise_percentage=0, sigma_percentage=0),
#         drift_stream=HyperplaneGenerator(random_state=14, n_features=20, n_drift_features=14, noise_percentage=0, sigma_percentage=0),
#         random_state=2, position=5000, width=500
#     )),
#     random_state=0, position=5000, width=500)

# target_small_abrupt_generator = ConceptDriftStream(
#     stream=HyperplaneGenerator(random_state=10, n_features=20, n_drift_features=10, noise_percentage=0, sigma_percentage=0),
#     drift_stream=(ConceptDriftStream(
#         stream=HyperplaneGenerator(random_state=12, n_features=20, n_drift_features=12, noise_percentage=0, sigma_percentage=0),
#         drift_stream=HyperplaneGenerator(random_state=14, n_features=20, n_drift_features=14, noise_percentage=0, sigma_percentage=0),
#         random_state=2, position=1000, width=1
#     )),
#     random_state=0, position=1000, width=1)

# target_small_gradual_generator = ConceptDriftStream(
#     stream=HyperplaneGenerator(random_state=10, n_features=20, n_drift_features=10, noise_percentage=0, sigma_percentage=0),
#     drift_stream=(ConceptDriftStream(
#         stream=HyperplaneGenerator(random_state=12, n_features=20, n_drift_features=12, noise_percentage=0, sigma_percentage=0),
#         drift_stream=HyperplaneGenerator(random_state=14, n_features=20, n_drift_features=14, noise_percentage=0, sigma_percentage=0),
#         random_state=2, position=1000, width=100
#     )),
#     random_state=0, position=1000, width=100)
#%%%
# source_large_abrupt_generator = ConceptDriftStream(
#     stream=HyperplaneGenerator(random_state=16, n_features=20, n_drift_features=16, noise_percentage=0, sigma_percentage=0),
#     drift_stream=(ConceptDriftStream(
#         stream=HyperplaneGenerator(random_state=17, n_features=20, n_drift_features=17, noise_percentage=0, sigma_percentage=0),
#         drift_stream=HyperplaneGenerator(random_state=18, n_features=20, n_drift_features=18, noise_percentage=0, sigma_percentage=0),
#         random_state=2, position=5000, width=1
#     )),
#     random_state=0, position=5000, width=1)

# source_large_gradual_generator = ConceptDriftStream(
#     stream=HyperplaneGenerator(random_state=16, n_features=20, n_drift_features=16, noise_percentage=0, sigma_percentage=0),
#     drift_stream=(ConceptDriftStream(
#         stream=HyperplaneGenerator(random_state=17, n_features=20, n_drift_features=17, noise_percentage=0, sigma_percentage=0),
#         drift_stream=HyperplaneGenerator(random_state=18, n_features=20, n_drift_features=18, noise_percentage=0, sigma_percentage=0),
#         random_state=2, position=5000, width=500
#     )),
#     random_state=0, position=5000, width=500)

# source_small_abrupt_generator = ConceptDriftStream(
#     stream=HyperplaneGenerator(random_state=16, n_features=20, n_drift_features=16, noise_percentage=0, sigma_percentage=0),
#     drift_stream=(ConceptDriftStream(
#         stream=HyperplaneGenerator(random_state=17, n_features=20, n_drift_features=17, noise_percentage=0, sigma_percentage=0),
#         drift_stream=HyperplaneGenerator(random_state=18, n_features=20, n_drift_features=18, noise_percentage=0, sigma_percentage=0),
#         random_state=2, position=1000, width=1
#     )),
#     random_state=0, position=1000, width=1)

source_small_gradual_generator = ConceptDriftStream(
    stream=HyperplaneGenerator(random_state=16, n_features=20, n_drift_features=16, noise_percentage=0, sigma_percentage=0),
    drift_stream=(ConceptDriftStream(
        stream=HyperplaneGenerator(random_state=17, n_features=20, n_drift_features=17, noise_percentage=0, sigma_percentage=0),
        drift_stream=HyperplaneGenerator(random_state=18, n_features=20, n_drift_features=18, noise_percentage=0, sigma_percentage=0),
        random_state=2, position=1000, width=100
    )),
    random_state=0, position=1000, width=100)

# stream = target_large_abrupt_generator.next_sample(15000)
# file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Large_Abrupt_Target.csv'
# stream = target_large_gradual_generator.next_sample(15000)
# file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Large_Gradual_Target.csv'
# stream = target_small_abrupt_generator.next_sample(3000)
# file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Small_Abrupt_Target.csv'
# stream = target_small_gradual_generator.next_sample(3000)
# file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Small_Gradual_Target.csv'
# stream = source_large_abrupt_generator.next_sample(15000)
# file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Large_Abrupt_Source.csv'
# stream = source_large_gradual_generator.next_sample(15000)
# file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Large_Gradual_Source.csv'
# stream = source_small_abrupt_generator.next_sample(3000)
# file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Small_Abrupt_Source.csv'
stream = source_small_gradual_generator.next_sample(3000)
file_name = 'Dissimilar_Datasets/Synthetic/Hyperplane_Small_Gradual_Source.csv'

# drift_point = [5000, 10000]
drift_point = [1000, 2000]
stream_x_all = stream[0]
stream_y_all = stream[1]
sample_pass = 0
sample_num = 10

HT_WithoutDD = HoeffdingTreeClassifier()
HT_WithDD = HoeffdingTreeClassifier()
true_labels = []
pred_withoutDD = []
pred_withDD = []
accuracy_withoutDD = []
accuracy_withDD = []

output_stream = []
for i in range(len(stream_x_all)):
    data = np.hstack([stream_x_all[i], stream_y_all[i]])
    output_stream.append(data)
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

df = pd.DataFrame(output_stream)
df.to_csv(file_name, index=False)

plt.plot(accuracy_withoutDD, label=['Without DD'])
plt.plot(accuracy_withDD, label=['With DD'])
plt.legend()
plt.show()
print(np.count_nonzero(stream_y_all))
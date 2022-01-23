import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%%
def plot_result(generator, concept_size, drift_type, current_fig, count):
    DD_result = pd.read_csv("Case_Study/Case_Study_Results/" + generator + "_" + concept_size + "_" + drift_type +"_DD.csv", header=None)[0].to_numpy()
    TDM_result = pd.read_csv("Case_Study/Case_Study_Results/" + generator + "_" + concept_size + "_" + drift_type +"_TDM.csv", header=None)[0].to_numpy()

    current_fig.plot(DD_result[100:], label='Retrain')
    current_fig.plot(TDM_result[100:], label='TDM')

    for label in (current_fig.get_xticklabels() + current_fig.get_yticklabels()):
        label.set_fontsize(12)

    current_fig.set_title(generator + "_" + concept_size + "_" + drift_type, fontsize=14)

    if count in [9, 10, 11, 12]:
        current_fig.set_xlabel('Number of Instances', fontsize=16)
    if count in [1, 5, 9]:
        current_fig.set_ylabel('Accuracy', fontsize=16)
    current_fig.legend()


sns.set()
generators = ['AGRAWAL', 'RandomRBF', 'SEA']
concept_sizes = ['Large', 'Small']
drift_types = ['Abrupt', 'Gradual']

fig = plt.figure(figsize=(24, 4 * len(generators)))
count = 1
for generator in generators:
    for concept_size in concept_sizes:
        for drift_type in drift_types:
            current_fig = fig.add_subplot(len(generators), 4, count)
            plot_result(generator, concept_size, drift_type, current_fig, count)
            count += 1

fig.show()
fig.savefig('Case_Study/TDM_Case_Study.pdf', bbox_inches='tight')
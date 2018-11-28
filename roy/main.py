import os
import pandas as pd
import networkx as nx
import numpy as np
from graph_measures.features_infra.graph_features import GraphFeatures
from graph_measures.loggers import PrintLogger
from explore_exploit import ExploreExploit
from explore_exploit import StandardML
from graph_measures import feature_meta
import matplotlib.pyplot as plt

CHOSEN_FEATURES = feature_meta.NODE_FEATURES
Data_Sets = ['signaling_pathways', 'subelj_cora',  'moreno_blogs']
# Data_Sets = ['hard_challenge']

epsilons = [0, 0.01, 0.05]
labels = range(2)
Results = {x: {} for x in Data_Sets}
for x in Data_Sets:
    for y in labels:
        Results[x][y] = {}

data_to_eps_to_steps = {(x + "_" + str(i)): {y: {} for y in epsilons} for i in labels for x in Data_Sets}


class LoadData:
    def __init__(self, data_name):
        self._data_dir = os.path.join("data_sets", data_name)
        self._name = data_name
        df1 = pd.read_csv(os.path.join(self._data_dir, "graph_edges.txt"))
        self._gnx = nx.from_pandas_edgelist(df1, "n1", "n2", create_using=nx.DiGraph())
        self._logger = PrintLogger("MyLogger")
        self._features_mx = None
        self._tags = {}
        print(str(self._name)+" was loaded")

    def build_features(self):
        gnx_ftr = GraphFeatures(self._gnx, CHOSEN_FEATURES, dir_path=os.path.join(self._data_dir, "features"),
                                logger=self._logger)
        gnx_ftr.build(should_dump=True)  # build ALL_FEATURES
        self._features_mx = gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix)
        print(self._features_mx.shape)

    def build_tags(self):
            self._tags = []
            with open(os.path.join(self._data_dir, "tags.txt"), "r") as f:
                # for node, line in enumerate(f):
                    # tag = str(line.rstrip("\n"))
                    # tag = tag[1:tag.find('/', 1)]
                for i, line in enumerate(f):
                    node, tag = line.split()
                    # if node == str(i+1):
                    if node.isdigit():
                        node = int(node)
                    if tag.isdigit():
                        tag = int(tag)
                    self._gnx.node[node]['label'] = tag
                    # if not(self._gnx.has_node(node)):
                    #     self._gnx.add_node(node)
                for node in sorted(self._gnx):
                    self._tags.append(self._gnx.node[node]['label'])


class Learning:
    def __init__(self, features_mx, tags, data_name):
        self._mx = features_mx
        self._tags = tags
        self._name = data_name

    def run_eps_greedy(self, dist_calc_type="one_class", learn_type="ML", recall=0.7, epochs=5):
        # finds recall achieved per percent of revealed data
        print("finding recall using greedy epsilon algorithm")

        for label in labels:
            print(label, dist_calc_type, learn_type)
            exploration = ExploreExploit(self._mx, self._tags, label, recall)
            for eps in epsilons:
                mean_steps_per_recall = {round(x, 2): 0 for x in np.arange(0, recall + 0.01, 0.05)}
                # when updated - update also steps_per_recall at
                print("epsilon =", eps)
                time_tag_dict = {}
                for i in range(1, epochs + 1):
                    steps_per_recall, tags = exploration.run_recall(dist_calc_type, learn_type, batch_size=10, eps=eps)
                    mean_steps_per_recall = {k: mean_steps_per_recall[k] + steps_per_recall[k]
                                             for k in mean_steps_per_recall}
                    time_tag_dict[i] = tags
                time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
                time_tag_df.to_csv(str(label) + "_" + dist_calc_type + "_output.csv")

                mean_steps_per_recall = {k: mean_steps_per_recall[k] / epochs
                                         for k in mean_steps_per_recall}
                print("the mean num of steps using eps = " + str(eps) + " is: " + str(mean_steps_per_recall[recall]))
                data_to_eps_to_steps[self._name+"_"+str(label)][eps] = mean_steps_per_recall[recall]
                Results[self._name][label]["eps_greedy_"+str(eps)] = mean_steps_per_recall

    def run_stand_recall(self, dist_calc_type="one_class", learn_type="ML", recall=0.7, epochs=3):
        # finds recall achieved per percent of revealed data
        print("finding recall using standard ml")
        my_range = np.linspace(0, recall, 15)
        my_range = [round(i, 2) for i in my_range]
        certainties = 4
        for label in labels:
            learning = StandardML(self._mx, self._tags, label)
            print(label, dist_calc_type, learn_type)
            temp_rand_results = {x: 0 for x in my_range}
            for certainty in np.linspace(0.15, 0.5, certainties):     # when updated - update also Results[rand]
                temp_recall = []
                temp_steps = []
                # temp_stand_results = {x: [0, 0] for x in my_range}
                certainty = round(certainty, 2)
                for i in range(1, epochs + 1):
                    # Results[self._name][label]["standML" + str(certainty) + "_" + str(i)] = {}
                    for p in my_range:
                        stand_recall, stand_steps, rand_recall, rand_steps = learning.recall_per_data(p, certainty)
                        temp_rand_results[p] += rand_recall
                        temp_recall.append(stand_recall)
                        temp_steps.append(stand_steps)
                        # temp_stand_results[p][0] += stand_recall
                        # temp_stand_results[p][1] += stand_steps
                p = np.polyfit(temp_steps, temp_recall, 3)
                f = np.poly1d(p)

                Results[self._name][label]["standML_"+str(certainty)] = {max(f(x), 0): x for x in my_range}
            Results[self._name][label]["rand"] = {temp_rand_results[x]/(epochs*certainties): x
                                                  for x in temp_rand_results}

    def run_standard_ml(self, epochs=3):
        sum_acc_ml = 0
        sum_acc_dl = 0
        for i in range(epochs):
            learning = StandardML(self._mx, self._tags)
            acc_ml, acc_dl = learning.run_acc()
            print("accuracy of standard ml using random forest is: " + str(acc_ml)
                  + " for deep is: " + str(acc_dl))
            sum_acc_ml += acc_ml
            sum_acc_dl += acc_dl
        Results[self._name]['acc_ml'] = sum_acc_ml / epochs
        Results[self._name]['acc_dl'] = sum_acc_dl / epochs
        print("mean accuracy using random forest is: " + str(Results[self._name]['acc_ml'])
              + " for deep is: " + str(Results[self._name]['acc_dl']))

    # def run_al(self, learn_type="ML"):
    #     dist_calc_type = "one_class"
    #     print("active using learning type: " + str(learn_type))
    #     for label in ['moreno']:
    #         for eps in [0.05]:
    #             mean_steps = 0
    #             print(label, dist_calc_type)
    #             time_tag_dict = {}
    #             for i in range(1, 11):
    #                 exploration = ExploreExploit(self._tags[label], self._features_mx, 0.7, eps)
    #                 tags, n1 = exploration.run_opposite_active(dist_calc_type, learn_type)
    #                 time_tag_dict[i] = tags
    #                 Results['Active'] += n1
    #             time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
    #             time_tag_df.to_csv(label + "_active_" + dist_calc_type + "_output.csv")
    #             Results['Active'] /= 10


def my_plot():
    for data_set in Results:
        for label in Results[data_set]:
            for i in Results[data_set][label]:
                xs = [Results[data_set][label][i][x] for x in Results[data_set][label][i]]
                ys = [x for x in Results[data_set][label][i]]
                plt.plot(xs, ys, label=str(i))
            plt.title(str(data_set)+"_"+str(label)+"_recall per time")
            plt.legend()
            plt.xlabel('data_revealed')
            plt.ylabel('recall')
            fig = plt.gcf()
            fig.show()
            fig.savefig(str(data_set)+"__"+str(label)+'_recall_per_time.png')


if __name__ == '__main__':
    for data in Data_Sets:
        dl = LoadData(data)
        dl.build_features()
        dl.build_tags()
        learn = Learning(dl._features_mx, dl._tags, dl._name)
        learn.run_eps_greedy(learn_type="ML", recall=0.7, epochs=10)
        learn.run_stand_recall(learn_type="ML", recall=0.7, epochs=10)
        # learn.run_standard_ml(5)
        # dl.run_eps_greedy("DL")
        # dl.run_al("ML")
        # # dl.run_al("DL")
        # dl.run_standard_ml()
    # df = pd.DataFrame(data_to_eps_to_steps)
    # print(df)
    #
    # my_plot()
    print("done")

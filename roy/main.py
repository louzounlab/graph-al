import os
import pandas as pd
import networkx as nx
import numpy as np
from graph_measures.features_infra.graph_features import GraphFeatures
from graph_measures.loggers import PrintLogger
from explore_exploit import ExploreExploit
from explore_exploit import ExploreExploitF1
from explore_exploit import StandardML
from explore_exploit import grid_learn_xgb, grid_learn_nn
from graph_measures import feature_meta
import matplotlib.pyplot as plt
from graph_measures.feature_meta import FeatureMeta
from graph_measures.features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator

import warnings
warnings.filterwarnings("ignore")

CHOSEN_FEATURES = feature_meta.NODE_FEATURES

# Data_Sets = ['moreno_blogs', 'signaling_pathways', 'subelj_cora']
Data_Sets = ['cora2', 'citeseer2', 'pubmed2']

epsilons = [0, 0.01, 0.03, 0.05, 0.075, 0.1]
labels = range(2)
Results = {x: {} for x in Data_Sets}
for x in Data_Sets:
    for y in labels:
        Results[x][y] = {}

data_to_eps_to_steps = {(x + "_" + str(i)): {y: {} for y in epsilons} for i in labels for x in Data_Sets}


def add_neighbor_feature():
    neighbor_features = {
        "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
        "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
    }
    CHOSEN_FEATURES.update(neighbor_features)
    return None


class LoadData:
    def __init__(self, data_name):
        self._data_dir = os.path.join("data_sets", data_name)
        self._name = data_name
        df1 = pd.read_csv(os.path.join(self._data_dir, "graph_edges.txt"))
        self._gnx = nx.from_pandas_edgelist(df1, "n1", "n2", create_using=nx.DiGraph())
        self._logger = PrintLogger("MyLogger")
        self._features_mx = None
        self._tags = {}
        self.gnx_ftr = None
        print(str(self._name)+" was loaded")

    def build_features(self):
        self.gnx_ftr = GraphFeatures(self._gnx, CHOSEN_FEATURES, dir_path=os.path.join(self._data_dir, "features"),
                                     logger=self._logger)
        self.gnx_ftr.build(should_dump=True)  # build ALL_FEATURES
        self._features_mx = self.gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix)
        for i, node in enumerate(self._gnx):
            self._gnx.node[node]['features'] = self._features_mx[i]
        print(self._features_mx.shape)

    def build_tags(self):
        self._gnx.graph["node_labels"] = []
        self._tags = []
        for node in self._gnx.node:
            typ = type(node)
            break
        with open(os.path.join(self._data_dir, "tags.txt"), "r") as f:
            # for node, line in enumerate(f):
                # tag = str(line.rstrip("\n"))
                # tag = tag[1:tag.find('/', 1)]
            for i, line in enumerate(f):
                node, tag = line.split()
                node = typ(node)
                if tag.isdigit():
                    tag = int(tag)
                self._gnx.node[node]['label'] = tag
                if tag not in self._gnx.graph["node_labels"]:
                    self._gnx.graph["node_labels"].append(tag)
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

    def run_eps_greedy_f1(self, learn_type="ML", budget=20, epochs=5):
        print("finding f1 using greedy epsilon algorithm")

        exploration = ExploreExploitF1(self._mx, self._tags, budget=budget)
        for eps in epsilons:
            print("epsilon =", eps)
            mean_macrof1 = 0
            mean_microf1 = 0
            for i in range(1, epochs + 1):
                macrof1, microf1 = exploration.run(eps=eps)
                mean_macrof1 += macrof1
                mean_microf1 += microf1
            mean_macrof1 /= epochs
            mean_microf1 /= epochs

            print("the mean macroF1 is: " + str(mean_macrof1) + "the mean microF1 is: " + str(mean_microf1))

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
        sum_acc_rf = 0
        sum_acc_dl = 0
        sum_acc_xgb = 0
        for i in range(epochs):
            learning = StandardML(self._mx, self._tags)
            acc_xgb, acc_rf, acc_dl = learning.run_acc()    # (print_train=True)
            print("accuracy is:  XGB - " + str(acc_xgb) +
                  "   RF - " + str(acc_rf) + "   NN - " + str(acc_dl))
            sum_acc_rf += acc_rf
            sum_acc_dl += acc_dl
            sum_acc_xgb += acc_xgb
        mean_acc_rf = sum_acc_rf / epochs
        mean_acc_dl = sum_acc_dl / epochs
        mean_acc_xgb = sum_acc_xgb / epochs
        print("mean accuracy is: XGB - " + str(mean_acc_xgb) + "   RF - "
              + str(mean_acc_rf) + "   NN - " + str(mean_acc_dl))

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
            fig.savefig(str(data_set)+"_"+str(label)+'_recall_per_time.png')


if __name__ == '__main__':
    # add_neighbor_feature()
    for data in Data_Sets:
        dl = LoadData(data)
        dl.build_tags()
        dl.build_features()
        # learn = Learning(dl._features_mx, dl._tags, dl._name)
        # learn.run_eps_greedy(learn_type="ML", recall=0.7, epochs=3)
        # learn.run_stand_recall(learn_type="ML", recall=0.7, epochs=4)
        # learn.run_standard_ml(epochs=2)
        # learn.run_eps_greedy_f1(budget=20, epochs=7)
        grid_learn_xgb(dl._name, dl._features_mx, dl._tags, epochs=20)
        grid_learn_nn(dl._name, dl._features_mx, dl._tags, epochs=20)

    # df = pd.DataFrame(data_to_eps_to_steps)
    # print(df)
    # print(df.to_string())

    # my_plot()
    print("done")

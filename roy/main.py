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


class LoadData:
    def __init__(self):
        self._data_dir = os.path.join("data_sets", "moreno_blogs")
        # self._data_dir = "signaling_pathways"
        self._label = "moreno"
        df1 = pd.read_csv(os.path.join(self._data_dir, "graph_edges.txt"))
        # df1 = pd.read_csv(os.path.join(self._data_dir, "signaling_pathways_2004.txt"))
        self._gnx = nx.from_pandas_edgelist(df1, "n1", "n2", create_using=nx.DiGraph())
        # self._gnx = nx.DiGraph()  # should be a subclass of Graph
        # self._gnx.add_edges_from([(0, 1), (0, 2), (1, 3), (3, 2)])
        self._logger = PrintLogger("MyLogger")
        self._features_mx = None
        self._tags = {}

    def build_features(self):
        gnx_ftr = GraphFeatures(self._gnx, CHOSEN_FEATURES, dir_path=os.path.join(self._data_dir, "features"),
                                logger=self._logger)
        gnx_ftr.build(should_dump=True)  # build ALL_FEATURES
        self._features_mx = gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix)
        print(self._features_mx.shape)

    def build_tags(self):
        for label in ['moreno']:
            # ['Adapter', 'Ligand', 'Vesicles', 'Ribosomes', 'Membrane']
            self._tags[label] = []
            with open(os.path.join(self._data_dir, "tags_" + label + ".txt"), "r") as f:
                for node, line in enumerate(f):
                    # node, tag = line.split()
                    # self._gnx.node[int(node)][label] = int(tag)
                    tag = str(line.rstrip("\n"))
                    self._gnx.node[node+1][label] = tag
            for node in sorted(self._gnx):
                self._tags[label].append(self._gnx.node[node][label])

    def run_eps_greedy(self, learn_type="ML"):
        dist_calc_type = "one_class"
        label_to_eps_to_steps = {x: {y: 0 for y in [0, 0.01, 0.05]} for x in ['moreno']}
        # ['Adapter', 'Ligand', 'Vesicles', 'Ribosomes', 'Membrane']
        print("using learning type: " + str(learn_type))
        s1 = "greedy_" + learn_type + "_0_40"
        s2 = "greedy_" + learn_type
        for label in ['moreno']:
            for eps in [0.01]:
                mean_steps = 0
                print(label, dist_calc_type)
                time_tag_dict = {}
                for i in range(1, 11):
                    exploration = ExploreExploit(self._tags[label], self._features_mx, 0.7, eps)
                    num_steps, tags, n1, n2 = exploration.run(dist_calc_type, learn_type)
                    print(" an recall of 70% was achieved in " + str(num_steps) + " steps")
                    mean_steps += num_steps
                    time_tag_dict[i] = tags
                    Results[s1] += n1
                    Results[s2] += n2
                time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
                time_tag_df.to_csv(label + "_" + dist_calc_type + "_output.csv")
                print("the mean num of steps is: " + str(mean_steps / 10))
                label_to_eps_to_steps[label][eps] = mean_steps / 10
                Results[s1] /= 10
                Results[s2] /= 10
            df = pd.DataFrame(label_to_eps_to_steps)
            print(df)

    def run_al(self, learn_type="ML"):
        dist_calc_type = "one_class"
        print("active using learning type: " + str(learn_type))
        for label in ['moreno']:
            for eps in [0.05]:
                mean_steps = 0
                print(label, dist_calc_type)
                time_tag_dict = {}
                for i in range(1, 11):
                    exploration = ExploreExploit(self._tags[label], self._features_mx, 0.7, eps)
                    tags, n1 = exploration.run_opposite_active(dist_calc_type, learn_type)
                    time_tag_dict[i] = tags
                    Results['Active'] += n1
                time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
                time_tag_df.to_csv(label + "_active_" + dist_calc_type + "_output.csv")
                Results['Active'] /= 10

    def run_standard_ml(self):
        label = 'moreno'
        sum_acc_ml = 0
        sum_acc_dl = 0
        for i in range(10):
            learning = StandardML(self._tags[label], self._features_mx)
            acc_ml, acc_dl = learning.run()
            print("accuracy of standard ml using random forest is: " + str(acc_ml)
                  + " for deep is: " + str(acc_dl))

            sum_acc_ml += acc_ml
            sum_acc_dl += acc_dl
        Results['standard_ML'] = sum_acc_ml / 10
        Results['standard_DL'] = sum_acc_dl / 10
        print("mean accuracy using random forest is: " + str(Results['standard_ML'])
              + " for deep is: " + str(Results['standard_DL']))


def my_plot():
    x = ['greedy_ML_0_40', 'greedy_ML', 'greedy_DL_0_40', 'greedy_DL', 'Active', 'standard_ML', 'standard_DL']
    xs = ['ML_0_40', 'Greed_ML', 'DL_0_40', 'Greed_DL', 'Active' 'stand_ML', 'stand_DL']
    ys = [Results[i] for i in x]
    plt.bar(xs, ys)
    plt.savefig('test6.png')
    plt.show()


Results = {x: 0 for x in ['greedy_ML_0_40', 'greedy_ML', 'greedy_DL_0_40', 'greedy_DL', 'Active',
                          'standard_ML', 'standard_DL']}

if __name__ == '__main__':
    dl = LoadData()
    dl.build_features()
    dl.build_tags()
    dl.run_eps_greedy("ML")
    # dl.run_eps_greedy("DL")
    dl.run_al("ML")
    # dl.run_al("DL")
    dl.run_standard_ml()
    my_plot()
    print("done")

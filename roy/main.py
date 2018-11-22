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
labels = ['cora']
# labels = ['Adapter', 'Ligand', 'Vesicles', 'Ribosomes', 'Membrane']
Results = {}


class LoadData:
    def __init__(self):
        self._data_dir = os.path.join("data_sets", "subelj_cora")
        self._label = "cora"
        df1 = pd.read_csv(os.path.join(self._data_dir, "graph_edges.txt"))
        self._gnx = nx.from_pandas_edgelist(df1, "n1", "n2", create_using=nx.DiGraph())
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
        for label in labels:
            self._tags[label] = []
            with open(os.path.join(self._data_dir, "tags_" + label + ".txt"), "r") as f:
                for node, line in enumerate(f):
                    # for line in f:
                    # node, tag = line.split()
                    tag = str(line.rstrip("\n"))
                    tag = tag[1:tag.find('/', 1)]
                    # if not(self._gnx.has_node(node)):
                    #     self._gnx.add_node(node)
                    # self._gnx.node[node][label] = tag
                    self._gnx.node[node+1][label] = tag
            for node in sorted(self._gnx):
                self._tags[label].append(self._gnx.node[node][label])


class Learning:
    def __init__(self, features_mx, tags):
        self._mx = features_mx
        self._tags = tags

    def run_eps_greedy(self, dist_calc_type="one_class", learn_type="ML", recall=0.7, epochs=5):
        # finds recall achieved per percent of revealed data
        epsilons = [0, 0.01, 0.05]
        label_to_eps_to_steps = {x: {y: {} for y in epsilons} for x in labels}

        for label in labels:
            for eps in epsilons:
                mean_steps_per_recall = {round(x, 2): 0 for x in np.arange(0, recall + 0.05, 0.05)}
                print(label, dist_calc_type, learn_type)
                time_tag_dict = {}
                for i in range(1, epochs + 1):
                    exploration = ExploreExploit(self._mx, self._tags[label], recall, eps)
                    steps_per_recall, tags = exploration.run_recall(dist_calc_type, learn_type)
                    print(" an recall of " + str(recall*100) + "% was achieved in "
                          + str(steps_per_recall[recall]) + " steps")
                    mean_steps_per_recall = {k: mean_steps_per_recall[k] + steps_per_recall[k]
                                             for k in mean_steps_per_recall}
                    time_tag_dict[i] = tags
                time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
                time_tag_df.to_csv(label + "_" + dist_calc_type + "_output.csv")

                mean_steps_per_recall = {k: mean_steps_per_recall[k] / epochs
                                         for k in mean_steps_per_recall}
                print("the mean num of steps is: " + str(mean_steps_per_recall[recall]))
                label_to_eps_to_steps[label][eps] = mean_steps_per_recall[recall]
                Results[str(label)+'_'+str(eps)] = mean_steps_per_recall
        df = pd.DataFrame(label_to_eps_to_steps)
        print(df)

    def run_stand_recall(self, dist_calc_type="one_class", learn_type="ML", epochs=5):
        # finds recall achieved per percent of revealed data
        print("finding recall using standard ml")
        for label in labels:
            learning = StandardML(self._mx, self._tags[label])
            print(label, dist_calc_type, learn_type)
            for i in range(1, epochs + 1):
                for certainty in np.arange(0.2, 0.6, 0.1):
                    certainty = round(certainty, 2)
                    Results[str(label) + "_standML" + str(certainty) + "_" + str(i)] = {0: 0}
                    for p in np.arange(0.05, 1.05, 0.05):
                        p = round(p, 2)
                        recall, steps = learning.recall_per_data(p, certainty)
                        Results[str(label) + "_standML" + str(certainty) + "_" + str(i)][recall] = steps

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
        # TODO: adding an option for multi labels
        label = labels[0]
        sum_acc_ml = 0
        sum_acc_dl = 0
        for i in range(10):
            learning = StandardML(self._features_mx, self._tags[label])
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
    for i in Results:
        xs = [Results[i][x] for x in Results[i]]
        ys = [x for x in Results[i]]
        plt.plot(xs, ys, label=str(i))
    plt.title("recall per time")
    plt.legend()
    plt.xlabel('data_revealed')
    plt.ylabel('recall')
    fig = plt.gcf()
    fig.show()
    fig.savefig('recall_per_time.png')


if __name__ == '__main__':
    dl = LoadData()
    dl.build_features()
    dl.build_tags()
    learn = Learning(dl._features_mx, dl._tags)
    # learn.run_eps_greedy(learn_type="ML", recall=0.7, epochs=4)
    learn.run_stand_recall(learn_type="ML", epochs=2)
    # dl.run_eps_greedy("DL")
    # dl.run_al("ML")
    # # dl.run_al("DL")
    # dl.run_standard_ml()
    my_plot()
    print("done")

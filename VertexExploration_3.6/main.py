import os
import pandas as pd
import networkx as nx
import numpy as np
from graph_features import GraphFeatures
from loggers import PrintLogger
# from features_infra.feature_calculators import FeatureMeta
from explore_exploit import ExploreExploit
# from features_algorithms.vertices.multi_dimensional_scaling import MultiDimensionalScaling
import feature_meta


CHOSEN_FEATURES = feature_meta.NODE_FEATURES
# CHOSEN_FEATURES = {"multi_dimensional_scaling": FeatureMeta(MultiDimensionalScaling, {"mds"})}


class LoadData:
    def __init__(self):
        self._data_dir = "signaling_pathways"
        self._label = "Adapter"
        df1 = pd.read_csv(os.path.join(self._data_dir, "signaling_pathways_2004.txt"))
        self._gnx = nx.from_pandas_edgelist(df1, "n1", "n2", create_using=nx.DiGraph())
        self._logger = PrintLogger("Signaling pathways")
        self._features_mx = None
        self._tags = {}

    def build_features(self):
        gnx_ftr = GraphFeatures(self._gnx, CHOSEN_FEATURES, dir_path=os.path.join(self._data_dir, "features"),
                                logger=self._logger)
        gnx_ftr.build(should_dump=True)  # build ALL_FEATURES
        self._features_mx = gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix)
        print(self._features_mx.shape)



    def build_tags(self):
        for label in ['Adapter', 'Ligand', 'Vesicles', 'Ribosomes', 'Membrane']:
            self._tags[label] = []
            with open(os.path.join(self._data_dir, "signaling_pathways_tags_" + label + ".txt"), "r") as f:
                for line in f:
                    node, tag = line.split()
                    self._gnx.node[node][label] = int(tag)
            for node in sorted(self._gnx):
                self._tags[label].append(self._gnx.node[node][label])

    def run_eps_greedy(self):
        dist_calc_type = "one_class"
        label_to_eps_to_steps = {x: {y: 0 for y in [0, 0.01, 0.05]} for x in
                                 ['Adapter', 'Ligand', 'Vesicles', 'Ribosomes', 'Membrane']}
        for label in ['Adapter', 'Ligand', 'Vesicles', 'Ribosomes', 'Membrane']:
            for eps in [0, 0.01, 0.05]:
                mean_steps = 0
                print(label, dist_calc_type)
                time_tag_dict = {}
                for i in range(1, 11):
                    exploration = ExploreExploit(self._tags[label], self._features_mx, 0.7, eps)
                    num_steps, tags = exploration.run(dist_calc_type)
                    print(" an recall of 70% was achieved in " + str(num_steps) + " steps")
                    mean_steps += num_steps
                    time_tag_dict[i] = tags
                time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
                time_tag_df.to_csv(label+"_"+dist_calc_type+"_output.csv")
                print("the mean num of steps is: " + str(mean_steps/10))
                label_to_eps_to_steps[label][eps] = mean_steps/10
        df = pd.DataFrame(label_to_eps_to_steps)


if __name__ == '__main__':
    dl = LoadData()
    dl.build_features()
    dl.build_tags()
    dl.run_eps_greedy()
    print("bla")

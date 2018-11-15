from collections import Counter

import networkx as nx
import numpy as np

from graph_measures.features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class BfsMomentsCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        for node in self._gnx:
            # calculate BFS distances
            distances = nx.single_source_shortest_path_length(self._gnx, node)
            # distances.pop(node)
            # if not distances:
            #     self._features[node] = [0., 0.]
            #     continue
            node_dist = Counter(distances.values())
            dists, weights = zip(*node_dist.items())
            # This was in the previous version
            # instead of the above commented fix
            adjusted_dists = [x + 1 for x in dists]
            self._features[node] = [float(np.average(weights, weights=adjusted_dists)), float(np.std(weights))]


feature_entry = {
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
}

if __name__ == "__main__":
    from graph_measures.measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(BfsMomentsCalculator, is_max_connected=True)

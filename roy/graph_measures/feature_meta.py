from graph_measures.features_algorithms.vertices.attractor_basin import AttractorBasinCalculator
from graph_measures.features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from graph_measures.features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
from graph_measures.features_algorithms.vertices.bfs_moments import BfsMomentsCalculator
from graph_measures.features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from graph_measures.features_algorithms.vertices.communicability_betweenness_centrality import \
     CommunicabilityBetweennessCentralityCalculator
from graph_measures.features_algorithms.vertices.eccentricity import EccentricityCalculator
from graph_measures.features_algorithms.vertices.fiedler_vector import FiedlerVectorCalculator
from graph_measures.features_algorithms.vertices.flow import FlowCalculator
from graph_measures.features_algorithms.vertices.general import GeneralCalculator
from graph_measures.features_algorithms.vertices.hierarchy_energy import HierarchyEnergyCalculator
from graph_measures.features_algorithms.vertices.k_core import KCoreCalculator
from graph_measures.features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from graph_measures.features_algorithms.vertices.louvain import LouvainCalculator
from graph_measures.features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator
from graph_measures.features_algorithms.vertices.motifs import nth_nodes_motif
from graph_measures.features_algorithms.vertices.page_rank import PageRankCalculator
from graph_measures.features_infra.feature_calculators import FeatureMeta, FeatureCalculator

NODE_FEATURES = {
    # Passed
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
    "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
    "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),

    # Didn't pass - but no logic
    "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
                                                          {"communicability"}),

    # Passed
    "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
    "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),

    # Previous version bug
    "flow": FeatureMeta(FlowCalculator, {}),
    # Passed
    "general": FeatureMeta(GeneralCalculator, {"gen"}),

    # Isn't OK - also in previous version
    "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),

    # Passed
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
    "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),

    # Didn't pass - but no logic
    "louvain": FeatureMeta(LouvainCalculator, {"lov"}),

    # Previous version bug
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),

    # Passed
    "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
    "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
    # "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    # "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}

GRAPH_FEATURES = {
    "feat_name": FeatureMeta(FeatureCalculator, {"abbr1", "abbr2"}),
}

ALL_FEATURES = {
    "node": NODE_FEATURES,
    # "edges": EDGE_FEATURES,
    "graph": GRAPH_FEATURES,
}

# DIRECTED_GRAPH_NODE_FEATURES = {
#     "feat_name": FeatureMeta(FeatureCalculator, {"abbr1", "abbr2"}),
# }

TEST_FEATURES = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
    "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
    "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
    "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
                                                          {"communicability"}),
    "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
    "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),  # Undirected graphs
    "flow": FeatureMeta(FlowCalculator, {}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
    "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
    "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),
    "louvain": FeatureMeta(LouvainCalculator, {"lov"}),  # Undirected graphs
    "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
}


def test_main():
    import numpy as np
    from features_infra.graph_features import GraphFeatures
    from loggers import PrintLogger
    import os
    import pickle
    import networkx as nx

    dataset = "citeseer"
    logger = PrintLogger("MetaTest")
    base_dir = r"/home/benami/git/pygcn/data"
    gnx = pickle.load(open(os.path.join(base_dir, dataset, "gnx.pkl"), 'rb'))

    max_subgnx = max(nx.connected_component_subgraphs(gnx.to_undirected()), key=len)
    gnx = gnx.subgraph(max_subgnx)

    features = GraphFeatures(gnx, TEST_FEATURES, dir_path="./%s_features_sub" % dataset, logger=logger)
    features.build(should_dump=True)
    measures_mx = features.to_matrix(add_ones=False, dtype=np.float32, mtype=np.matrix)
    logger.info("Finished")


if __name__ == "__main__":
    test_main()

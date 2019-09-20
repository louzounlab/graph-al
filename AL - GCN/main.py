import numpy as np
from gcn.train import main_clean, run_model, build_model, aggregate_results, mean_results
from explore_exploit import ExploreExploit, GraphExploreExploit, ExploreExploitF1, GraphExploreExploitF1, Learning
from gcn.data__loader import LoadData, GraphLoader
from grids import grid_learn_xgb, grid_learn_nn, grid_plot_xgb, grid_learn_gcn
from tools import StandardML, check_scores, average_dist_from_random_set
from plots import my_plot, my_active_plot, interactive_plot, table_plot, my_passive_plot, all_passive_plot, \
    plot_result_combined_with_avg_dist
from graph_measures import feature_meta
import pandas as pd
import os
import traceback
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats.mstats import zscore
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Data Sets
Data_Sets = ['cora', 'citeseer', 'email-Eu',
             'wikispeedia', 'pubmed', 'subelj_cora']
# Data_Sets = ['subelj_cora']

# chose features for the model
NODE_FEATURES = feature_meta.NODE_FEATURES
NEIGHBOR_FEATURES = feature_meta.NEIGHBOR_FEATURES


Results = {x: {} for x in Data_Sets}


def run_active_model(model, method, data, batch=1, iterations=1, out_interval=25, eps=0.05, out_prog=True, **params):
    try:
        res = model.run(option=method, iterations=iterations, out_prog=out_prog, batch_size=batch, eps=eps,
                        out_interval=out_interval, clear_model=True, epochs=100, **params)
        for key, val in res.items():
            if key not in Results[data]:
                Results[data][key] = {}
            if eps not in Results[data][key]:
                Results[data][key][eps] = {}
            Results[data][key][eps][model.eval_method] = val
        print("finish {} active eps={}".format(method, eps))
    except Exception:
        print("Error running {} active".format(method))
        traceback.print_exc()
    return


def active_learning(data, budget, batch=5, iterations=3, out_interval=25, eps=[0.05]):
    model = GraphExploreExploit(data, budget)
    params = [data, batch, iterations, out_interval]
    for epsilon in eps:
        # run_active_model(model, 'random', *params, eps=epsilon)
        # run_active_model(model, 'random', *params, eps=epsilon, **{'balance': True})
        # run_active_model(model, 'entropy', *params, eps=epsilon)
        # run_active_model(model, 'region_entropy', *params, eps=epsilon)
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'region_average_entropy': True})
        # run_active_model(model, 'rep_dist', *params, eps=epsilon, **{'representation_measure': 'mahalanobis'})
        # run_active_model(model, 'rep_dist', *params, eps=epsilon, **{'representation_measure': 'local_outlier_factor'})
        # run_active_model(model, 'rep_dist', *params, eps=epsilon, **{'representation_measure': 'mahalanobis',
        #                                                              'representation_region': True})
        # run_active_model(model, 'rep_dist', *params, eps=epsilon, **{'representation_measure': 'local_outlier_factor',
        #                                                              'representation_region': True})
        # run_active_model(model, 'geo_dist', *params, eps=epsilon)
        # run_active_model(model, 'APR', *params, eps=epsilon)
        # run_active_model(model, 'centrality', *params, eps=epsilon)
        # run_active_model(model, 'feature', *params, eps=epsilon, **{'feature': 'attractor_basin'})
        # run_active_model(model, 'geo_cent', *params, eps=epsilon)
        # run_active_model(model, 'Chang', *params, eps=epsilon)
        # run_active_model(model, 'k_truss', *params, eps=epsilon)
        # run_active_model(model, 'entropy', *params, eps=epsilon, **{'margin': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'margin': True})
        run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'margin': True,
                                                                           'region_average_entropy': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'region_include_self': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'margin': True,
        #                                                                    'region_include_self': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'region_average_entropy': True,
        #                                                                    'region_include_self': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'margin': True,
        #                                                                    'region_average_entropy': True,
        #                                                                    'region_include_self': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'region_include_self': True,
        #                                                                    'region_second_order': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'margin': True,
        #                                                                    'region_include_self': True,
        #                                                                    'region_second_order': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'region_average_entropy': True,
        #                                                                    'region_include_self': True,
        #                                                                    'region_second_order': True})
        # run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'margin': True,
        #                                                                    'region_average_entropy': True,
        #                                                                    'region_include_self': True,
        #                                                                    'region_second_order': True})

    my_active_plot(Results, data)
    print(Results)

    return None


def run_randoms(data, train_size={5}, times=1000):
    results = np.vstack(run_model(data, train_size=train_size) for _ in range(times))

    df = pd.DataFrame(results, columns=['% train', 'mic_f1', 'mac_f1', 'loss', 'acc', 'train'])
    interactive_plot(data_name=data, df=df)

    return results


def run_passive(data, trains, iterations=3, features_typ='neighbors'):
    Results[data][features_typ] = {}

    model = build_model(data, features=features_typ)
    gl = model.loader
    num_class = gl.num_labels
    for train in trains:
        try:
            tmp = []
            for _ in range(iterations):
                gl.split_test(1 - train)
                x_train, y_train, x_test, y_test = gl.get_train_test()

                # # code for normal features and labels
                # features = pickle.load(
                #     open(os.path.join(os.getcwd(), 'data_sets', data, 'additional_features.pkl'), 'rb'))
                # features = np.nan_to_num(zscore(features, axis=1))
                # labels = np.array(
                #     pickle.load(open(os.path.join(os.getcwd(), 'data_sets', data, 'all_labels.pkl'), 'rb')))
                # num_class = len(set(labels))
                # class_count = Counter(labels)
                # class_weights = np.array([1 / class_count[x] for x in sorted(class_count.keys())])
                # class_weights = class_weights / class_weights.sum()
                #
                # sss = StratifiedShuffleSplit(1, test_size=1-train)
                # for train_index, test_index in sss.split(features, labels):
                #     x_train, x_test = features[train_index], features[test_index]
                #     y_train, y_test = labels[train_index], labels[test_index]
                # score = check_scores(x_train, y_train, x_test, y_test, num_class, class_weights=class_weights)

                score = check_scores(x_train, y_train, x_test, y_test, num_class)
                gcn_model = model.train(epochs=200, early_stopping=True, split_val=True, verbose=1)
                gcn_score = model.test(gcn_model)
                score.update(gcn_score)
                tmp.append(score)

            aggregated = {}
            for result in tmp:
                for method, vals in result.items():
                    if method not in aggregated:
                        aggregated[method] = {}
                    for key, val in vals.items():
                        if key not in aggregated[method]:
                            aggregated[method][key] = []
                        aggregated[method][key].append(val)
            for method, vals in aggregated.items():
                if method not in Results[data][features_typ]:
                    Results[data][features_typ][method] = {}
                Results[data][features_typ][method][train] = {key: np.mean(val) for key, val
                                                              in aggregated[method].items()}
        except Exception:
            print("Error running {} {} passive".format(features_typ, train))
            traceback.print_exc()

    # for param in ['acc', 'mic_f1', 'mac_f1']:
    #     my_passive_plot(Results, data, features_typ, param)

    return


def passive_learning(data, trains, iterations=3):
    try:
        run_passive(data, trains, iterations=iterations, features_typ='neighbors')
        run_passive(data, trains, iterations=iterations, features_typ='features')
        run_passive(data, trains, iterations=iterations, features_typ='both')

        for param in ['acc', 'mic_f1', 'mac_f1']:
            all_passive_plot(Results, data, param)
        print(Results)
    except Exception:
        print("Error running {} passive".format(data))
        traceback.print_exc()
    return


def average_dist_to_labeled_nodes(data, stop=0.8, iterations=10, batch_size=1):
    try:
        print("start")
        g_path = os.path.join(os.getcwd(), 'data_sets', data, 'gnx.pkl')
        gnx = nx.read_gpickle(g_path)
        xs, ys = average_dist_from_random_set(gnx, stop=stop, iterations=iterations, batch_size=batch_size)
        with open('average distance {}.txt'.format(data), 'w+') as f:
            f.write(','.join(str(x) for x in xs))
            f.write('\n')
            f.write(','.join(str(y) for y in ys))
        plt.figure()
        plt.plot(xs, ys)
        plt.title("{} average distance per labeled size".format(data))
        plt.xlabel('% of data revealed')
        plt.ylabel('average_distance to labeled')
        fig = plt.gcf()
        # fig.show()
        fig.savefig("{} average distance per labeled size.png".format(data))
        print("finish {}".format(data))
    except Exception:
        print("Error running average distance on {}".format(data))
        traceback.print_exc()
    return


if __name__ == '__main__':

    e1 = [0]
    for dataset in Data_Sets:
        active_learning(dataset, 0.15, out_interval=60, eps=e1, iterations=20, batch=1)
        # active_learning(dataset, 200, out_interval=50, eps=e1, iterations=20, batch=1)
        # run_randoms(dataset, train_size={70}, times=1)
        # table_plot(dataset, std=False)
        # passive_learning(dataset, trains=np.linspace(0.05, 0.3, 6), iterations=5)
        # average_dist_to_labeled_nodes(dataset, stop=0.2, iterations=15, batch_size=25)
        # plot_result_combined_with_avg_dist(dataset)
        # main_clean(dataset, train_size={5}, early_stopping=False, val_p=0.1)

    print(Results)
    print("done")

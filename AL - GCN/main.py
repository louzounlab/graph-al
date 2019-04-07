import numpy as np
from gcn.train import main_clean, run_model
from explore_exploit import ExploreExploit, GraphExploreExploit, ExploreExploitF1, GraphExploreExploitF1, Learning
from grids import grid_learn_xgb, grid_learn_nn, grid_plot_xgb, grid_learn_gcn
from tools import check_acc, StandardML
from plots import my_plot, my_active_plot, interactive_plot, table_plot
from graph_measures import feature_meta

import warnings
warnings.filterwarnings("ignore")

# Data Sets
# Data_Sets = ['personality_14_nov', 'signaling_pathways']
Data_Sets = ['moreno_blogs', 'email-Eu',
             'cora2', 'citeseer2', 'pubmed2',
             'wikispeedia', 'subelj_cora', 'hard_challenge']
# Data_Sets = ['moreno_blogs', 'email-Eu',
#              'cora2', 'citeseer2']


# chose features for the model
CHOSEN_FEATURES = feature_meta.NODE_FEATURES
NEIGHBOR_FEATURES = feature_meta.NEIGHBOR_FEATURES


Results = {x: {} for x in Data_Sets}


def run_active_model(model, method, data, batch=1, iterations=1, out_interval=25, eps=0.05, out_prog=True):
    res = model.run(option=method, iterations=iterations, out_prog=out_prog, batch_size=batch, out_interval=out_interval, eps=eps,
                    clear_model=True, epochs=100)
    for key, val in res.items():
        if key not in Results[data]:
            Results[data][key] = {}
        if eps not in Results[data][key]:
            Results[data][key][eps] = {}
        Results[data][key][eps][method] = val
    print("finish {} active eps={}".format(method, eps))
    return


def active_learning(data, budget, batch=5, iterations=3, out_interval=25, epsilons=[0.05]):
    model = GraphExploreExploit(data, budget)
    params = [data, batch, iterations, out_interval]
    for epsilon in epsilons:
        run_active_model(model, 'entropy', *params, eps=epsilon)
        run_active_model(model, 'region_entropy', *params, eps=epsilon)
        run_active_model(model, 'representation_euclidean_dist', *params, eps=epsilon)
        run_active_model(model, 'representation_one_class_dist', *params, eps=epsilon)
        run_active_model(model, 'centrality', *params, eps=epsilon)
        run_active_model(model, 'Chang', *params, eps=epsilon)
        run_active_model(model, 'geo_dist', *params, eps=epsilon)
        run_active_model(model, 'geo_in_dist', *params, eps=epsilon)
        run_active_model(model, 'geo_out_dist', *params, eps=epsilon)
        run_active_model(model, 'Roy', *params, eps=epsilon)

    # pass_res = main_clean(data, budget=budget, iterations=iterations, early_stopping=False,
    #                       val_p=0, out_intervals=out_interval)
    # for key in Results[data].keys():
    #     res = {train_s: pass_res[train_s][key] for train_s in pass_res.keys()}
    #     for key2 in Results[data][key].keys():
    #         Results[data][key][key2]['Passive'] = res
    # print("finish passive")

    my_active_plot(Results, data)
    print(Results)

    return None


def run_randoms(data, train_size={5}, times=1000):
    results = np.vstack(run_model(data, train_size=train_size) for _ in range(times))

    # df = pd.DataFrame(results, columns=['% train', 'mic_f1', 'mac_f1', 'loss', 'acc', 'train'])
    # interactive_plot(data_name=data, df=df)

    return results


if __name__ == '__main__':

    epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for dataset in Data_Sets:
        active_learning(dataset, 0.4, out_interval=80, epsilons=epsilons, iterations=3, batch=3)
        # run_randoms(dataset, train_size={70}, times=1)
        table_plot(dataset)

    print(Results)
    print("done")

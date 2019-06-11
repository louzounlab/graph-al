import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
import time
import numpy as np
import glob
import traceback
import bisect


def my_plot(results):
    for data_set in results:
        for label in results[data_set]:
            for i in results[data_set][label]:
                xs = [results[data_set][label][i][x] for x in results[data_set][label][i]]
                ys = [x for x in results[data_set][label][i]]
                plt.plot(xs, ys, label=str(i))
            plt.title(str(data_set)+"_"+str(label)+"_recall per time")
            plt.legend()
            plt.xlabel('data_revealed')
            plt.ylabel('recall')
            fig = plt.gcf()
            fig.show()
            fig.savefig(str(data_set)+"_"+str(label)+'_recall_per_time.png')


def my_active_plot(results, data_set=None, param='acc'):
    try:
        for model in results[data_set]:
            result = results[data_set][model]
            for epsilon in result.keys():
                plt.figure()
                colors = pl.cm.jet(np.linspace(0, 1, len(result[epsilon])))
                for i, method in enumerate(result[epsilon]):
                    xs = list(result[epsilon][method].keys())
                    xs.sort()
                    ys = [result[epsilon][method][x][param] for x in xs]
                    plt.plot(xs, ys, label=str(method), color=colors[i])
                plt.title("{} {} eps={} {} per time".format(data_set, model, epsilon, param))
                plt.legend()
                plt.xlabel('% of data revealed')
                plt.ylabel(param)
                fig = plt.gcf()
                # fig.show()
                fig.savefig("{} {} eps={} {}_per_time.png".format(data_set, model, epsilon, param))
    except Exception:
        print("Error data set: {}".format(data_set))
        traceback.print_exc()


def my_passive_plot(results, data_set=None, features='neighbor', param='acc'):
    result = results[data_set][features]
    plt.figure()
    # colors = pl.cm.jet(np.linspace(0, 1, len(result)))
    # for i, method in enumerate(result):
    for method in result:
        xs = list(result[method].keys())
        xs.sort()
        ys = [result[method][x][param] for x in xs]
        # plt.plot(xs, ys, label=str(method), color=colors[i])
        plt.plot(xs, ys, label=method if method in {'XGB', 'RF', 'NN'} else 'GCN')
    plt.title("passive {} {} {} per train".format(data_set, features, param))
    plt.legend()
    plt.xlabel('% train size')
    plt.ylabel(param)
    plt.savefig("passive {} {} {} per train.png".format(data_set, features, param))
    # fig = plt.gcf()
    # fig.show()
    # fig.savefig("passive {} {} {} per train.png".format(data_set, features, param))

    return


def all_passive_plot(results, data_set=None, param='acc'):
    f = open(os.path.join(os.getcwd(), 'passive', "{} {}.csv".format(data_set, param)), 'w')
    w = csv.writer(f)
    result = results[data_set]
    # plt.figure()
    # colors = pl.cm.jet(np.linspace(0, 1, len(result)*len(result[list(result.keys())[0]])))
    # i = 0
    for feat_typ in result.keys():
        for method in result[feat_typ]:
            xs = list(result[feat_typ][method].keys())
            xs.sort()
            ys = [result[feat_typ][method][x][param] for x in xs]
            # plt.plot(xs, ys, label=feat_typ+method if method in {'XGB', 'RF', 'NN'} else feat_typ+'GCN',
            #          color=colors[i])
            # i += 1
            w.writerow([feat_typ+method if method in {'XGB', 'RF', 'NN'} else feat_typ+'GCN'])
            w.writerow(xs)
            w.writerow(ys)
    # plt.title("passive {} {} per train".format(data_set, param))
    # plt.legend()
    # plt.xlabel('% train size')
    # plt.ylabel(param)
    # plt.savefig("passive {} {} per train.png".format(data_set, param))
    # fig = plt.gcf()
    # # fig.show()
    # fig.savefig("passive {} {} per train.png".format(data_set, param))
    f.close()

    return


def interactive_plot(data_name, df: pd.DataFrame):

    # Output the visualization directly in the notebook
    output_file('random samples {}.html'.format(data_name), title='random samples {}'.format(data_name))

    # Create a figure with no toolbar and axis ranges of [0,3]
    fig = figure(title='random samples {}'.format(data_name),
                 plot_height=600, plot_width=400,
                 x_range=(0, 15), y_range=(0, 1),
                 toolbar_location=None)

    # Draw the coordinates as circles
    fig.circle(x=df.columns[0], y=df.columns[1], source=df,
               color='green', size=5, alpha=0.5)

    # Format the tooltip
    tooltips = [('{}'.format(col), '@{}'.format(col)) for col in df.columns[1:]]

    # Add the HoverTool to the figure
    fig.add_tools(HoverTool(tooltips=tooltips))

    # Show plot
    show(fig)


def table_plot(data_name, std=False):
    try:
        log_file = os.path.join(os.getcwd(), 'tables', 'logs', 'results_{}.csv'.format(data_name))
        df = pd.DataFrame.from_csv(log_file)
        if std:
            stds = df.groupby(['model_name', 'train_p']).std()
            out = stds.swaplevel(0, 1).T.stack()
        else:
            means = df.groupby(['model_name', 'train_p']).mean()
            out = means.swaplevel(0, 1).T.stack()

        if not os.path.exists(os.path.join(os.getcwd(), 'tables', data_name)):
            os.mkdir(os.path.join('tables', data_name))

        out_file = data_name + '_table' + time.strftime("-%Y-%m-%d-%H%M%S") + '.xlsx'
        out_path = os.path.join(os.getcwd(), 'tables', data_name, out_file)
        out.to_excel(out_path)
    except Exception:
        print("Error table {}".format(data_name))
        traceback.print_exc()

    return


def plot_result_combined_with_avg_dist(data_name, budget=15):
    try:
        # txt_path = os.path.join(os.getcwd(), 'plots', 'average distance {}.txt'.format(data_name))
        # avg_dist = []
        # with open(txt_path, 'r') as f2:
        #     for i, line in enumerate(f2):
        #         split_l = line.split(',')
        #         if i == 0:
        #             avg_dist.append([float(x) * 100 for x in split_l])
        #         else:
        #             avg_dist.append([float(x) for x in split_l])

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('% of data revealed')
        ax1.set_ylabel('accuracy (compared to random)')
        xls_path = os.path.join(os.getcwd(), 'plots', '{}_table.xlsx'.format(data_name))
        f1 = pd.read_excel(xls_path, sheet_name='Sheet3')
        for i in range(len(f1)):
            ax1.plot(f1.iloc[i], label=f1.iloc[i].name)

        # ax2 = ax1.twinx()
        # ax2.set_ylabel('avg distance (labeled to unlabeled)')
        # budget_limit = bisect.bisect_left(avg_dist[0], budget)
        # ax2.plot(avg_dist[0][:budget_limit+1], avg_dist[1][:budget_limit+1], label='avg_dists')
        fig.suptitle("{}".format(data_name), va='center')
        fig.legend(loc='center right')
        fig.show()
        fig.savefig("{} best.png".format(data_name))
    except Exception:
        print("Error data set {}".format(data_name))
        traceback.print_exc()

    return


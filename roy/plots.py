import os
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
import time


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
    for model in results[data_set]:
        result = results[data_set][model]
        for epsilon in result.keys():
            plt.figure()
            for method in result[epsilon]:
                xs = list(result[epsilon][method].keys())
                xs.sort()
                ys = [result[epsilon][method][x][param] for x in xs]
                plt.plot(xs, ys, label=str(method))
            plt.title("{} {} eps={} {} per time".format(data_set, model, epsilon, param))
            plt.legend()
            plt.xlabel('% of data revealed')
            plt.ylabel(param)
            fig = plt.gcf()
            # fig.show()
            fig.savefig("{} {} eps={} 2lin {}_per_time.png".format(data_set, model, epsilon, param))


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


def table_plot(data_name):
    log_file = os.path.join(os.getcwd(), 'tables', 'logs', 'results_{}.csv'.format(data_name))
    df = pd.DataFrame.from_csv(log_file)
    means = df.groupby(['model_name', 'train_p']).mean()
    out = means.swaplevel(0, 1).T.stack()

    if not os.path.exists(os.path.join(os.getcwd(), 'tables', data_name)):
        os.mkdir(os.path.join('tables', data_name))

    out_file = data_name + '_table' + time.strftime("-%Y-%m-%d-%H%M%S") + '.xlsx'
    out_path = os.path.join(os.getcwd(), 'tables', data_name, out_file)
    out.to_excel(out_path)

    return


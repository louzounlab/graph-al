import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import csv
import itertools
from neural_network.nn_activator import FeedForwardNet
from neural_network.nn_models import NeuralNet
from gcn.train import main_clean
import matplotlib.pyplot as plt


# def grid_learn_xgb(data_name, x_data, labels, epochs=100):
def grid_learn_xgb(data_name, x_train, y_train, x_test, y_test, labels, epochs=100, train_size=0.8, n_run=1):

    y_train = [labels.index(i) for i in y_train]
    y_test = [labels.index(i) for i in y_test]

    if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check')):
        os.mkdir('parameter_check')
    if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check', data_name)):
        os.mkdir(os.path.join('parameter_check', data_name))
    # train percentage
    for train_p in [train_size*100]:
        f = open(os.path.join(os.getcwd(), 'parameter_check', data_name, "results_train_p" + str(train_p) +
                              "_" + str(n_run) + "_dart" + ".csv"), 'w')
        w = csv.writer(f)
        w.writerow(['max_depth', 'lambda', 'eta', 'rate drop', 'gamma',
                    'train_microF1', 'train_macroF1', 'train acc',
                    'test_microF1', 'test_macroF1', 'test acc'])
        for max_depth, lamb, eta, rate_drop, gam in \
                itertools.product(range(3, 16, 4), range(1, 20, 6), np.logspace(-3, -0.5, 5),
                                  [0.05, 0.1, 0.2, 0.35, 0.5, 0.7],
                                  [0, 0.1, 0.5, 1, 3, 7]):
            # acc_train = []
            # acc_test = []
            # macrof1_test = []
            # microf1_test = []
            # macrof1_train = []
            # microf1_train = []
            for num_splits in range(1, epochs+1):
                # x_train, x_test, y_train, y_test = train_test_split(x_data, new_labels, test_size=1 - float(train_p)/100)
                n_x_train, n_x_eval, n_y_train, n_y_eval = train_test_split(x_train, y_train, test_size=0.1)
                dtrain = xgb.DMatrix(n_x_train, label=n_y_train, silent=True)
                dtest = xgb.DMatrix(x_test, label=y_test, silent=True)
                deval = xgb.DMatrix(n_x_eval, label=n_y_eval, silent=True)
                params = {'silent': True, 'booster': 'dart', 'tree_method': 'auto', 'max_depth': max_depth,
                          'lambda': lamb / 10, 'eta': eta, 'rate_drop': rate_drop, 'gamma': gam,
                          'sample_type': 'weighted', 'normalize_type': 'forest',
                          'objective': 'multi:softprob', 'num_class': len(labels)}
                clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                    early_stopping_rounds=10, verbose_eval=False)
                y_score_test = clf_xgb.predict(dtest, ntree_limit=1)
                y_pred_test = [np.argmax(i) for i in y_score_test]
                y_score_train = clf_xgb.predict(dtrain, ntree_limit=1)
                y_pred_train = [np.argmax(i) for i in y_score_train]
                # ROC AUC has a problem with only one class
                try:
                    test_macf1 = f1_score(y_test, y_pred_test, average='macro')
                    test_micf1 = f1_score(y_test, y_pred_test, average='micro')
                    test_acc = accuracy_score(y_test, y_pred_test)
                    # r1 = roc_auc_score(y_test, y_score_test)
                except ValueError:
                    continue
                # macrof1_test.append(macf1)
                # microf1_test.append(micf1)
                # acc_test.append(test_acc)

                try:
                    train_macf1 = f1_score(n_y_train, y_pred_train, average='macro')
                    train_micf1 = f1_score(n_y_train, y_pred_train, average='micro')
                    train_acc = accuracy_score(n_y_train, y_pred_train)
                    # r2 = roc_auc_score(y_train, y_score_train)
                except ValueError:
                    continue
                # macrof1_train.append(macf1)
                # microf1_train.append(micf1)
                # acc_train.append(train_acc)
            w.writerow([str(max_depth), str(lamb / 10), str(eta), str(rate_drop), str(gam),
                        str(train_micf1), str(train_macf1), str(train_acc),
                        str(test_micf1), str(test_macf1), str(test_acc)])
                        # str(rate_drop / 10), str(np.mean(microf1_train)), str(np.mean(macrof1_train)),
                        # str(np.mean(acc_train)), str(np.mean(microf1_test)), str(np.mean(macrof1_test)),
                        # str(np.mean(acc_test))])
    return None


# def grid_learn_nn(data_name, x_data, labels, epochs=100):
def grid_learn_nn(data_name, x_train, y_train, x_test, y_test, labels, epochs=100, n_run=1):

    num_classes = len(labels)

    if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check')):
        os.mkdir('parameter_check')
    if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check', data_name)):
        os.mkdir(os.path.join('parameter_check', data_name))

    in_dim = x_train.shape[1]
    layers = [(in_dim, int(in_dim*2/3), int(in_dim/3)), (in_dim, int(in_dim*3), int(in_dim*1.5), int(in_dim*0.5)),
              (in_dim, int(in_dim * 2), int(in_dim * 1.2), int(in_dim * 0.3))]

    # train percentage
    for train_p in [80]:
        f = open(os.path.join(os.getcwd(), 'parameter_check', data_name, "results_train_p" + str(train_p) +
                              "_" + str(n_run) + "_nn" + ".csv"), 'w')
        w = csv.writer(f)
        w.writerow(['batch size', 'layers dim', 'lr', 'drop out', 'min l2 penalty weight', 'activation func',
                    'macro f1 test', 'micro f1 test', 'acc test', 'loss test',
                    'macro f1 train', 'micro f1 train', 'acc train', 'loss train'])
        for batch_size, layers_dim, lr, drop_out, l2_penalty, activation_func in \
                itertools.product([32, 64, 128], layers, [0.0001, 0.0005, 0.001, 0.0015, 0.003],
                                  range(1, 11, 2),  np.logspace(-3, -0.5, 5), ["relu", "elu"]):
            # acc_train = []
            # acc_test = []
            # loss_train = []
            # loss_test = []
            # macrof1_test = []
            # microf1_test = []
            # macrof1_train = []
            # microf1_train = []
            for num_splits in range(1, epochs+1):
                # x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=1 - float(train_p) / 100)

                net = FeedForwardNet(NeuralNet(num_classes, batch_size=batch_size, layers_dim=layers_dim,
                                               drop_out=drop_out / 10, l2_penalty=l2_penalty,
                                               activation_func=activation_func, lr=lr), gpu=True)
                net.set_data(x_train, y_train)
                net.train(total_epoch=80, stop_loss=True)
                train_loss = net.test(train=True)
                test_loss = net.test()
                y_pred_train = net.predict(x_train)
                y_pred_test = net.predict(x_test)

                # scores for test
                test_macf1 = f1_score(y_test, y_pred_test, average='macro')
                test_micf1 = f1_score(y_test, y_pred_test, average='micro')
                test_acc = accuracy_score(y_test, y_pred_test)
                # macrof1_test.append(macf1)
                # microf1_test.append(micf1)
                # acc_test.append(test_acc)
                # loss_test.append(test_loss)

                # scores for train
                train_macf1 = f1_score(y_train, y_pred_train, average='macro')
                train_micf1 = f1_score(y_train, y_pred_train, average='micro')
                train_acc = accuracy_score(y_train, y_pred_train)
                # macrof1_train.append(macf1)
                # microf1_train.append(micf1)
                # acc_train.append(train_acc)
                # loss_train.append(train_loss)

            w.writerow([str(batch_size), str(layers_dim), str(lr), str(drop_out), str(l2_penalty),
                        str(activation_func), str(test_macf1), str(test_micf1),
                        str(test_acc), str(test_loss), str(train_macf1),
                        str(train_micf1), str(train_acc), str(train_loss)])
                        # str(activation_func), str(np.mean(macrof1_test)), str(np.mean(microf1_test)),
                        # str(np.mean(acc_test)), str(np.mean(loss_test)), str(np.mean(macrof1_train)),
                        # str(np.mean(microf1_train)), str(np.mean(acc_train)), str(np.mean(loss_train))])
    return None


# def grid_learn_xgb(data_name, x_data, labels, epochs=100):
def grid_plot_xgb(data_name, train_sizes=[80], thresholds=None, n_runs=1):

    args = ['depth', 'lambda', 'eta', 'drop_rate', 'gamma']
    if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check', data_name, 'plots')):
        os.mkdir(os.path.join(os.getcwd(), 'parameter_check', data_name, 'plots'))
    # drop_rates = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7]
    # drop_rates = [str(rate) for rate in drop_rates]
    for a, arg in enumerate(args):
        for p, train_p in enumerate(train_sizes):
            # results = {rate: [[], []] for rate in drop_rates}
            xs = []
            ys = []
            for i in range(n_runs):
                with open(os.path.join(os.getcwd(), 'parameter_check', data_name, "results_train_p" + str(train_p) +
                                       "_" + str(i) + "_dart" + ".csv"), 'r') as f:
                    # for line in f:
                    #     lis = line.split()
                    #     results[lis[3]][0].append(lis[8])
                    #     results[lis[3]][1].append(lis[9])
                    lis = [line.split(',') for line in f]
                    # xs.extend(float(lis[l][5]) for l in range(1, len(lis)))
                    # ys.extend(float(lis[l][8]) for l in range(1, len(lis)))
                    for l in range(1, len(lis)):
                        if thresholds[p][0][0] <= float(lis[l][5]) <= thresholds[p][0][1] and \
                                float(lis[l][8]) > thresholds[p][1] and float(lis[l][3]) == 0.2 and \
                                float(lis[l][4]) == 7 and float(lis[l][0]) == 7 \
                                and float(lis[l][1]) == 1.3:

                            xs.append(float(lis[l][a]))
                            ys.append(float(lis[l][8]))

                    # xs.extend(float(lis[l][5]) if float(lis[l][5]) > thresholds[p][0] else None for l in range(1, len(lis)))
                    # ys.extend(float(lis[l][8]) if float(lis[l][8]) > thresholds[p][1] else None for l in range(1, len(lis)))
                    # for l in range(1, len(lis)):
                    #     results[lis[l][3]][0].append(float(lis[l][8]))
                    #     results[lis[l][3]][1].append(float(lis[l][9]))
            # for rate in drop_rates:
            #     plt.scatter(results[rate][0], results[rate][1])
            plt.scatter(xs, ys)
            # plt.title(str(data_name) + "_train:" + str(train_p) + "_drop:" + str(rate))
            plt.title(str(data_name) + "_train:" + str(train_p) + "_" + str(arg))
            # plt.legend()
            plt.xlabel(arg)
            plt.ylabel('test_microF1')
            fig = plt.gcf()
            # fig.show()
            name = str(data_name) + "_train_" + str(train_p) + "_" + str(arg) + ".png"
            name = 'Threshold_' + name if thresholds else name
            fig.savefig(os.path.join(os.getcwd(), 'parameter_check', data_name, 'plots', name))

    return None


def grid_learn_gcn(data_name):

    data_path = os.path.join("data_sets", data_name)

    if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check')):
        os.mkdir('parameter_check')
    if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check', data_name)):
        os.mkdir(os.path.join('parameter_check', data_name))

    f = open(os.path.join(os.getcwd(), 'parameter_check', data_name, "results_gcn.csv"), 'w')
    w = csv.writer(f)
    w.writerow(['model', 'dropout', 'learning rate', 'weight decay', 'hidden_layers',
                'Loss', 'MicroF1', 'Accuracy'])
    # dropouts = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    # learning_rates = [0.001, 0.01, 0.05, 0.1]
    # weights_decay = [0.0005, 0.001, 0.005, 0.01]
    # hidden_layers = [[16], [64, 16], [100, 35], [225, 75, 25]]
    dropouts = [0.2, 0.5, 0.8]
    learning_rates = [0.001, 0.01, 0.05]
    weights_decay = [0.001, 0.01]
    hidden_layers = [[16], [64, 16], [100, 35], [225, 75, 25]]

    for dropout, lr, weight_decay, layers in \
            itertools.product(dropouts, learning_rates, weights_decay, hidden_layers):
        results = main_clean(data_name, data_path, dropout=dropout, lr=lr,
                             weight_decay=weight_decay, hidden_layers=layers)
        print("results are: ")
        print(results)

        for name, vals in results.items():
            w.writerow([str(name), str(dropout), str(lr), str(weight_decay), str(layers),
                        str(vals['loss']), str(vals['mic_f1']), str(vals['acc'])])

    return None

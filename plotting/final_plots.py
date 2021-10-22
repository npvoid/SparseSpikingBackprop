import os
from plot_utils import plot_error, plt_set, panel_specs, label_panel
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib
import pickle
import numpy as np
from scipy import stats
import seaborn as sns

import matplotlib.style as style

from matplotlib import cm
tab_colors = cm.get_cmap('tab20', 20).colors
paired_colors = cm.get_cmap('Paired', 12).colors

COLORS_LAYERS = [tab_colors[0], tab_colors[2]]
COLORS_ORIG_SPARSE = ['grey', tab_colors[2]]
HEATMAP = "YlGnBu"
COLORS_DATASET = ['#000000', '#1F77B4', '#FF7F0E']


plt_set(0.75)

def read_data(nb_trials, hidden_list, path_read, dataset_name):
    hidden_dict = dict(zip(hidden_list, np.arange(len(hidden_list))))

    loss_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwd_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwd_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwdm_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwdm_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    train_acc_orig = np.zeros((nb_trials, len(hidden_list)))
    test_acc_orig = np.zeros((nb_trials, len(hidden_list)))
    loss_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwd_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwd_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwdm_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwdm_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    train_acc_s3gd = np.zeros((nb_trials, len(hidden_list)))
    test_acc_s3gd = np.zeros((nb_trials, len(hidden_list)))
    spike_counts1 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    spike_counts2 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    active_counts1 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    active_counts2 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]

    PATH_READ = path_read
    count = 0
    for root, dirs, files in os.walk(PATH_READ):
        for file in files:
            if file == 'data.p':
                count += 1
                print(count)
                d = pickle.load(open(os.path.join(root, file), 'rb'))

                if 'ORIG' in root:
                    trial = d['prs_orig']['seed']-1
                    if (trial > nb_trials - 1) or (d['prs_orig']['nb_hidden'] not in hidden_list):
                        continue
                    n = hidden_dict[d['prs_orig']['nb_hidden']]
                    loss_orig[trial][n] = np.squeeze(np.array(d['loss_orig']))
                    fwd_orig[trial][n] = np.squeeze(np.array(d['fwd_orig']))
                    bwd_orig[trial][n] = np.squeeze(np.array(d['bwd_orig']))
                    fwdm_orig[trial][n] = np.squeeze(np.array(d['fwdm_orig']))
                    bwdm_orig[trial][n] = np.squeeze(np.array(d['bwdm_orig']))
                    train_acc_orig[trial, n] = np.squeeze(np.array(d['train_acc_orig']))
                    test_acc_orig[trial, n] = np.squeeze(np.array(d['test_acc_orig']))

                elif 'S3GD' in root:
                    trial = d['prs_s3gd']['seed']-1

                    if (trial > nb_trials - 1) or (d['prs_s3gd']['nb_hidden'] not in hidden_list):
                        continue
                    n = hidden_dict[d['prs_s3gd']['nb_hidden']]
                    loss_s3gd[trial][n] = np.squeeze(np.array(d['loss_s3gd']))
                    fwd_s3gd[trial][n] = np.squeeze(np.array(d['fwd_s3gd']))
                    bwd_s3gd[trial][n] = np.squeeze(np.array(d['bwd_s3gd']))
                    fwdm_s3gd[trial][n] = np.squeeze(np.array(d['fwdm_s3gd']))
                    bwdm_s3gd[trial][n] = np.squeeze(np.array(d['bwdm_s3gd']))
                    train_acc_s3gd[trial, n] = np.squeeze(np.array(d['train_acc_s3gd']))
                    test_acc_s3gd[trial, n] = np.squeeze(np.array(d['test_acc_s3gd']))
                    counts = d['counts']
                    spike_counts1[trial][n] = counts['spike_counts1']
                    spike_counts2[trial][n] = counts['spike_counts2']
                    active_counts1[trial][n] = counts['active_counts1']
                    active_counts2[trial][n] = counts['active_counts2']

                    batch_size = d['prs_s3gd']['batch_size']
                    nb_steps = d['prs_s3gd']['nb_steps']

    # Process data
    loss_orig = np.array(loss_orig)
    fwd_orig = np.array(fwd_orig)
    bwd_orig = np.array(bwd_orig)
    fwdm_orig = np.array(fwdm_orig) / (1024 ** 2)  # MiB
    bwdm_orig = np.array(bwdm_orig) / (1024 ** 2)
    loss_s3gd = np.array(loss_s3gd)
    fwd_s3gd = np.array(fwd_s3gd)
    bwd_s3gd = np.array(bwd_s3gd)
    fwdm_s3gd = np.array(fwdm_s3gd) / (1024 ** 2)
    bwdm_s3gd = np.array(bwdm_s3gd) / (1024 ** 2)
    spike_counts1 = np.array(spike_counts1)
    spike_counts2 = np.array(spike_counts2)
    active_counts1 = np.array(active_counts1)
    active_counts2 = np.array(active_counts2)

    train_acc_orig_mean = np.mean(train_acc_orig, axis=0)
    train_acc_orig_error = stats.sem(train_acc_orig, axis=0)
    train_acc_s3gd_mean = np.mean(train_acc_s3gd, axis=0)
    train_acc_s3gd_error = stats.sem(train_acc_s3gd, axis=0)
    test_acc_orig_mean = np.mean(test_acc_orig, axis=0)
    test_acc_orig_error = stats.sem(test_acc_orig, axis=0)
    test_acc_s3gd_mean = np.mean(test_acc_s3gd, axis=0)
    test_acc_s3gd_error = stats.sem(test_acc_s3gd, axis=0)

    loss_orig_tavg = np.mean(loss_orig, axis=2)
    fwd_orig_tavg = np.mean(fwd_orig, axis=2)
    bwd_orig_tavg = np.mean(bwd_orig, axis=2)
    fwdm_orig_tavg = np.mean(fwdm_orig, axis=2)
    bwdm_orig_tavg = np.mean(bwdm_orig, axis=2)
    loss_s3gd_tavg = np.mean(loss_s3gd, axis=2)
    fwd_s3gd_tavg = np.mean(fwd_s3gd, axis=2)
    bwd_s3gd_tavg = np.mean(bwd_s3gd, axis=2)
    fwdm_s3gd_tavg = np.mean(fwdm_s3gd, axis=2)
    bwdm_s3gd_tavg = np.mean(bwdm_s3gd, axis=2)

    d = {
        'loss_orig': loss_orig,
        'fwd_orig': fwd_orig,
        'bwd_orig': bwd_orig,
        'fwdm_orig': fwdm_orig,
        'bwdm_orig': bwdm_orig,
        'loss_s3gd': loss_s3gd,
        'fwd_s3gd': fwd_s3gd,
        'bwd_s3gd': bwd_s3gd,
        'fwdm_s3gd': fwdm_s3gd,
        'bwdm_s3gd': bwdm_s3gd,
        'spike_counts1': spike_counts1,
        'spike_counts2': spike_counts2,
        'active_counts1': active_counts1,
        'active_counts2': active_counts2,

        'train_acc_orig_mean': train_acc_orig_mean,
        'train_acc_orig_error': train_acc_orig_error,
        'train_acc_s3gd_mean': train_acc_s3gd_mean,
        'train_acc_s3gd_error': train_acc_s3gd_error,
        'test_acc_orig_mean': test_acc_orig_mean,
        'test_acc_orig_error': test_acc_orig_error,
        'test_acc_s3gd_mean': test_acc_s3gd_mean,
        'test_acc_s3gd_error': test_acc_s3gd_error,

        'loss_orig_tavg': loss_orig_tavg,
        'fwd_orig_tavg': fwd_orig_tavg,
        'bwd_orig_tavg': bwd_orig_tavg,
        'fwdm_orig_tavg': fwdm_orig_tavg,
        'bwdm_orig_tavg': bwdm_orig_tavg,
        'loss_s3gd_tavg': loss_s3gd_tavg,
        'fwd_s3gd_tavg': fwd_s3gd_tavg,
        'bwd_s3gd_tavg': bwd_s3gd_tavg,
        'fwdm_s3gd_tavg': fwdm_s3gd_tavg,
        'bwdm_s3gd_tavg': bwdm_s3gd_tavg,

        'batch_size': batch_size,
        'nb_steps': nb_steps,

        'hidden_dict': hidden_dict,
        'dataset_name': dataset_name,
        'dataset_title': {'F-MNIST': 'F-MNIST', 'N-MNIST': 'N-MNIST', 'SHD': 'SHD'}[dataset_name],

    }
    return d

# ####################################### Panel 1 ####################################### #
# A
def plot_activity(d_DATASET, ax, nb_neurons=200, colors=None):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors
    def numfmt(x, pos):
        s = '{}'.format(x / 1000.0)
        return s

    n = d_DATASET['hidden_dict'][nb_neurons]
    total_vars_first_hidden = d_DATASET['batch_size'] * d_DATASET['nb_steps'] * nb_neurons
    total_vars_second_hidden = d_DATASET['batch_size'] * d_DATASET['nb_steps'] * nb_neurons
    active_counts1 = d_DATASET['active_counts1'][:, n, :]
    active_counts2 = d_DATASET['active_counts2'][:, n, :]

    mean_activity_layer1 = active_counts1.mean() / total_vars_first_hidden
    mean_activity_layer2 = active_counts2.mean() /total_vars_second_hidden
    print("Mean activity Layer 1 {}: {:.4f}".format(d_DATASET['dataset_title'], mean_activity_layer1))
    print("Mean activity Layer 2 {}: {:.4f}".format(d_DATASET['dataset_title'], mean_activity_layer2))
    print("Improve grad_w {}: {:.4f}".format(d_DATASET['dataset_title'], ( 1/(mean_activity_layer2)) ))
    print("Improve grad_s {}: {:.4f}".format(d_DATASET['dataset_title'], (1/(mean_activity_layer1+mean_activity_layer2)) ))

    plot_error(100 * active_counts1 / total_vars_first_hidden, label='First hidden layer', ax=ax, color=colors[0])
    plot_error(100 * active_counts2 / total_vars_second_hidden, label='Second hidden layer', ax=ax, color=colors[1])
    xfmt = tkr.FuncFormatter(numfmt)
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel(r'Number of gradient updates ($\times 10^3$)')
    ax.set_title(d_DATASET['dataset_title'])
    ax.set_ylim(bottom=0)
# B
def plot_gradients(ax, grad, i, j, vmin, vmax, heatmap=None):
    cmap = "YlGnBu" if heatmap is None else heatmap
    sns.heatmap(grad,
                cmap=cmap,
                cbar=False,
                xticklabels=i==1,
                yticklabels=j==0,
                vmin=vmin,
                vmax=vmax,
                ax=ax)
# C
def plot_loss(d_all, ax, nb_neurons=200, colors=None):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors
    def numfmt(x, pos):
        s = '{}'.format(x / 1000.0)
        return s

    n_shd = d_all[2]['hidden_dict'][nb_neurons]
    loss_orig_shd = d_all[2]['loss_orig'][:, n_shd, :]
    loss_s3gd_shd = d_all[2]['loss_s3gd'][:, n_shd, :]

    plot_error(loss_orig_shd, label='Original SHD', ax=ax, color=colors[0])
    plot_error(loss_s3gd_shd, label='Sparse SHD', ax=ax, color=colors[1])
    xfmt = tkr.FuncFormatter(numfmt)
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel(r'Number of gradient updates ($\times 10^3$)')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0)
    ax.set_title('Loss {}'.format(d_all[2]['dataset_title']))
# D
def plot_test_accuracy(d_all, ax, nb_neurons=200, colors=None):

    colors = ['tab:blue', 'tab:orange'] if colors is None else colors

    ## Plot accuracies ##
    x = np.arange(3)  # This would be the number of datasets
    width = 0.35  # the width of the bars
    n_fmnist = d_all[0]['hidden_dict'][nb_neurons]
    n_nmnist = d_all[1]['hidden_dict'][nb_neurons]
    n_shd = d_all[2]['hidden_dict'][nb_neurons]
    test_acc_orig_mean_fmnist = d_all[0]['test_acc_orig_mean'][n_fmnist]
    test_acc_orig_error_fmnist = d_all[0]['test_acc_orig_error'][n_fmnist]
    test_acc_s3gd_mean_fmnist = d_all[0]['test_acc_s3gd_mean'][n_fmnist]
    test_acc_s3gd_error_fmnist = d_all[0]['test_acc_s3gd_error'][n_fmnist]
    test_acc_orig_mean_nmnist = d_all[1]['test_acc_orig_mean'][n_nmnist]
    test_acc_orig_error_nmnist = d_all[1]['test_acc_orig_error'][n_nmnist]
    test_acc_s3gd_mean_nmnist = d_all[1]['test_acc_s3gd_mean'][n_nmnist]
    test_acc_s3gd_error_nmnist = d_all[1]['test_acc_s3gd_error'][n_nmnist]
    test_acc_orig_mean_shd = d_all[2]['test_acc_orig_mean'][n_shd]
    test_acc_orig_error_shd = d_all[2]['test_acc_orig_error'][n_shd]
    test_acc_s3gd_mean_shd = d_all[2]['test_acc_s3gd_mean'][n_shd]
    test_acc_s3gd_error_shd = d_all[2]['test_acc_s3gd_error'][n_shd]

    # Testing
    rects11 = ax.bar(0 - width / 2, 100*test_acc_orig_mean_fmnist, width, yerr=100*test_acc_orig_error_fmnist, color=colors[0], label='Original')
    rects12 = ax.bar(0 + width / 2, 100*test_acc_s3gd_mean_fmnist, width, yerr=100*test_acc_s3gd_error_fmnist, color=colors[1],  label='Sparse')
    rects21 = ax.bar(1 - width / 2, 100*test_acc_orig_mean_nmnist, width, yerr=100*test_acc_orig_error_nmnist, color=colors[0])
    rects22 = ax.bar(1 + width / 2, 100*test_acc_s3gd_mean_nmnist, width, yerr=100*test_acc_s3gd_error_nmnist, color=colors[1])
    rects31 = ax.bar(2 - width / 2, 100*test_acc_orig_mean_shd, width, yerr=100*test_acc_orig_error_shd, color=colors[0])
    rects32 = ax.bar(2 + width / 2, 100*test_acc_s3gd_mean_shd, width, yerr=100*test_acc_s3gd_error_shd, color=colors[1])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([d_all[0]['dataset_title'], d_all[1]['dataset_title'], d_all[2]['dataset_title']])

    ax.legend()
# ####################################################################################### #

# ####################################### Panel 2 ####################################### #
# A1
def plot_speedup(d_ALL, ax, nb_neurons=200, colors=None, log=False):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    def numfmt(x, pos):
        s = '{}'.format(x / 1000.0)
        return s
    def speedupfmt(x, pos):
        s = '{}x'.format(int(x))
        return s

    n_fmnist = d_ALL[0]['hidden_dict'][nb_neurons]
    n_nmnist = d_ALL[1]['hidden_dict'][nb_neurons]
    n_shd = d_ALL[2]['hidden_dict'][nb_neurons]
    bwd_orig_fmnist = d_ALL[0]['bwd_orig'][:, n_fmnist, :]
    bwd_s3gd_fmnist = d_ALL[0]['bwd_s3gd'][:, n_fmnist, :]
    bwd_orig_nmnist = d_ALL[1]['bwd_orig'][:, n_nmnist, :]
    bwd_s3gd_nmnist = d_ALL[1]['bwd_s3gd'][:, n_nmnist, :]
    bwd_orig_shd = d_ALL[2]['bwd_orig'][:, n_shd, :]
    bwd_s3gd_shd = d_ALL[2]['bwd_s3gd'][:, n_shd, :]

    x_fmnist = 100*np.arange(bwd_orig_fmnist.shape[1])/(bwd_orig_fmnist.shape[1]+1)
    x_nmnist = 100*np.arange(bwd_orig_nmnist.shape[1])/(bwd_orig_nmnist.shape[1]+1)
    x_shd = 100*np.arange(bwd_orig_shd.shape[1])/(bwd_orig_shd.shape[1]+1)
    plot_error(bwd_orig_fmnist / (bwd_s3gd_fmnist + 1e-30), label=d_ALL[0]['dataset_title'], color=colors[0], ax=ax, x=x_fmnist)
    plot_error(bwd_orig_nmnist / (bwd_s3gd_nmnist + 1e-30), label=d_ALL[1]['dataset_title'], color=colors[1], ax=ax, x=x_nmnist)
    plot_error(bwd_orig_shd / (bwd_s3gd_shd + 1e-30), label=d_ALL[2]['dataset_title'], color=colors[2], ax=ax, x=x_shd)
    ax.set_xlabel(r'Gradient updates (% of total)')
    ax.set_ylim([1., 100.])
    ax.set_ylabel('Speedup')
    if log:
        ax.set_yscale('log')
        ax.set_yticks([1, 10, 100])
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(speedupfmt))  # add the custom ticks
    else:
        ax.set_yticks([1., 10, 20, 30, 40, 50])
        yfmt = tkr.FuncFormatter(speedupfmt)
        ax.yaxis.set_major_formatter(yfmt)

    ax.set_title('Backward Speedup')
# A2
def plot_mem(d_ALL, ax, nb_neurons=200, colors=None):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    n_fmnist = d_ALL[0]['hidden_dict'][nb_neurons]
    n_nmnist = d_ALL[1]['hidden_dict'][nb_neurons]
    n_shd = d_ALL[2]['hidden_dict'][nb_neurons]
    bwdm_orig_fmnist = d_ALL[0]['bwdm_orig'][:, n_fmnist, :]
    bwdm_s3gd_fmnist = d_ALL[0]['bwdm_s3gd'][:, n_fmnist, :]
    bwdm_orig_nmnist = d_ALL[1]['bwdm_orig'][:, n_nmnist, :]
    bwdm_s3gd_nmnist = d_ALL[1]['bwdm_s3gd'][:, n_nmnist, :]
    bwdm_orig_shd = d_ALL[2]['bwdm_orig'][:, n_shd, :]
    bwdm_s3gd_shd = d_ALL[2]['bwdm_s3gd'][:, n_shd, :]
    
    x_fmnist = 100*np.arange(bwdm_orig_fmnist.shape[1])/(bwdm_orig_fmnist.shape[1]+1)
    x_nmnist = 100*np.arange(bwdm_orig_nmnist.shape[1])/(bwdm_orig_nmnist.shape[1]+1)
    x_shd = 100*np.arange(bwdm_orig_shd.shape[1])/(bwdm_orig_shd.shape[1]+1)
    plot_error(100*(1.-bwdm_s3gd_fmnist / (bwdm_orig_fmnist + 1e-30)), label=d_ALL[0]['dataset_title'], color=colors[0], ax=ax, x=x_fmnist)
    plot_error(100*(1.-bwdm_s3gd_nmnist / (bwdm_orig_nmnist + 1e-30)), label=d_ALL[1]['dataset_title'], color=colors[1], ax=ax, x=x_nmnist)
    plot_error(100*(1.-bwdm_s3gd_shd / (bwdm_orig_shd + 1e-30)), label=d_ALL[2]['dataset_title'], color=colors[2], ax=ax, x=x_shd)
    ax.set_xlabel(r'Gradient updates (% of total)')
    ax.set_ylim([0., 105.])
    ax.set_ylabel('Memory saved (%)')

    ax.set_title('Memory Saved')
    ax.legend()
# B
def plot_time(d_ALL, ax, nb_neurons=200, colors=None):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    n_fmnist = d_ALL[0]['hidden_dict'][nb_neurons]
    n_nmnist = d_ALL[1]['hidden_dict'][nb_neurons]
    n_shd = d_ALL[2]['hidden_dict'][nb_neurons]
    bwd_orig_shd = d_ALL[2]['bwd_orig'][:, n_shd, :]
    bwd_s3gd_shd = d_ALL[2]['bwd_s3gd'][:, n_nmnist, :]

    x_shd = 100 * np.arange(bwd_orig_shd.shape[1]) / (bwd_orig_shd.shape[1] + 1)
    plot_error(bwd_orig_shd, label='Original', color=colors[0], ax=ax, x=x_shd)
    plot_error(bwd_s3gd_shd, label='Sparse', color=colors[1], ax=ax, x=x_shd)
    ax.set_xlabel(r'Gradient updates (% of total)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Backward time {}'.format(d_all[2]['dataset_title']))
    ax.legend()
# C1
def plot_speedup_neurons_bar(d_ALL, ax, j, hidden_list, colors=None, log=False):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    def speedupfmt(x, pos):
        s = '{}x'.format(int(x))
        return s

    x = np.array(hidden_list)
    width = 200*0.25  # the width of the bars
    bwd_orig_tavg_fmnist = d_ALL[0]['bwd_orig_tavg']
    bwd_s3gd_tavg_fmnist = d_ALL[0]['bwd_s3gd_tavg']
    bwd_orig_tavg_nmnist = d_ALL[1]['bwd_orig_tavg']
    bwd_s3gd_tavg_nmnist = d_ALL[1]['bwd_s3gd_tavg']
    bwd_orig_tavg_shd = d_ALL[2]['bwd_orig_tavg']
    bwd_s3gd_tavg_shd = d_ALL[2]['bwd_s3gd_tavg']

    mean = lambda data: np.mean(np.array(data), axis=0)
    error = lambda data: stats.sem(np.array(data), axis=0)

    # Testing
    ax.bar(x - width, mean(bwd_orig_tavg_fmnist/(bwd_s3gd_tavg_fmnist + 1e-30)), width,
           yerr=error(bwd_orig_tavg_fmnist/(bwd_s3gd_tavg_fmnist + 1e-30)), color=colors[0], label=d_ALL[0]['dataset_title'])
    ax.bar(x, mean(bwd_orig_tavg_nmnist/(bwd_s3gd_tavg_nmnist + 1e-30)), width,
           yerr=error(bwd_orig_tavg_nmnist/(bwd_s3gd_tavg_nmnist + 1e-30)), color=colors[1], label=d_ALL[1]['dataset_title'])
    ax.bar(x + width, mean(bwd_orig_tavg_shd/(bwd_s3gd_tavg_shd + 1e-30)), width,
           yerr=error(bwd_orig_tavg_shd/(bwd_s3gd_tavg_shd + 1e-30)), color=colors[2], label=d_ALL[2]['dataset_title'])
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Number neurons in hidden layers')
    ax.set_title('Backward Speedup')
    ax.set_xticks(hidden_list)
    ax.set_xticklabels(hidden_list)
    if log:
        ax.set_ylim([1., 100.])
        ax.set_yscale('log')
        ax.set_yticks([1, 10, 100])
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(speedupfmt))  # add the custom ticks
    else:
        yfmt = tkr.FuncFormatter(speedupfmt)
        ax.yaxis.set_major_formatter(yfmt)
# C2
def plot_overallspeedup_neurons_bar(d_ALL, ax, j, hidden_list, colors=None, log=False):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    def speedupfmt(x, pos):
        s = '{}x'.format(int(x))
        return s

    x = np.array(hidden_list)
    width = 200*0.25  # the width of the bars
    bwd_orig_tavg_fmnist = d_ALL[0]['bwd_orig_tavg']
    bwd_s3gd_tavg_fmnist = d_ALL[0]['bwd_s3gd_tavg']
    bwd_orig_tavg_nmnist = d_ALL[1]['bwd_orig_tavg']
    bwd_s3gd_tavg_nmnist = d_ALL[1]['bwd_s3gd_tavg']
    bwd_orig_tavg_shd = d_ALL[2]['bwd_orig_tavg']
    bwd_s3gd_tavg_shd = d_ALL[2]['bwd_s3gd_tavg']
    fwd_orig_tavg_fmnist = d_ALL[0]['fwd_orig_tavg']
    fwd_s3gd_tavg_fmnist = d_ALL[0]['fwd_s3gd_tavg']
    fwd_orig_tavg_nmnist = d_ALL[1]['fwd_orig_tavg']
    fwd_s3gd_tavg_nmnist = d_ALL[1]['fwd_s3gd_tavg']
    fwd_orig_tavg_shd = d_ALL[2]['fwd_orig_tavg']
    fwd_s3gd_tavg_shd = d_ALL[2]['fwd_s3gd_tavg']

    mean = lambda data: np.mean(np.array(data), axis=0)
    error = lambda data: stats.sem(np.array(data), axis=0)

    # Testing
    ax.bar(x - width, mean((bwd_orig_tavg_fmnist+fwd_orig_tavg_fmnist)/(bwd_s3gd_tavg_fmnist + fwd_s3gd_tavg_fmnist + 1e-30)), width,
           yerr=error((bwd_orig_tavg_fmnist+fwd_orig_tavg_fmnist)/(bwd_s3gd_tavg_fmnist + fwd_s3gd_tavg_fmnist + 1e-30)), color=colors[0], label=d_ALL[0]['dataset_title'])
    ax.bar(x, mean((bwd_orig_tavg_nmnist+fwd_orig_tavg_nmnist)/(bwd_s3gd_tavg_nmnist + fwd_s3gd_tavg_nmnist + 1e-30)), width,
           yerr=error((bwd_orig_tavg_nmnist+fwd_orig_tavg_nmnist)/(bwd_s3gd_tavg_nmnist+ fwd_s3gd_tavg_nmnist + 1e-30)), color=colors[1], label=d_ALL[1]['dataset_title'])
    ax.bar(x + width, mean((bwd_orig_tavg_shd+fwd_orig_tavg_shd)/(bwd_s3gd_tavg_shd+ fwd_s3gd_tavg_shd + 1e-30)), width,
           yerr=error((bwd_orig_tavg_shd+fwd_orig_tavg_shd)/(bwd_s3gd_tavg_shd+ fwd_s3gd_tavg_shd + 1e-30)), color=colors[2], label=d_ALL[2]['dataset_title'])
    # ax.set_ylabel('Speedup')
    ax.set_xlabel('Number neurons in hidden layers')
    ax.set_title('Overall Speedup')
    ax.set_xticks(hidden_list)
    ax.set_xticklabels(hidden_list)
    if log:
        def ticks_format(value, index):
            return ''
        ax.set_ylim([1., 10.])
        ax.set_yscale('log')
        ax.set_yticks([1, 10])
        plt.setp(ax.get_yticklabels(minor=True), visible=False)
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(speedupfmt))  # add the custom ticks
    else:
        yfmt = tkr.FuncFormatter(speedupfmt)
        ax.yaxis.set_major_formatter(yfmt)
    ax.legend()
# C3
def plot_mem_neurons_bar(d_ALL, ax, j, hidden_list, colors=None):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors


    x = np.array(hidden_list)
    width = 200*0.25  # the width of the bars
    bwdm_orig_tavg_fmnist = d_ALL[0]['bwdm_orig_tavg']
    bwdm_s3gd_tavg_fmnist = d_ALL[0]['bwdm_s3gd_tavg']
    bwdm_orig_tavg_nmnist = d_ALL[1]['bwdm_orig_tavg']
    bwdm_s3gd_tavg_nmnist = d_ALL[1]['bwdm_s3gd_tavg']
    bwdm_orig_tavg_shd = d_ALL[2]['bwdm_orig_tavg']
    bwdm_s3gd_tavg_shd = d_ALL[2]['bwdm_s3gd_tavg']

    mean = lambda data: np.mean(np.array(data), axis=0)
    error = lambda data: stats.sem(np.array(data), axis=0)

    # Testing
    ax.bar(x - width, mean(100*(1.-(bwdm_s3gd_tavg_fmnist/(bwdm_orig_tavg_fmnist + 1e-30)))), width,
           yerr=error(100.*(1.-(bwdm_s3gd_tavg_fmnist/(bwdm_orig_tavg_fmnist + 1e-30)))), color=colors[0], label=d_ALL[0]['dataset_title'])
    ax.bar(x, mean(100.*(1.-(bwdm_s3gd_tavg_nmnist/(bwdm_orig_tavg_nmnist + 1e-30)))), width,
           yerr=error(100.*(1.-(bwdm_s3gd_tavg_nmnist/(bwdm_orig_tavg_nmnist + 1e-30)))), color=colors[1], label=d_ALL[1]['dataset_title'])
    ax.bar(x + width, mean(100.*(1.-(bwdm_s3gd_tavg_shd/(bwdm_orig_tavg_shd + 1e-30)))), width,
           yerr=error(100.*(1.-(bwdm_s3gd_tavg_shd/(bwdm_orig_tavg_shd + 1e-30)))), color=colors[2], label=d_ALL[2]['dataset_title'])
    ax.set_ylabel('Memory save (%)')
    ax.set_xlabel('Number neurons in hidden layers')
    ax.set_title('Backward memory saved')
    ax.set_xticks(hidden_list)
    ax.set_xticklabels(hidden_list)

# ####################################################################################### #

# ####################################### Panel 3 ####################################### #

def read_data_bth(path_read):
    d = pickle.load(open(os.path.join(path_read, 'data_bth.p'), 'rb'))

    loss_orig = d['loss_orig']
    fwd_orig = d['fwd_orig']
    bwd_orig = d['bwd_orig']
    fwdm_orig = d['fwdm_orig']
    bwdm_orig = d['bwdm_orig']
    train_acc_orig = d['train_acc_orig']
    test_acc_orig = d['test_acc_orig']

    loss_s3gd = d['loss_s3gd']
    fwd_s3gd = d['fwd_s3gd']
    bwd_s3gd = d['bwd_s3gd']
    fwdm_s3gd = d['fwdm_s3gd']
    bwdm_s3gd = d['bwdm_s3gd']
    train_acc_s3gd = d['train_acc_s3gd']
    test_acc_s3gd = d['test_acc_s3gd']

    # Process data
    loss_orig = np.array(loss_orig)
    fwd_orig = np.array(fwd_orig)
    bwd_orig = np.array(bwd_orig)
    fwdm_orig = np.array(fwdm_orig) / (1024 ** 2)  # MiB
    bwdm_orig = np.array(bwdm_orig) / (1024 ** 2)
    loss_s3gd = np.array(loss_s3gd)
    fwd_s3gd = np.array(fwd_s3gd)
    bwd_s3gd = np.array(bwd_s3gd)
    fwdm_s3gd = np.array(fwdm_s3gd) / (1024 ** 2)
    bwdm_s3gd = np.array(bwdm_s3gd) / (1024 ** 2)

    train_acc_orig_mean = np.mean(train_acc_orig, axis=0)
    train_acc_orig_error = stats.sem(train_acc_orig)[0]
    train_acc_s3gd_mean = np.mean(train_acc_s3gd, axis=0)
    train_acc_s3gd_error = stats.sem(train_acc_s3gd)[0]
    test_acc_orig_mean = np.mean(test_acc_orig, axis=0)
    test_acc_orig_error = stats.sem(test_acc_orig)[0]
    test_acc_s3gd_mean = np.mean(test_acc_s3gd, axis=0)
    test_acc_s3gd_error = stats.sem(test_acc_s3gd)[0]

    loss_orig_tavg = np.mean(loss_orig, axis=2)
    fwd_orig_tavg = np.mean(fwd_orig, axis=2)
    bwd_orig_tavg = np.mean(bwd_orig, axis=2)
    fwdm_orig_tavg = np.mean(fwdm_orig, axis=2)
    bwdm_orig_tavg = np.mean(bwdm_orig, axis=2)
    loss_s3gd_tavg = np.mean(loss_s3gd, axis=2)
    fwd_s3gd_tavg = np.mean(fwd_s3gd, axis=2)
    bwd_s3gd_tavg = np.mean(bwd_s3gd, axis=2)
    fwdm_s3gd_tavg = np.mean(fwdm_s3gd, axis=2)
    bwdm_s3gd_tavg = np.mean(bwdm_s3gd, axis=2)

    total_vars_first_hidden = d['prs_s3gd']['batch_size'] * d['prs_s3gd']['nb_steps'] * d['prs_s3gd']['nb_hidden']
    total_vars_second_hidden = d['prs_s3gd']['batch_size'] * d['prs_s3gd']['nb_steps'] * d['prs_s3gd']['nb_hidden2']
    activity1 = 100*np.mean(np.squeeze(np.array(d['active_counts1']))[:, 50:], axis=1) / total_vars_first_hidden
    activity2 = 100*np.mean(np.squeeze(np.array(d['active_counts2']))[:, 50:], axis=1) / total_vars_second_hidden

    d_bth = {
        'loss_orig': loss_orig,
        'fwd_orig': fwd_orig,
        'bwd_orig': bwd_orig,
        'fwdm_orig': fwdm_orig,
        'bwdm_orig': bwdm_orig,
        'loss_s3gd': loss_s3gd,
        'fwd_s3gd': fwd_s3gd,
        'bwd_s3gd': bwd_s3gd,
        'fwdm_s3gd': fwdm_s3gd,
        'bwdm_s3gd': bwdm_s3gd,

        'train_acc_orig_mean': train_acc_orig_mean,
        'train_acc_orig_error': train_acc_orig_error,
        'train_acc_s3gd_mean': train_acc_s3gd_mean,
        'train_acc_s3gd_error': train_acc_s3gd_error,
        'test_acc_orig_mean': test_acc_orig_mean,
        'test_acc_orig_error': test_acc_orig_error,
        'test_acc_s3gd_mean': test_acc_s3gd_mean,
        'test_acc_s3gd_error': test_acc_s3gd_error,

        'loss_orig_tavg': loss_orig_tavg,
        'fwd_orig_tavg': fwd_orig_tavg,
        'bwd_orig_tavg': bwd_orig_tavg,
        'fwdm_orig_tavg': fwdm_orig_tavg,
        'bwdm_orig_tavg': bwdm_orig_tavg,
        'loss_s3gd_tavg': loss_s3gd_tavg,
        'fwd_s3gd_tavg': fwd_s3gd_tavg,
        'bwd_s3gd_tavg': bwd_s3gd_tavg,
        'fwdm_s3gd_tavg': fwdm_s3gd_tavg,
        'bwdm_s3gd_tavg': bwdm_s3gd_tavg,

        'activity1': activity1,
        'activity2': activity2,

        'dataset_title': 'SHD',
    }
    return d_bth


# A11
def plot_activity_bth(x, d_bth, ax, width):
    activity = (d_bth['activity1']+d_bth['activity2']) / 2.
    rects1 = ax.bar(x[:4], activity[::-1][:4], width, color="tab:orange", label="Learning intact")
    rects1 = ax.bar(x[4:-3], activity[::-1][4:-3], width, color="tab:orange", hatch="///", label="Learning corrupted")
    rects1 = ax.bar(x[-3:], activity[::-1][-3:], width, color="tab:orange", hatch="xxx", label="No Learning")
    ax.set_ylabel('Activity (%)')
    ax.set_xlabel(r"$B_{th}$")
    ax.set_title('{:s} Hidden Activity'.format(dataset_name))
    ax.set_xticks(x[::-1])
    ax.set_xticklabels(bth_list, fontsize=8)

# A12
def plot_accuracy_bth(x, d_bth, ax, width):
    rects1 = ax.bar(x - width / 2, 100*d_bth['test_acc_orig_mean'][::-1], width, label='Original')
    rects3 = ax.bar(x[:4] + width / 2, 100*d_bth['test_acc_s3gd_mean'][::-1][:4], width, color="tab:orange",
                    label="Learning intact")
    rects3 = ax.bar(x[4:-3] + width / 2, 100*d_bth['test_acc_s3gd_mean'][::-1][4:-3], width, color="tab:orange",
                    hatch="///", label="Learning corrupted")
    rects3 = ax.bar(x[-3:] + width / 2, 100*d_bth['test_acc_s3gd_mean'][::-1][-3:], width, color="tab:orange",
                    hatch="xxx", label="No Learning")
    ax.set_ylim([0, 100])
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel(r'$B_{th}$')
    ax.set_title('{:s} Test Accuracy'.format(dataset_name))
    ax.set_xticks(x[::-1])
    ax.set_xticklabels(bth_list, fontsize=8)

# A21
def plot_speedup_bth(x, d_bth, ax, width):
    def speedupfmt(x, pos):  # your custom formatter function: divide by 1000.0
        s = '{}x'.format(int(x))
        return s
    rects1 = ax.bar(x[:4], (np.squeeze(d_bth['bwd_orig_tavg'] / (d_bth['bwd_s3gd_tavg'] + 1e-30)))[::-1][:4], width, color="tab:orange",
                    label="Learning intact")
    rects1 = ax.bar(x[4:-3], (np.squeeze(d_bth['bwd_orig_tavg'] / (d_bth['bwd_s3gd_tavg'] + 1e-30)))[::-1][4:-3], width,
                    color="tab:orange", hatch="///", label="Learning corrupted")
    rects1 = ax.bar(x[-3:], (np.squeeze(d_bth['bwd_orig_tavg'] / (d_bth['bwd_s3gd_tavg'] + 1e-30)))[::-1][-3:], width, color="tab:orange",
                    hatch="xxx", label="No Learning")
    ax.set_xlabel(r"$B_{th}$")
    ax.set_yscale('log')
    ax.set_ylabel("Speedup")
    ax.set_title("{:s} Backward Speedup ".format(dataset_name))
    ax.set_yticks([1, 10, 100, 1000])
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(speedupfmt))  # add the custom ticks
    ax.set_xticks(x)
    ax.set_xticklabels(bth_list[::-1], fontsize=8)

# A22
def plot_mem_bth(x, d_bth, ax, width):
    rects1 = ax.bar(x[:4], (np.squeeze(100 * ((d_bth['bwdm_orig_tavg'] / (d_bth['bwdm_s3gd_tavg'] + 1e-30)) - 1)))[::-1][:4], width,
                    color="tab:orange", label="Learning intact")
    rects1 = ax.bar(x[4:-3], (np.squeeze(100 * ((d_bth['bwdm_orig_tavg'] / (d_bth['bwdm_s3gd_tavg'] + 1e-30)) - 1)))[::-1][4:-3], width,
                    color="tab:orange", hatch="///", label="Learning degraded")
    rects1 = ax.bar(x[-3:], (np.squeeze(100 * ((d_bth['bwdm_orig_tavg'] / (d_bth['bwdm_s3gd_tavg'] + 1e-30)) - 1)))[::-1][-3:], width,
                    color="tab:orange", hatch="xxx", label="No Learning")
    ax.set_xticks(x)
    ax.set_xticklabels(bth_list[::-1], fontsize=8)
    ax.set_xlabel(r"$B_{th}$")
    ax.set_ylabel(' GPU Memory Save (%) ')
    ax.set_title("{:s} Backward Memory Saved ".format(dataset_name))
    ax.set_ylim(bottom=0)

# ####################################################################################### #

if __name__ == "__main__":

    save = False

    hidden_list = [200, 400, 600, 800, 1000]
    nb_trials = 1
    root_path = 'results_data/'
    dirs = ['RUN_FMNIST_FINAL', 'RUN_NMNIST_FINAL', 'RUN_SHD_FINAL']
    dataset_names = ['F-MNIST', 'N-MNIST', 'SHD']
    gpus = ['RTX6000', '1080ti', '1060']
    d_orig_grad = pickle.load(open(os.path.join(root_path, 'orig_grads_shd.p'), 'rb'))
    d_s3gd_grad = pickle.load(open(os.path.join(root_path, 's3gd_grads_shd.p'), 'rb'))

    PATH_READ_FMNIST = os.path.join(root_path, dirs[0])
    PATH_READ_NMNIST = os.path.join(root_path, dirs[1])
    PATH_READ_SHD = os.path.join(root_path, dirs[2])

    d_FMNIST = read_data(nb_trials, hidden_list, PATH_READ_FMNIST, dataset_names[0])
    d_NMNIST = read_data(nb_trials, hidden_list, PATH_READ_NMNIST, dataset_names[1])
    d_SHD = read_data(nb_trials, hidden_list, PATH_READ_SHD, dataset_names[2])

    d_all = [d_FMNIST, d_NMNIST, d_SHD]

    ############################## PANEL 1 #################################
    layout = '''
        AAA
        BBC
        BBD
        '''
    fig = plt.figure(figsize=(12, 8), dpi=150)
    specs, gs = panel_specs(layout, fig=fig)

    # A: activity
    N = 1
    M = 3
    subgs = specs['A'].subgridspec(N, M, wspace=0.15, hspace=0.15)
    triaxes = {}
    for i in range(N):
        for j in range(M):
            triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
            # Plotting activity
            plot_activity(d_all[j], ax=ax, nb_neurons=200, colors=COLORS_LAYERS)
            if j==0:
                ax.set_ylabel('Active neurons (%)')
                ax.legend()
    label_panel(triaxes[0, 0], 'A')


    # B: Weight heatmap
    range_w = 20
    range_s = 20
    grad_w_orig = d_orig_grad['w1_grad'][:range_w, :range_w]
    grad_w_s3gd = d_s3gd_grad['w1_grad'][:range_w, :range_w]
    grad_s_orig = d_orig_grad['spk_rec12_grad'][:range_s, :range_s]
    grad_s_s3gd = d_s3gd_grad['spk_rec12_grad'][:range_s, :range_s]
    vmin_w = np.mean(np.stack((grad_w_orig, grad_w_s3gd)).flatten()) - 1*np.std(np.stack((grad_w_orig, grad_w_s3gd)).flatten())
    vmax_w = np.mean(np.stack((grad_w_orig, grad_w_s3gd)).flatten()) + 1*np.std(np.stack((grad_w_orig, grad_w_s3gd)).flatten())

    vmin_s = min(np.min(grad_s_orig), np.min(grad_s_s3gd))
    vmax_s = max(np.max(grad_s_orig), np.max(grad_s_s3gd))
    max_s = max(abs(vmin_s), abs(vmax_s))
    vmin_s, vmax_s = -max_s, max_s

    N = 2
    M = 3
    subgs = specs['B'].subgridspec(N, M, wspace=0.2, hspace=0.2)
    subgs.set_width_ratios([20, 20, 1])
    triaxes = {}
    for i in range(N):
        for j in range(M):
            triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
            ax.set_facecolor('lightgrey')
            if i==0:
                plt.fill_between([0, range_w], [range_w, range_w], color="none", hatch="////", edgecolor="silver", linewidth=0.)  # Hatch
                if j==0:
                    grad_w_orig[grad_w_orig==0.] = np.NaN
                    hm = plot_gradients(ax, grad_w_orig, i, j, vmin_w, vmax_w, heatmap=HEATMAP)
                    ax.set_title(r'Original $\nabla W^{(0)}$')
                    ax.set_ylabel('Output Neuron index')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                    ax.set_yticklabels(['0', '5', '10', '15', '20'])
                elif j==1:
                    grad_w_s3gd[grad_w_s3gd==0.] = np.NaN
                    hm = plot_gradients(ax, grad_w_s3gd, i, j, vmin_w, vmax_w, heatmap=HEATMAP)
                    ax.set_title(r'Sparse $\nabla W^{(0)}$')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                else:
                    norm = matplotlib.colors.Normalize(vmin=vmin_w, vmax=vmax_w)
                    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=matplotlib.cm.get_cmap(name=HEATMAP), norm=norm, orientation='vertical')
                    cb.ax.tick_params(labelsize=8)
                    cb.ax.yaxis.get_offset_text().set_fontsize(8)
            else:
                plt.fill_between([0, range_s], [range_s, range_s], color="none", hatch="////", edgecolor='silver', linewidth=0.)  # Hatch
                if j==0:
                    grad_s_orig[grad_s_orig==0.] = np.NaN
                    hm = plot_gradients(ax, grad_s_orig, i, j, vmin_s, vmax_s, heatmap=HEATMAP)
                    ax.set_title(r'Original $\nabla S^{(1)}$')
                    ax.set_ylabel('Time index')
                    ax.set_xlabel('Input Neuron index')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                    ax.set_xticklabels(['0', '5', '10', '15', '20'], rotation=0)
                    ax.set_yticklabels(['0', '5', '10', '15', '20'])
                elif j==1:
                    grad_s_s3gd[grad_s_s3gd==0.] = np.NaN
                    hm = plot_gradients(ax, grad_s_s3gd, i, j, vmin_s, vmax_s, heatmap=HEATMAP)
                    ax.set_title(r'Sparse $\nabla S^{(1)}$')
                    ax.set_xlabel('Input Neuron index')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                    ax.set_xticklabels(['0', '5', '10', '15', '20'], rotation=0)
                else:
                    norm = matplotlib.colors.Normalize(vmin=vmin_s, vmax=vmax_s)
                    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=matplotlib.cm.get_cmap(name=HEATMAP), norm=norm, orientation='vertical')
                    cb.ax.tick_params(labelsize=8)
                    cb.ax.yaxis.get_offset_text().set_fontsize(8)
    label_panel(triaxes[0, 0], 'B')

    # C: loss
    ax = fig.add_subplot(specs['C'])
    plot_loss(d_all, ax=ax, nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    ax.legend()
    label_panel(ax, 'C')

    # D: test accuracy
    ax = fig.add_subplot(specs['D'])
    plot_test_accuracy(d_all, ax=ax, nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    ax.legend()
    label_panel(ax, 'D')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join('panel1.pdf'))


    ############################## PANEL 2 #################################
    layout = '''
        AAB
        CCC
        '''
    fig = plt.figure(figsize=(12, 5.3), dpi=150)
    specs, gs = panel_specs(layout, fig=fig)

    # A: performance over time
    N = 1
    M = 2
    subgs = specs['A'].subgridspec(N, M, wspace=0.25, hspace=0.25)
    triaxes = {}
    for i in range(N):
        for j in range(M):
            triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
            # Plotting speedup and mem
            if j==0:
                plot_speedup(d_all, ax, nb_neurons=200, colors=COLORS_DATASET, log=True)
            else:
                plot_mem(d_all, ax, nb_neurons=200, colors=COLORS_DATASET)
    label_panel(triaxes[0, 0], 'A')

    # B: Time SHD
    ax = fig.add_subplot(specs['B'])
    plot_time(d_all, ax=ax, nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    ax.legend()
    label_panel(ax, 'B')

    # C: performance varying hidden
    N = 1
    M = 3
    subgs = specs['C'].subgridspec(N, M, wspace=0.2, hspace=0.2)
    triaxes = {}
    for i in range(N):
        for j in range(M):
            triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
            if j==0:
                # Plotting speedup varying neurons
                plot_speedup_neurons_bar(d_all, ax, j, hidden_list, colors=COLORS_DATASET, log=True)
            elif j==1:
                plot_overallspeedup_neurons_bar(d_all, ax, j, hidden_list, colors=COLORS_DATASET, log=True)
            else:
                plot_mem_neurons_bar(d_all, ax, j, hidden_list, colors=COLORS_DATASET)
    label_panel(triaxes[0, 0], 'C')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join('panel2.pdf'))


    ############################## PANEL 3 #################################
    PATH_RESULTS = "results_bth"

    bth_list = [0.999999, 0.99999, 0.9999, 0.999, 0.99, 0.95, 0.9, 0.8, 0.75]
    nb_hidden = 400
    dataset_name = "SHD"

    d_bth = read_data_bth(PATH_RESULTS)
    x = np.arange(len(bth_list))

    layout = '''
        AC
        BC
        BC
        '''
    fig = plt.figure(figsize=(12, 8), dpi=150)
    specs, gs = panel_specs(layout, fig=fig)

    # A: Accuracy
    N = 1
    M = 1
    ax = fig.add_subplot(specs['A'])
    width = 0.35
    plot_accuracy_bth(x, d_bth, ax, width)
    ax.legend()
    label_panel(ax, 'A')

    # B: Loss
    def numfmt(x, pos):
        s = '{}'.format(x / 1000.0)
        return s
    N = 2
    M = 2
    n_dict = {(0, 0): 1, (0, 1): 3, (1, 0): 4, (1, 1): 5}
    bth_dict = {(0, 0): 0.99999, (0, 1): 0.999, (1, 0): 0.99, (1, 1): 0.95}
    subgs = specs['B'].subgridspec(N, M, wspace=0.15, hspace=0.15)
    triaxes = {}
    for i in range(N):
        for j in range(M):
            triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
            n = n_dict[i, j]
            b_th = bth_dict[i, j]
            plot_error(d_bth['loss_orig'][:, n, :], label='Original', ax=ax)
            plot_error(d_bth['loss_s3gd'][:, n, :], label='Sparse', ax=ax)
            if n == 0:
                ax.set_title(r"{:s} = {:.6f}".format(r"$B_{th}$", b_th))
            elif n == 1:
                ax.set_title(r"{:s} = {:.5f}".format(r"$B_{th}$", b_th))
            elif n == 2:
                ax.set_title(r"{:s} = {:.4f}".format(r"$B_{th}$", b_th))
            elif n == 3:
                ax.set_title(r"{:s} = {:.3f}".format(r"$B_{th}$", b_th))
            else:
                ax.set_title(r"{:s} = {:.2f}".format(r"$B_{th}$", b_th))
            ax.set_xticks([0, 250, 500, 750, 1000])
            if i==0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r'Gradient updates')
            if j==0:
                ax.set_ylabel("Loss")
            else:
                ax.set_yticklabels([])
            if i==0 and j==1:
                ax.legend()
    label_panel(triaxes[0, 0], 'B')


    # A: Activity, Sppedup, Memory
    N = 3
    M = 1
    subgs = specs['C'].subgridspec(N, M, wspace=0., hspace=0.15)
    triaxes = {}
    for i in range(N):
        for j in range(M):
            triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
            width = 0.5
            if i==0:
                plot_activity_bth(x, d_bth, ax, width)
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.legend()
            elif i==1:
                plot_speedup_bth(x, d_bth, ax, width)
                ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                plot_mem_bth(x, d_bth, ax, width)
    label_panel(triaxes[0, 0], 'C')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join('panel3.pdf'))


    plt.show()



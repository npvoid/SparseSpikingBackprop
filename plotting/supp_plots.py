import os
from plot_utils import plot_error, plt_set, panel_specs, tight_xticklabels, label_panel
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pickle
import numpy as np
from scipy import stats
import seaborn as sns

from matplotlib.ticker import ScalarFormatter
import matplotlib.style as style

style.use('tableau-colorblind10')

from matplotlib import cm

tab_colors = cm.get_cmap('tab20', 20).colors
paired_colors = cm.get_cmap('Paired', 12).colors

COLORS_LAYERS = [tab_colors[0], tab_colors[2]]  # plt.rcParams['axes.prop_cycle'].by_key()['color']
COLORS_ORIG_SPARSE = ['grey', tab_colors[2]]  # plt.rcParams['axes.prop_cycle'].by_key()['color']
HEATMAP = "YlGnBu"  # "YlGnBu"
COLORS_DATASET = ['#000000', '#1F77B4', '#FF7F0E']  # plt.rcParams['axes.prop_cycle'].by_key()['color']

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


def supp_read_data(nb_trials, hidden_list, path_read, dataset_name):
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

                trial = d['prs_orig']['seed']
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

                trial = d['prs_s3gd']['seed']

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



# Accuracies all num neurons all datasets
def supp_plot_test_accuracy(d_dataset, ax, nb_neurons=None, colors=None, ylabel=False, legend=False):
    # One plot for the given dataset
    colors = ['tab:blue', 'tab:orange'] if colors is None else colors
    width = 0.35  # the width of the bars

    ## Plot accuracies ##
    for x, nb_neuron in enumerate(nb_neurons):
        n = d_dataset['hidden_dict'][nb_neuron]
        test_orig_mean = d_dataset['test_acc_orig_mean'][n]
        test_orig_error = d_dataset['test_acc_orig_error'][n]
        test_sparse_mean = d_dataset['test_acc_s3gd_mean'][n]
        test_sparse_error = d_dataset['test_acc_s3gd_error'][n]
        rects11 = ax.bar(x - width / 2, 100 * test_orig_mean, width, yerr=test_orig_error,
                         color=colors[0], label='Original' if (legend and x==0) else '')
        rects12 = ax.bar(x + width / 2, 100 * test_sparse_mean, width, yerr=test_sparse_error,
                         color=colors[1], label='Sparse' if (legend and x==0) else '')
    if ylabel:
        ax.set_ylabel('Accuracy (%)')
    ax.set_ylim([0, 100])
    ax.set_title(d_dataset['dataset_title']+' Test Accuracy')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(nb_neurons)
    ax.set_xlabel('Number neurons in hidden layers')
    if legend:
        ax.legend()

# Losses all datasets 200 neurons  (all num neurons?)
def supp_plot_loss(d_dataset, ax, nb_neurons=200, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    def numfmt(x, pos):
        s = '{}'.format(x / 1000.0)
        return s

    n = d_dataset['hidden_dict'][nb_neurons]
    loss_orig = d_dataset['loss_orig'][:, n, :]
    loss_s3gd = d_dataset['loss_s3gd'][:, n, :]
    plot_error(loss_orig, label='Original', ax=ax, color=colors[0])
    plot_error(loss_s3gd, label='Sparse', ax=ax, color=colors[1])
    ax.set_ylim(bottom=0)
    ax.set_title('Loss {}'.format(d_dataset['dataset_title']))
    xfmt = tkr.FuncFormatter(numfmt)
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel(r'Number of gradient updates ($\times 10^3$)')
    if ylabel:
        ax.set_ylabel('Loss')
    if legend:
        ax.legend()


# Activities as we increase number neurons (average)
def supp_plot_activity(d_dataset, ax, nb_neurons=None, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    for x, nb_neuron in enumerate(nb_neurons):
        n = d_dataset['hidden_dict'][nb_neuron]
        total_vars_first_hidden = d_dataset['batch_size'] * d_dataset['nb_steps'] * nb_neuron
        total_vars_second_hidden = d_dataset['batch_size'] * d_dataset['nb_steps'] * nb_neuron
        active_counts1 = d_dataset['active_counts1'][:, n, :]
        active_counts2 = d_dataset['active_counts2'][:, n, :]
        mean_activity_layer1 = active_counts1.mean() / total_vars_first_hidden
        mean_activity_layer2 = active_counts2.mean() / total_vars_second_hidden
        error_activity_layer1 = stats.sem(active_counts1.mean(1)/ total_vars_first_hidden)
        error_activity_layer2 = stats.sem(active_counts2.mean(1)/ total_vars_second_hidden)

        # Activity
        width = 0.35  # the width of the bars
        rects11 = ax.bar(x - width / 2, 100*mean_activity_layer1, width, yerr=100*error_activity_layer1,
                         color=colors[0], label='First Hidden Layer' if (legend and x==0) else '')
        rects12 = ax.bar(x + width / 2, 100*mean_activity_layer2, width, yerr=100*error_activity_layer2,
                         color=colors[1], label='Second Hidden Layer' if (legend and x==0) else '')

    ax.set_ylim([0, 2])
    if ylabel:
        ax.set_ylabel('Active neurons (%)')
    ax.set_title('{} Activity'.format(d_dataset['dataset_title']))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(nb_neurons)
    ax.set_xlabel('Number neurons in hidden layers')
    if legend:
        ax.legend(loc='upper right')


# Forward times 200
def supp_forward_times(d_dataset, ax, nb_neurons=200, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    n = d_dataset['hidden_dict'][nb_neurons]
    fwd_orig = d_dataset['fwd_orig'][:, n, :]
    fwd_s3gd = d_dataset['fwd_s3gd'][:, n, :]
    x = 100 * np.arange(fwd_orig.shape[1]) / (fwd_orig.shape[1] + 1)
    plot_error(fwd_orig, label='Original', color=colors[0], ax=ax, x=x)
    plot_error(fwd_s3gd, label='Sparse', color=colors[1], ax=ax, x=x)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(r'Gradient updates (% of total)')
    if ylabel:
        ax.set_ylabel('Time (s)')
    ax.set_title('Forward time {}'.format(d_dataset['dataset_title']))
    if legend:
        ax.legend()

# Forward times all neurons average
def supp_forward_times_all(d_dataset, ax, nb_neurons=None, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    for x, nb_neuron in enumerate(nb_neurons):
        n = d_dataset['hidden_dict'][nb_neuron]
        fwd_orig = d_dataset['fwd_orig'][:, n, :]
        fwd_s3gd = d_dataset['fwd_s3gd'][:, n, :]
        mean_fwd_orig = fwd_orig.mean()
        mean_fwd_s3gd = fwd_s3gd.mean()
        error_fwd_orig = stats.sem(fwd_orig.mean(1))
        error_fwd_s3gd = stats.sem(fwd_s3gd.mean(1))

        # Activity
        width = 0.35  # the width of the bars
        rects11 = ax.bar(x - width / 2, mean_fwd_orig, width, yerr=error_fwd_orig,
                         color=colors[0], label='Original' if (legend and x==0) else '')
        rects12 = ax.bar(x + width / 2, mean_fwd_s3gd, width, yerr=error_fwd_s3gd,
                         color=colors[1], label='Sparse' if (legend and x==0) else '')

    ax.set_ylim([0, 0.4])
    if ylabel:
        ax.set_ylabel('Time (s)')
    ax.set_title('{} Forward Time'.format(d_dataset['dataset_title']))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(nb_neurons)
    ax.set_xlabel('Number neurons in hidden layers')
    if legend:
        ax.legend(loc='upper right')


# Backward times 200
def supp_backward_times(d_dataset, ax, nb_neurons=200, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    n = d_dataset['hidden_dict'][nb_neurons]
    bwd_orig = d_dataset['bwd_orig'][:, n, :]
    bwd_s3gd = d_dataset['bwd_s3gd'][:, n, :]
    x = 100 * np.arange(bwd_orig.shape[1]) / (bwd_orig.shape[1] + 1)
    plot_error(bwd_orig, label='Original', color=colors[0], ax=ax, x=x)
    plot_error(bwd_s3gd, label='Sparse', color=colors[1], ax=ax, x=x)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(r'Gradient updates (% of total)')
    # ax.set_ylim([1., 100.])
    if ylabel:
        ax.set_ylabel('Time (s)')
    ax.set_title('Bacward time {}'.format(d_dataset['dataset_title']))
    if legend:
        ax.legend()
        
# Backward times all neurons average
def supp_backward_times_all(d_dataset, ax, nb_neurons=None, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    for x, nb_neuron in enumerate(nb_neurons):
        n = d_dataset['hidden_dict'][nb_neuron]
        bwd_orig = d_dataset['bwd_orig'][:, n, :]
        bwd_s3gd = d_dataset['bwd_s3gd'][:, n, :]
        mean_bwd_orig = bwd_orig.mean()
        mean_bwd_s3gd = bwd_s3gd.mean()
        error_bwd_orig = stats.sem(bwd_orig.mean(1))
        error_bwd_s3gd = stats.sem(bwd_s3gd.mean(1))

        # Activity
        width = 0.35  # the width of the bars
        rects11 = ax.bar(x - width / 2, mean_bwd_orig, width, yerr=error_bwd_orig,
                         color=colors[0], label='Original' if (legend and x==0) else '')
        rects12 = ax.bar(x + width / 2, mean_bwd_s3gd, width, yerr=error_bwd_s3gd,
                         color=colors[1], label='Sparse' if (legend and x==0) else '')

    # ax.set_ylim([0, 1.5])
    ax.set_ylim(bottom=0.)
    if ylabel:
        ax.set_ylabel('Time (s)')
    ax.set_title('{} Backward Time'.format(d_dataset['dataset_title']))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(nb_neurons)
    ax.set_xlabel('Number neurons in hidden layers')
    if legend:
        ax.legend(loc='upper left')


# Memory forward 200
def supp_forward_mem(d_dataset, ax, nb_neurons=200, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    n = d_dataset['hidden_dict'][nb_neurons]
    fwdm_orig = d_dataset['fwdm_orig'][:, n, :]
    fwdm_s3gd = d_dataset['fwdm_s3gd'][:, n, :]
    x = 100 * np.arange(fwdm_orig.shape[1]) / (fwdm_orig.shape[1] + 1)
    plot_error(fwdm_orig, label='Original', color=colors[0], ax=ax, x=x)
    plot_error(fwdm_s3gd, label='Sparse', color=colors[1], ax=ax, x=x)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(r'Gradient updates (% of total)')
    ax.set_ylim([0., 1.15*max(np.max(fwdm_orig), np.max(fwdm_s3gd))])
    if ylabel:
        ax.set_ylabel('Memory (MiB)')
    ax.set_title('Forward memory {}'.format(d_dataset['dataset_title']))
    if legend:
        ax.legend()

# Memory forward all neurons
def supp_forward_mem_all(d_dataset, ax, nb_neurons=None, colors=None, ylabel=False, legend=False):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    for x, nb_neuron in enumerate(nb_neurons):
        n = d_dataset['hidden_dict'][nb_neuron]
        fwdm_orig = d_dataset['fwdm_orig'][:, n, :]
        fwdm_s3gd = d_dataset['fwdm_s3gd'][:, n, :]
        mean_fwdm_orig = fwdm_orig.mean()
        mean_fwdm_s3gd = fwdm_s3gd.mean()
        error_fwdm_orig = stats.sem(fwdm_orig.mean(1))
        error_fwdm_s3gd = stats.sem(fwdm_s3gd.mean(1))

        # Activity
        width = 0.35  # the width of the bars
        rects11 = ax.bar(x - width / 2, mean_fwdm_orig, width, yerr=error_fwdm_orig,
                         color=colors[0], label='Original' if (legend and x==0) else '')
        rects12 = ax.bar(x + width / 2, mean_fwdm_s3gd, width, yerr=error_fwdm_s3gd,
                         color=colors[1], label='Sparse' if (legend and x==0) else '')

    # ax.set_ylim([0, 1.5])
    ax.set_ylim(bottom=0.)
    if ylabel:
        ax.set_ylabel('Memory (MiB)')
    ax.set_title('Forward memory {}'.format(d_dataset['dataset_title']))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(nb_neurons)
    ax.set_xlabel('Number neurons in hidden layers')
    if legend:
        ax.legend(loc='upper left')

# Memory backward 200
def supp_backward_mem(d_dataset, ax, nb_neurons=200, colors=None, ylabel=False, legend=False):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    n = d_dataset['hidden_dict'][nb_neurons]
    bwdm_orig = d_dataset['bwdm_orig'][:, n, :]
    bwdm_s3gd = d_dataset['bwdm_s3gd'][:, n, :]
    x = 100 * np.arange(bwdm_orig.shape[1]) / (bwdm_orig.shape[1] + 1)
    plot_error(bwdm_orig, label='Original', color=colors[0], ax=ax, x=x)
    plot_error(bwdm_s3gd, label='Sparse', color=colors[1], ax=ax, x=x)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(r'Gradient updates (% of total)')
    ax.set_ylim([0., 1.15*max(np.max(bwdm_orig), np.max(bwdm_s3gd))])
    if ylabel:
        ax.set_ylabel('Memory (MiB)')
    ax.set_title('Backward memory {}'.format(d_dataset['dataset_title']))
    if legend:
        ax.legend()

# Memory backward all neurons
def supp_backward_mem_all(d_dataset, ax, nb_neurons=None, colors=None, ylabel=False, legend=False):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    for x, nb_neuron in enumerate(nb_neurons):
        n = d_dataset['hidden_dict'][nb_neuron]
        bwdm_orig = d_dataset['bwdm_orig'][:, n, :]
        bwdm_s3gd = d_dataset['bwdm_s3gd'][:, n, :]
        mean_bwdm_orig = bwdm_orig.mean()
        mean_bwdm_s3gd = bwdm_s3gd.mean()
        error_bwdm_orig = stats.sem(bwdm_orig.mean(1))
        error_bwdm_s3gd = stats.sem(bwdm_s3gd.mean(1))

        # Activity
        width = 0.35  # the width of the bars
        rects11 = ax.bar(x - width / 2, mean_bwdm_orig, width, yerr=error_bwdm_orig,
                         color=colors[0], label='Original' if (legend and x==0) else '')
        rects12 = ax.bar(x + width / 2, mean_bwdm_s3gd, width, yerr=error_bwdm_s3gd,
                         color=colors[1], label='Sparse' if (legend and x==0) else '')

    # ax.set_ylim([0, 1.5])
    ax.set_ylim(bottom=0.)
    if ylabel:
        ax.set_ylabel('Memory (MiB)')
    ax.set_title('Backward memory {}'.format(d_dataset['dataset_title']))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(nb_neurons)
    ax.set_xlabel('Number neurons in hidden layers')
    if legend:
        ax.legend(loc='upper left')

# Memory saved forward all
def supp_mem_saved_all(d_dataset, ax, nb_neurons=None, colors=None, ylabel=False, legend=False):

    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    for x, nb_neuron in enumerate(nb_neurons):
        n = d_dataset['hidden_dict'][nb_neuron]
        fwdm_orig = d_dataset['fwdm_orig'][:, n, :]
        fwdm_s3gd = d_dataset['fwdm_s3gd'][:, n, :]
        saved_fwdm = 100 * (1. - (fwdm_s3gd / (fwdm_orig + 1e-30)))
        mean_saved_fwdm = saved_fwdm.mean()
        error_fwdm_orig = stats.sem(saved_fwdm.mean(1))

        bwdm_orig = d_dataset['bwdm_orig'][:, n, :]
        bwdm_s3gd = d_dataset['bwdm_s3gd'][:, n, :]
        saved_bwdm = 100 * (1. - (bwdm_s3gd / (bwdm_orig + 1e-30)))
        mean_saved_bwdm = saved_bwdm.mean()
        error_bwdm_orig = stats.sem(saved_bwdm.mean(1))

        # Activity
        width = 0.35  # the width of the bars
        rects11 = ax.bar(x - width / 2, mean_saved_fwdm, width, yerr=error_fwdm_orig,
                         color=colors[0], label='Forward' if (legend and x==0) else '')
        rects11 = ax.bar(x + width / 2, mean_saved_bwdm, width, yerr=error_bwdm_orig,
                         color=colors[1], label='Backward' if (legend and x==0) else '')

    # ax.set_ylim([0, 1.5])
    ax.set_ylim(bottom=0.)
    if ylabel:
        ax.set_ylabel('Memory saved (%)')
    ax.set_title('Saved memory {}'.format(d_dataset['dataset_title']))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(nb_neurons)
    ax.set_xlabel('Number neurons in hidden layers')
    if legend:
        ax.legend(loc='upper left')

# Other GPUS
def supp_plot_speedup_neurons_bar(d_ALL, ax, hidden_lists, colors=None, log=False, gpu=''):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    def speedupfmt(x, pos):
        s = '{}x'.format(int(x))
        return s

    x_fmnist = np.array(hidden_lists[0])
    x_nmnist = np.array(hidden_lists[1])
    x_shd = np.array(hidden_lists[2])
    width = 200 * 0.25  # the width of the bars
    bwd_orig_tavg_fmnist = d_ALL[0]['bwd_orig_tavg']
    bwd_s3gd_tavg_fmnist = d_ALL[0]['bwd_s3gd_tavg']
    bwd_orig_tavg_nmnist = d_ALL[1]['bwd_orig_tavg']
    bwd_s3gd_tavg_nmnist = d_ALL[1]['bwd_s3gd_tavg']
    bwd_orig_tavg_shd = d_ALL[2]['bwd_orig_tavg']
    bwd_s3gd_tavg_shd = d_ALL[2]['bwd_s3gd_tavg']

    mean = lambda data: np.mean(np.array(data), axis=0)
    error = lambda data: stats.sem(np.array(data), axis=0)

    # Testing
    ax.bar(x_fmnist - width, mean(bwd_orig_tavg_fmnist / (bwd_s3gd_tavg_fmnist + 1e-30)), width,
           yerr=error(bwd_orig_tavg_fmnist / (bwd_s3gd_tavg_fmnist + 1e-30)), color=colors[0],
           label=d_ALL[0]['dataset_title'])
    ax.bar(x_nmnist, mean(bwd_orig_tavg_nmnist / (bwd_s3gd_tavg_nmnist + 1e-30)), width,
           yerr=error(bwd_orig_tavg_nmnist / (bwd_s3gd_tavg_nmnist + 1e-30)), color=colors[1],
           label=d_ALL[1]['dataset_title'])
    ax.bar(x_shd + width, mean(bwd_orig_tavg_shd / (bwd_s3gd_tavg_shd + 1e-30)), width,
           yerr=error(bwd_orig_tavg_shd / (bwd_s3gd_tavg_shd + 1e-30)), color=colors[2],
           label=d_ALL[2]['dataset_title'])
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Number neurons in hidden layers')
    ax.set_title('Backward Speedup (GPU: {})'.format(gpu))
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
    ax.legend()

def supp_plot_overallspeedup_neurons_bar(d_ALL, ax, hidden_lists, colors=None, log=False, gpu=''):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    def speedupfmt(x, pos):
        s = '{}x'.format(int(x))
        return s

    x_fmnist = np.array(hidden_lists[0])
    x_nmnist = np.array(hidden_lists[1])
    x_shd = np.array(hidden_lists[2])
    width = 200 * 0.25  # the width of the bars
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
    ax.bar(x_fmnist - width,
           mean((bwd_orig_tavg_fmnist + fwd_orig_tavg_fmnist) / (bwd_s3gd_tavg_fmnist + fwd_s3gd_tavg_fmnist + 1e-30)),
           width,
           yerr=error(
               (bwd_orig_tavg_fmnist + fwd_orig_tavg_fmnist) / (bwd_s3gd_tavg_fmnist + fwd_s3gd_tavg_fmnist + 1e-30)),
           color=colors[0], label=d_ALL[0]['dataset_title'])
    ax.bar(x_nmnist,
           mean((bwd_orig_tavg_nmnist + fwd_orig_tavg_nmnist) / (bwd_s3gd_tavg_nmnist + fwd_s3gd_tavg_nmnist + 1e-30)),
           width,
           yerr=error(
               (bwd_orig_tavg_nmnist + fwd_orig_tavg_nmnist) / (bwd_s3gd_tavg_nmnist + fwd_s3gd_tavg_nmnist + 1e-30)),
           color=colors[1], label=d_ALL[1]['dataset_title'])
    ax.bar(x_shd + width, mean((bwd_orig_tavg_shd + fwd_orig_tavg_shd) / (bwd_s3gd_tavg_shd + fwd_s3gd_tavg_shd + 1e-30)),
           width,
           yerr=error((bwd_orig_tavg_shd + fwd_orig_tavg_shd) / (bwd_s3gd_tavg_shd + fwd_s3gd_tavg_shd + 1e-30)),
           color=colors[2], label=d_ALL[2]['dataset_title'])
    # ax.set_ylabel('Speedup')
    ax.set_xlabel('Number neurons in hidden layers')
    ax.set_title('Overall Speedup (GPU: {})'.format(gpu))
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

def supp_plot_mem_neurons_bar(d_ALL, ax, hidden_lists, colors=None, gpu=''):
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if colors is None else colors

    x_fmnist = np.array(hidden_lists[0])
    x_nmnist = np.array(hidden_lists[1])
    x_shd = np.array(hidden_lists[2])
    width = 200 * 0.25  # the width of the bars
    bwdm_orig_tavg_fmnist = d_ALL[0]['bwdm_orig_tavg']
    bwdm_s3gd_tavg_fmnist = d_ALL[0]['bwdm_s3gd_tavg']
    bwdm_orig_tavg_nmnist = d_ALL[1]['bwdm_orig_tavg']
    bwdm_s3gd_tavg_nmnist = d_ALL[1]['bwdm_s3gd_tavg']
    bwdm_orig_tavg_shd = d_ALL[2]['bwdm_orig_tavg']
    bwdm_s3gd_tavg_shd = d_ALL[2]['bwdm_s3gd_tavg']

    mean = lambda data: np.mean(np.array(data), axis=0)
    error = lambda data: stats.sem(np.array(data), axis=0)

    # Testing
    ax.bar(x_fmnist - width, mean(100 * (1. - (bwdm_s3gd_tavg_fmnist / (bwdm_orig_tavg_fmnist + 1e-30)))), width,
           yerr=error(100. * (1. - (bwdm_s3gd_tavg_fmnist / (bwdm_orig_tavg_fmnist + 1e-30)))), color=colors[0],
           label=d_ALL[0]['dataset_title'])
    ax.bar(x_nmnist, mean(100. * (1. - (bwdm_s3gd_tavg_nmnist / (bwdm_orig_tavg_nmnist + 1e-30)))), width,
           yerr=error(100. * (1. - (bwdm_s3gd_tavg_nmnist / (bwdm_orig_tavg_nmnist + 1e-30)))), color=colors[1],
           label=d_ALL[1]['dataset_title'])
    ax.bar(x_shd + width, mean(100. * (1. - (bwdm_s3gd_tavg_shd / (bwdm_orig_tavg_shd + 1e-30)))), width,
           yerr=error(100. * (1. - (bwdm_s3gd_tavg_shd / (bwdm_orig_tavg_shd + 1e-30)))), color=colors[2],
           label=d_ALL[2]['dataset_title'])
    ax.set_ylabel('Memory save (%)')
    ax.set_xlabel('Number neurons in hidden layers')
    ax.set_title('Backward memory saved (GPU: {})'.format(gpu))
    ax.set_xticks(hidden_list)
    ax.set_xticklabels(hidden_list)



if __name__ == "__main__":

    # save = True
    save = False

    hidden_list = [200, 400, 600, 800, 1000]
    nb_trials = 1
    root_path = 'results_data/'
    dirs = ['RUN_FMNIST_FINAL', 'RUN_NMNIST_FINAL', 'RUN_SHD_FINAL']
    dataset_names = ['F-MNIST', 'N-MNIST', 'SHD']
    gpus = ['RTX6000', '1080ti', '1060']

    PATH_SAVE = 'supplmentary_extra_plots'

    PATH_READ_FMNIST = os.path.join(root_path, dirs[0])
    PATH_READ_NMNIST = os.path.join(root_path, dirs[1])
    PATH_READ_SHD = os.path.join(root_path, dirs[2])

    d_FMNIST = read_data(nb_trials, hidden_list, PATH_READ_FMNIST, dataset_names[0])
    d_NMNIST = read_data(nb_trials, hidden_list, PATH_READ_NMNIST, dataset_names[1])
    d_SHD = read_data(nb_trials, hidden_list, PATH_READ_SHD, dataset_names[2])

    d_all = [d_FMNIST, d_NMNIST, d_SHD]


    # Accuracies
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_plot_test_accuracy(d_FMNIST, axs[0], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, ylabel=True)
    supp_plot_test_accuracy(d_NMNIST, axs[1], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, )
    supp_plot_test_accuracy(d_SHD, axs[2], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, legend=True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_plot_test_accuracy.pdf'))
    # Loss
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_plot_loss(d_FMNIST, axs[0], nb_neurons=200, colors=COLORS_ORIG_SPARSE, ylabel=True)
    supp_plot_loss(d_NMNIST, axs[1], nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    supp_plot_loss(d_SHD, axs[2], nb_neurons=200, colors=COLORS_ORIG_SPARSE, legend=True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_plot_loss.pdf'))
    # Activity
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_plot_activity(d_FMNIST, axs[0], nb_neurons=hidden_list, colors=COLORS_LAYERS, ylabel=True, legend=True)
    supp_plot_activity(d_NMNIST, axs[1], nb_neurons=hidden_list, colors=COLORS_LAYERS)
    supp_plot_activity(d_SHD, axs[2], nb_neurons=hidden_list, colors=COLORS_LAYERS)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_plot_activity.pdf'))
    # Forward 200
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_forward_times(d_FMNIST, axs[0], nb_neurons=200, colors=COLORS_ORIG_SPARSE, ylabel=True)
    supp_forward_times(d_NMNIST, axs[1], nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    supp_forward_times(d_SHD, axs[2], nb_neurons=200, colors=COLORS_ORIG_SPARSE, legend=True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_forward_times.pdf'))
    # Forward all
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_forward_times_all(d_FMNIST, axs[0], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, ylabel=True, legend=True)
    supp_forward_times_all(d_NMNIST, axs[1], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    supp_forward_times_all(d_SHD, axs[2], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_forward_times_all.pdf'))
    # Backward 200
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_backward_times(d_FMNIST, axs[0], nb_neurons=200, colors=COLORS_ORIG_SPARSE, ylabel=True)
    supp_backward_times(d_NMNIST, axs[1], nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    supp_backward_times(d_SHD, axs[2], nb_neurons=200, colors=COLORS_ORIG_SPARSE, legend=True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_backward_times.pdf'))
    # Backward all
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_backward_times_all(d_FMNIST, axs[0], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, ylabel=True, legend=True)
    supp_backward_times_all(d_NMNIST, axs[1], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    supp_backward_times_all(d_SHD, axs[2], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_backward_times_all.pdf'))
    # Forward memory 200
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_forward_mem(d_FMNIST, axs[0], nb_neurons=200, colors=COLORS_ORIG_SPARSE, ylabel=True, legend=True)
    supp_forward_mem(d_NMNIST, axs[1], nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    supp_forward_mem(d_SHD, axs[2], nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_forward_mem.pdf'))
    # Forward memory 200
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_forward_mem_all(d_FMNIST, axs[0], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, ylabel=True, legend=True)
    supp_forward_mem_all(d_NMNIST, axs[1], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    supp_forward_mem_all(d_SHD, axs[2], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_forward_mem_all.pdf'))
    # Forward memory 200
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_backward_mem(d_FMNIST, axs[0], nb_neurons=200, colors=COLORS_ORIG_SPARSE, ylabel=True, legend=True)
    supp_backward_mem(d_NMNIST, axs[1], nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    supp_backward_mem(d_SHD, axs[2], nb_neurons=200, colors=COLORS_ORIG_SPARSE)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_backward_mem.pdf'))
    # Forward memory 200
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_backward_mem_all(d_FMNIST, axs[0], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, ylabel=True, legend=True)
    supp_backward_mem_all(d_NMNIST, axs[1], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    supp_backward_mem_all(d_SHD, axs[2], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_backward_mem_all.pdf'))
    # Saved memory forward all
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_mem_saved_all(d_FMNIST, axs[0], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE, ylabel=True, legend=True)
    supp_mem_saved_all(d_NMNIST, axs[1], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    supp_mem_saved_all(d_SHD, axs[2], nb_neurons=hidden_list, colors=COLORS_ORIG_SPARSE)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, 'supp_mem_saved_all.pdf'))


    # Other GPUs
    dirs = ['fmnist', 'nmnist', 'SHD']

    # 1060
    root_path = 'results_data/1060'
    hidden_lists = [[200, 400, 600, 800, 1000], [200, 400], [200]]
    PATH_READ_FMNIST_1060 = os.path.join(root_path, dirs[0])
    PATH_READ_NMNIST_1060 = os.path.join(root_path, dirs[1])
    PATH_READ_SHD_1060 = os.path.join(root_path, dirs[2])
    d_FMNIST_1060 = supp_read_data(1, hidden_lists[0], PATH_READ_FMNIST_1060, dataset_names[0])
    d_NMNIST_1060 = supp_read_data(1, hidden_lists[1], PATH_READ_NMNIST_1060, dataset_names[1])
    d_SHD_1060 = supp_read_data(1, hidden_lists[2], PATH_READ_SHD_1060, dataset_names[2])
    d_all_1060 = [d_FMNIST_1060, d_NMNIST_1060, d_SHD_1060]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_plot_speedup_neurons_bar(d_all_1060, axs[0], hidden_lists, colors=COLORS_DATASET, log=True, gpu='1060')
    supp_plot_overallspeedup_neurons_bar(d_all_1060, axs[1], hidden_lists, colors=COLORS_DATASET, log=True, gpu='1060')
    supp_plot_mem_neurons_bar(d_all_1060, axs[2], hidden_lists, colors=COLORS_DATASET, gpu='1060')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, '1060.pdf'))


    root_path = 'results_data/1080ti'
    hidden_lists = [[200, 400, 600, 800, 1000], [200, 400], [200, 400]]
    PATH_READ_FMNIST_1080ti = os.path.join(root_path, dirs[0])
    PATH_READ_NMNIST_1080ti = os.path.join(root_path, dirs[1])
    PATH_READ_SHD_1080ti = os.path.join(root_path, dirs[2])
    d_FMNIST_1080ti = supp_read_data(1, hidden_lists[0], PATH_READ_FMNIST_1080ti, dataset_names[0])
    d_NMNIST_1080ti = supp_read_data(1, hidden_lists[1], PATH_READ_NMNIST_1080ti, dataset_names[1])
    d_SHD_1080ti = supp_read_data(1, hidden_lists[2], PATH_READ_SHD_1080ti, dataset_names[2])
    d_all_1080ti = [d_FMNIST_1080ti, d_NMNIST_1080ti, d_SHD_1080ti]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    supp_plot_speedup_neurons_bar(d_all_1080ti, axs[0], hidden_lists, colors=COLORS_DATASET, log=True, gpu='1080ti')
    supp_plot_overallspeedup_neurons_bar(d_all_1080ti, axs[1], hidden_lists, colors=COLORS_DATASET, log=True, gpu='1080ti')
    supp_plot_mem_neurons_bar(d_all_1080ti, axs[2], hidden_lists, colors=COLORS_DATASET, gpu='1080ti')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PATH_SAVE, '1080ti.pdf'))

    print('DONE')


plt.show()







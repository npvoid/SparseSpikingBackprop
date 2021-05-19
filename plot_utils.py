import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import transforms


def plt_set(mult=2):
    # mult = 2
    SMALL_SIZE = 8 * mult
    MEDIUM_SIZE = 10 * mult * 1.6
    BIGGER_SIZE = 12 * mult * 1.3
    plt.rcParams['lines.linewidth'] = 3
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=int(SMALL_SIZE * 1.8))  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_error(data, ma=False, label='', x=None, ax=None, **kwargs):
    if ma:
        N = 30
        ma = lambda x: np.convolve(x, np.ones((N,)) / N, mode='valid')
        data_mean = ma(np.mean(np.array(data), axis=0))
    else:
        data_mean = np.mean(np.array(data), axis=0)
    error_bars = stats.sem(np.array(data))[:data_mean.size]
    x = [i for i in range(data_mean.size)] if x is None else x
    if ax is not None:
        ax.plot(x, data_mean, label=label, **kwargs)
        ax.fill_between(x,
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=0.3, **kwargs)
    else:
        plt.plot(x, data_mean, label=label, **kwargs)
        plt.fill_between(x,
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=0.3, **kwargs)


def panel_specs(layout, fig=None):
    # default arguments
    if fig is None:
        fig = plt.gcf()
    # format and sanity check grid
    lines = layout.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    linewidths = set(len(line) for line in lines)
    if len(linewidths) > 1:
        raise ValueError('Invalid layout (all lines must have same width)')
    width = linewidths.pop()
    height = len(lines)
    panel_letters = set(c for line in lines for c in line) - set('.')
    # find bounding boxes for each panel
    panel_grid = {}
    for letter in panel_letters:
        left = min(x for x in range(width) for y in range(height) if lines[y][x] == letter)
        right = 1 + max(x for x in range(width) for y in range(height) if lines[y][x] == letter)
        top = min(y for x in range(width) for y in range(height) if lines[y][x] == letter)
        bottom = 1 + max(y for x in range(width) for y in range(height) if lines[y][x] == letter)
        panel_grid[letter] = (left, right, top, bottom)
        # check that this layout is consistent, i.e. all squares are filled
        valid = all(lines[y][x] == letter for x in range(left, right) for y in range(top, bottom))
        if not valid:
            raise ValueError('Invalid layout (not all square)')
    # build axis specs
    gs = gridspec.GridSpec(ncols=width, nrows=height, figure=fig)
    specs = {}
    for letter, (left, right, top, bottom) in panel_grid.items():
        specs[letter] = gs[top:bottom, left:right]
    return specs, gs


def panels(layout, fig=None):
    # default arguments
    if fig is None:
        fig = plt.gcf()
    specs, gs = panel_specs(layout, fig=fig)
    axes = {}
    for letter, spec in specs.items():
        axes[letter] = fig.add_subplot(spec)
    return axes, gs


def label_panel(ax, letter, *,
                offset_left=0.8, offset_up=0.2, prefix='', postfix='.', **font_kwds):
    kwds = dict(fontsize=18)
    kwds.update(font_kwds)
    # this mad looking bit of code says that we should put the code offset a certain distance in
    # inches (using the fig.dpi_scale_trans transformation) from the top left of the frame
    # (which is (0, 1) in ax.transAxes transformation space)
    fig = ax.figure
    trans = ax.transAxes + transforms.ScaledTranslation(-offset_left, offset_up, fig.dpi_scale_trans)
    ax.text(0, 1, prefix + letter + postfix, transform=trans, **kwds)


def label_panels(axes, letters=None, **kwds):
    if letters is None:
        letters = axes.keys()
    for letter in letters:
        ax = axes[letter]
        label_panel(ax, letter, **kwds)


def tight_xticklabels(ax=None):
    if ax is None:
        ax = plt.gca()
    ticklabels = ax.get_xticklabels()
    ticklabels[0].set_ha('left')
    ticklabels[0].set_text(' ' + ticklabels[0].get_text())
    ticklabels[-1].set_ha('right')
    ticklabels[-1].set_text(ticklabels[-1].get_text() + ' ')
    ax.set_xticklabels(ticklabels)


def tight_yticklabels(ax=None):
    if ax is None:
        ax = plt.gca()
    ticklabels = ax.get_yticklabels()
    ticklabels[0].set_va('bottom')
    ticklabels[-1].set_va('top')

# sys
from pathlib import Path

# sci
import scipy as sp
from scipy import signal, stats
import numpy as np

# ephys
import neo
import quantities as pq
import elephant as ele

# vis
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# own
from functions import *

def get_colors(units, palette='hls', desat=None, keep=True):
    """ return dict mapping unit labels to colors """
    if keep:
        n_colors = np.array(units).astype('int32').max()+1
    else:
        n_colors = len(units)
    colors = sns.color_palette(palette, n_colors=n_colors, desat=desat)
    # unit_ids = sp.arange(n_colors).astype('U')
    return dict(zip(units,colors))

def plot_Model(Model, max_rate=None, N=5, ax=None):
    """ plots a single model on ax """
    if ax is None:
        fig, ax = plt.subplots()

    colors = sns.color_palette('inferno',n_colors=N)
    if max_rate is None:
        max_rate = np.clip(np.max(Model.frates),0,200)

    frates = sp.linspace(0, max_rate, N)
    for j,f in enumerate(frates):
        ax.plot(Model.predict(f),color=colors[j])
    ax.text(0.05, 0.05, "%.2f"%max_rate, horizontalalignment='left', verticalalignment='baseline', transform=ax.transAxes,fontsize='small')
    return ax

def plot_Models(Models, max_rates=None, N=5, unit_order=None, save=None, colors=None):
    """ plots all models """
    units = list(Models.keys())

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(ncols=len(units), sharey=True, figsize=[len(units),2])
    for i,unit in enumerate(units):
        if max_rates is not None:
            max_rate = max_rates[i]
        else:
            max_rate = None
        axes[i] = plot_Model(Models[unit], ax=axes[i], max_rate=max_rate)
        axes[i].set_title(unit,color=colors[unit])
        
    axes[0].set_ylabel('amplitude')
    sns.despine(fig)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    return fig, axes

def plot_templates(Templates, SpikeInfo, unit_column=None, unit_order=None, N=100, save=None, colors=None):
    """ plots all templates """

    if unit_column is None:
        unit_column = 'unit'

    units = get_units(SpikeInfo, unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(ncols=len(units), sharey=True,  figsize=[len(units),2])

    for i, unit in enumerate(units):
        try:
            ix = SpikeInfo.groupby([unit_column,'good']).get_group((unit,True))['id']
            if N is not None and ix.shape[0] > N:
                ix = ix.sample(N)
            T = Templates[:,ix]
            axes[i].plot(T, color=colors[unit],alpha=0.5,lw=1)
        except:
            pass

        try:
            ix = SpikeInfo.groupby([unit_column,'good']).get_group((unit,False))['id']
            if N is not None and ix.shape[0] > N:
                ix = ix.sample(N)
            T = Templates[:,ix]

            axes[i].plot(T, color='k',alpha=0.5,lw=1,zorder=-1)
        except:
            pass
        
        axes[i].set_title(unit)

    sns.despine()
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes

def plot_segment(Seg, units, sigma=0.05, zscore=False, save=None, colors=None):
    """ inspect plots a segment """
    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(nrows=3,sharex=True)
    asig = Seg.analogsignals[0]
    for i, unit in enumerate(units):
        St, = select_by_dict(Seg.spiketrains, unit=unit)
        for t in St.times:
            axes[0].plot([t,t],[i-0.4, i+0.4], color=colors[unit])
        tvec = sp.linspace(asig.times.magnitude[0], asig.times.magnitude[-1], 1000)
        fr = est_rate(St.times.magnitude, tvec, sigma)
        if zscore:
            fr = ele.signal_processing.zscore(fr)
        axes[1].plot(tvec,fr,color=colors[unit])

    axes[2].plot(asig.times, asig, color='k',lw=0.5)

    # deco
    axes[0].set_yticks(range(len(units)))
    axes[0].set_yticklabels(units)
    axes[1].set_ylabel('firing rate (Hz)')
    axes[1].set_xlabel('time (s)')
    title = Path(Seg.annotations['filename']).stem
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    sns.despine(fig)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes

def plot_fitted_spikes(Segment, j, Models, SpikeInfo, unit_column, unit_order=None, zoom=None, save=None, colors=None):
    """ plot to inspect fitted spikes """
    fig, axes =plt.subplots(nrows=2, sharex=True, sharey=True)
    
    asig = Segment.analogsignals[0]
    axes[0].plot(asig.times, asig, color='k', lw=1)
    axes[1].plot(asig.times, asig, color='k', lw=1)

    units = get_units(SpikeInfo, unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]
    
    if colors is None:
        colors = get_colors(units)

    fs = asig.sampling_rate

    for u, unit in enumerate(units):
        St, = select_by_dict(Segment.spiketrains, unit=unit)

        asig_recons = sp.zeros(asig.shape[0])
        asig_recons[:] = sp.nan 

        wsize = 4*pq.ms # HARDCODE!
        wsize = (wsize * fs).simplified.magnitude.astype('int32') # HARDCODE

        inds = (St.times * fs).simplified.magnitude.astype('int32')
        offset = (St.t_start * fs).simplified.magnitude.astype('int32')
        inds = inds - offset

        try:
            frates = SpikeInfo.groupby([unit_column, 'segment']).get_group((unit,j))['frate_fast'].values
            pred_spikes = [Models[unit].predict(f) for f in frates]

            for i, spike in enumerate(pred_spikes):
                asig_recons[int(inds[i]-wsize/2):int(inds[i]+wsize/2)] = spike

            axes[1].plot(asig.times, asig_recons, lw=2.0, color=colors[unit], alpha=0.8)
        except KeyError:
            # thrown when no spikes are present in this segment
            pass

    if zoom is not None:
        for ax in axes:
            ax.set_xlim(zoom)
            
    stim_name = Path(Segment.annotations['filename']).stem
    fig.suptitle(stim_name)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    sns.despine(fig)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes

def plot_convergence(ScoresSum, save=None):
    """ convergence check """
    fig, axes = plt.subplots(figsize=[4,3])

    its = range(len(ScoresSum))
    axes.plot(its, ScoresSum, ':', color='k')
    axes.plot(its, ScoresSum, 'o')
    axes.set_title('convergence plot')
    axes.set_xlabel('iteration')
    axes.set_ylabel('total sum of best Rss')
    
    sns.despine(fig)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)

def plot_clustering(Templates, SpikeInfo, unit_column, n_components=5, N=300, save=None, colors=None, unit_order=None):
    """ clustering inspect """
    units = get_units(SpikeInfo,unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    spike_labels = SpikeInfo[unit_column]

    # pca
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(Templates.T)

    fig, axes = plt.subplots(figsize=[8,8], nrows=n_components, ncols=n_components,sharex=True,sharey=True)

    for unit in units:
        ix = sp.where(spike_labels == unit)[0]
        x = X[ix,:]
        if N > x.shape[0]:
            N = x.shape[0]
        for i in range(n_components):
            for j in range(n_components):
                ix = np.random.randint(0,x.shape[0],size=N)
                axes[i,j].plot(x[ix,i],x[ix,j],'.',color=colors[unit],markersize=1, alpha=0.5)

    for ax in axes.flatten():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    sns.despine(fig)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


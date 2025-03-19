# sys
from pathlib import Path

# sci
import numpy as np

# ephys
import quantities as pq
import elephant as ele

# vis
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# own
import sssort.functions as sf
from sssort import sssio

logger = sssio.get_logger()


def get_colors(units, palette='hls', desat=None, keep=True):
    """return dict mapping unit labels to colors"""
    if keep:
        n_colors = np.array(units).astype('int32').max() + 1
    else:
        n_colors = len(units)
    colors = sns.color_palette(palette, n_colors=n_colors, desat=desat)

    D = dict(zip(units, colors))
    D['-1'] = (0.5, 0.5, 0.5)
    return D


def plot_Model(Model, max_rate=None, N=5, ax=None):
    """plots a single model on ax"""
    if ax is None:
        fig, ax = plt.subplots()

    colors = sns.color_palette('inferno', n_colors=N)
    if max_rate is None:
        max_rate = np.clip(np.max(Model.frates), 0, 200)

    frates = np.linspace(0, max_rate, N)
    for j, f in enumerate(frates):
        ax.plot(Model.predict(f), color=colors[j])
    ax.text(
        0.05,
        0.05,
        '%.2f' % max_rate,
        horizontalalignment='left',
        verticalalignment='baseline',
        transform=ax.transAxes,
        fontsize='small',
    )
    return ax


def plot_Models(Models, max_rates=None, N=5, unit_order=None, save=None, colors=None):
    """plots all models"""
    units = list(Models.keys())

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(
        ncols=len(units), sharey=True, figsize=[len(units) * 1.5, 3]
    )
    for i, unit in enumerate(units):
        if max_rates is not None:
            max_rate = max_rates[i]
        else:
            max_rate = None
        axes[i] = plot_Model(Models[unit], ax=axes[i], max_rate=max_rate)
        axes[i].set_title(unit, color=colors[unit])

    axes[0].set_ylabel('amplitude')
    sns.despine(fig)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    return fig, axes


def plot_waveforms(
    Waveforms,
    SpikeInfo,
    dt,
    unit_column=None,
    unit_order=None,
    N=100,
    save=None,
    colors=None,
):
    """plots all waveforms"""

    units = sf.get_units(SpikeInfo, unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    # unit colors
    if colors is None:
        colors = get_colors(units)

    tvec = np.arange(-1 * Waveforms.shape[0] * dt / 2, Waveforms.shape[0] * dt / 2, dt)

    fig, axes = plt.subplots(
        ncols=len(units), sharey=True, figsize=[len(units) * 1.5, 3]
    )

    for i, unit in enumerate(units):
        SIgroup = SpikeInfo.groupby([unit_column, 'good']).get_group((unit, True))

        # color by firing rate
        frates = SIgroup['frate_fast']

        # subsampling spikes
        ix = SIgroup['id']
        if N is not None and ix.shape[0] > N:
            ix = ix.sample(N)
        T = Waveforms[:, ix]

        frates = frates[ix].values
        frates_order = np.argsort(frates)

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=0, vmax=frates.max())
        cols = cmap(norm(frates))

        # axes[i].plot(tvec, T, color=colors[unit],alpha=0.5,lw=1)
        for j in range(T.shape[1]):
            axes[i].plot(
                tvec,
                T[:, frates_order[j]],
                color=cols[frates_order[j]],
                alpha=0.75,
                lw=1,
                zorder=1,
            )

        try:
            ix = SpikeInfo.groupby([unit_column, 'good']).get_group((unit, False))['id']
            if N is not None and ix.shape[0] > N:
                ix = ix.sample(N)
            T = Waveforms[:, ix]
        except:  # no good spikes for this unit
            pass

        axes[i].plot(tvec, T, color='gray', alpha=0.5, lw=1, zorder=-1)

        axes[i].set_title(unit, color=colors[unit])
        axes[i].set_xlabel('time (ms)')

    sns.despine()
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


def plot_compare_waveforms(
    Waveforms, SpikeInfo, unit_column, dt, units_compare, N=100, save=None, colors=None
):
    """for manual merging: comparison plot of two units / waveforms"""

    if colors is None:
        all_units = sf.get_units(SpikeInfo, unit_column)
        colors = get_colors(all_units)

    tvec = (
        np.linspace(
            -1 * Waveforms.shape[0] * dt / 2,
            Waveforms.shape[0] * dt / 2,
            Waveforms.shape[0],
        )
        * 1e3
    )

    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=[6, 3])

    # color by firing rate
    frates_max = []
    groups = SpikeInfo.groupby([unit_column, 'good'])
    for unit in units_compare:
        frates_max.append(groups.get_group((unit, True))['frate_fast'].values.max())
    frate_max = np.array(frates_max).max()
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=0, vmax=frate_max)

    avgs = {}
    for i, unit in enumerate(units_compare):
        # get good spikes for unit
        SIgroups = SpikeInfo.groupby([unit_column, 'good'])

        if (unit, True) in SIgroups.groups:
            group = SIgroups.get_group((unit, True))
            ix = group['id']
            if N is not None and ix.shape[0] > N:
                ix = ix.sample(N)
            T = Waveforms[:, ix]

            frates = group['frate_fast'][ix].values
            cols = cmap(norm(frates))
            frates_order = np.argsort(frates)

            for j in range(T.shape[1]):
                axes[i].plot(
                    tvec,
                    T[:, frates_order[j]],
                    color=cols[frates_order[j]],
                    alpha=0.75,
                    lw=1,
                    zorder=1,
                )
            # axes[i].plot(tvec, T, color=colors[unit],alpha=0.5,lw=1)
            avgs[unit] = np.average(T, axis=1)

        # get rejected spikes for unit
        if (unit, False) in SIgroups.groups:
            ix = SpikeInfo.groupby([unit_column, 'good']).get_group((unit, False))['id']
            if N is not None and ix.shape[0] > N:
                ix = ix.sample(N)
            T = Waveforms[:, ix]
            axes[i].plot(tvec, T, color='k', alpha=0.5, lw=0.75, zorder=-1)

        axes[i].set_title(unit, color=colors[unit])
        axes[i].set_xlabel('time (ms)')

    for unit in units_compare:
        axes[2].plot(tvec, avgs[unit], color=colors[unit], lw=2, alpha=0.8)

    axes[2].set_title('both')
    axes[2].set_xlabel('time (ms)')

    sns.despine()
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


def plot_segment(Seg, units, sigma=0.05, zscore=False, save=None, colors=None):
    """inspect plots a segment"""
    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(nrows=3, sharex=True)
    asig = Seg.analogsignals[0]
    for i, unit in enumerate(units):
        (St,) = sf.select_by_dict(Seg.spiketrains, unit=unit)
        for t in St.times:
            axes[0].plot([t, t], [i - 0.4, i + 0.4], color=colors[unit])
        tvec = np.linspace(asig.times.magnitude[0], asig.times.magnitude[-1], 1000)
        fr = sf.est_rate(St.times.magnitude, tvec, sigma)
        if zscore:  # FIXME TypeError: The input must be a list of AnalogSignal
            fr = ele.signal_processing.zscore(fr)
        axes[1].plot(tvec, fr, color=colors[unit])

    axes[2].plot(asig.times, asig, color='k', lw=0.5)

    # deco
    axes[0].set_yticks(range(len(units)))
    axes[0].set_yticklabels(units)
    axes[1].set_ylabel('firing rate (Hz)')
    axes[1].set_xlabel('time (s)')
    try:
        title = Path(Seg.annotations['filename']).stem
    except:
        title = 'Segment %s' % (Seg.name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    sns.despine(fig)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


def plot_fitted_spikes(
    Segment,
    j,
    Models,
    SpikeInfo,
    unit_column,
    wsize,
    unit_order=None,
    zoom=None,
    save=None,
    colors=None,
):
    """plot to inspect fitted spikes"""
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)

    asig = Segment.analogsignals[0]
    axes[0].plot(asig.times, asig, color='k', lw=1)
    axes[1].plot(asig.times, asig, color='k', lw=1)

    units = sf.get_units(SpikeInfo, unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fs = asig.sampling_rate

    # Transform ms in points
    wsize = (wsize * fs).simplified.magnitude.astype('int32')

    for u, unit in enumerate(units):
        (St,) = sf.select_by_dict(Segment.spiketrains, unit=unit)

        asig_recons = np.zeros(asig.shape[0])
        asig_recons[:] = np.nan

        inds = (St.times * fs).simplified.magnitude.astype('int32')
        offset = (St.t_start * fs).simplified.magnitude.astype('int32')
        inds = inds - offset

        ix = np.logical_and(
            SpikeInfo['time'].values > St.t_start, SpikeInfo['time'].values < St.t_stop
        )
        SIgroups = SpikeInfo.iloc[ix].groupby([unit_column, 'segment'])

        if (unit, j) in SIgroups.groups:
            frates = SIgroups.get_group((unit, j))['frate_fast'].values
            pred_spikes = [Models[unit].predict(f) for f in frates]

            for i, spike in enumerate(pred_spikes):
                asig_recons[int(inds[i] - wsize / 2) : int(inds[i] + wsize / 2)] = spike

            axes[1].plot(asig.times, asig_recons, lw=2.0, color=colors[unit], alpha=0.8)

    if zoom is not None:
        for ax in axes:
            ax.set_xlim(zoom)

    try:
        stim_name = Path(Segment.annotations['filename']).stem
    except:
        stim_name = 'Segment %s' % (Segment.name)
    fig.suptitle(stim_name)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    sns.despine(fig)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


def plot_convergence(ScoresSum, save=None):
    """convergence check"""
    fig, axes = plt.subplots(figsize=[4, 3])

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


def plot_clustering(
    Waveforms,
    SpikeInfo,
    unit_column,
    color_by=None,
    n_components=4,
    N=300,
    save=None,
    colors=None,
    unit_order=None,
):
    """clustering inspect"""
    units = sf.get_units(SpikeInfo, unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    frates = SpikeInfo['frate_fast']

    # pca
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(Waveforms.T)

    fig, axes = plt.subplots(
        figsize=[7, 7], nrows=n_components, ncols=n_components, sharex=True, sharey=True
    )

    for i, unit in enumerate(units):
        SIgroup = SpikeInfo.groupby([unit_column, 'good']).get_group((unit, True))

        frates = SIgroup['frate_fast']

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=0, vmax=frates.max())

        # get the n-dim representation
        X_unit = X[SIgroup.index, :]

        if N > X_unit.shape[0]:
            N = X_unit.shape[0]

        ix = np.random.randint(X_unit.shape[0], size=N)

        n_components = X.shape[1]
        x = X_unit[ix, :]
        frates = frates.values[ix]
        cols = cmap(norm(frates))

        for i in range(n_components):
            for j in range(n_components):
                if color_by is None:
                    axes[i, j].scatter(
                        x[:, i],
                        x[:, j],
                        c=colors[unit],
                        s=0.5,
                        alpha=0.5,
                        edgecolor=None,
                    )
                else:
                    if color_by == unit:
                        c = cols
                        z = 1
                        a = 1
                    else:
                        c = 'gray'
                        z = -1
                        a = 0.5
                    axes[i, j].scatter(x[:, i], x[:, j], c=c, s=0.5, alpha=a, zorder=z)

    for ax in axes.flatten():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    sns.despine(fig)
    if color_by is not None:
        title = 'Clustering, colored = %s' % color_by
    else:
        title = 'Clustering'

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


def plot_spike_detect_inspect(
    AnalogSignal, min_height, min_prominence, Config=None, w=2 * pq.ms
):
    """interactive window visualizing: prominence and amplitude for each spike
    in a scatter plot. Click on scatter point = shows spike shape"""

    peak_mode = 'positive'  # default

    mad = sf.MAD(AnalogSignal)

    st_pos = sf.spike_detect(
        AnalogSignal, min_height=0, min_prominence=0, mode='positive'
    )
    st_neg = sf.spike_detect(
        AnalogSignal, min_height=0, min_prominence=0, mode='negative'
    )

    fig, axes = plt.subplots(ncols=2, nrows=2)

    # pos
    peaks = st_pos.annotations['amplitudes'] / mad
    proms = st_pos.annotations['prominences'] / mad
    peak_ix_pos = st_pos.annotations['index']

    scatter_pos = axes[0, 0].scatter(proms, peaks, s=0.25, alpha=0.25, picker=True)
    height_line_pos = axes[0, 0].axhline(min_height, color='k', lw=1, linestyle=':')
    prom_line_pos = axes[0, 0].axvline(min_prominence, color='k', lw=1, linestyle=':')
    (line_pos,) = axes[0, 1].plot([], color='k', lw=1)

    a = np.max(peaks) * mad
    axes[0, 1].set_ylim(-a, a)
    axes[0, 1].set_xlim(-w, w)
    axes[0, 1].axhline(0, linestyle=':')

    # neg
    peaks = st_neg.annotations['amplitudes'] / mad
    proms = st_neg.annotations['prominences'] / mad
    peak_ix_neg = st_neg.annotations['index']

    scatter_neg = axes[1, 0].scatter(proms, peaks, s=0.25, alpha=0.25, picker=True)
    height_line_neg = axes[1, 0].axhline(min_height, color='k', lw=1, linestyle=':')
    prom_line_neg = axes[1, 0].axvline(min_prominence, color='k', lw=1, linestyle=':')
    (line_neg,) = axes[1, 1].plot([], color='k', lw=1)

    a = np.max(peaks) * mad
    axes[1, 1].set_ylim(-a, a)
    axes[1, 1].set_xlim(-w, w)
    axes[1, 1].axhline(0, linestyle=':')

    # deco
    axes[0, 0].set_ylabel('Positive spikes\nAmplitude')
    axes[1, 0].set_ylabel('Negative spikes\nAmplitude')
    axes[1, 0].set_xlabel('Prominence')
    axes[1, 1].set_xlabel('Time (ms)')

    # interative picker
    def onpick(event):
        if event.mouseevent.button == 1:
            if event.artist == scatter_pos:
                ix = event.ind[0]
                t = AnalogSignal.times[peak_ix_pos[ix]]
                y = AnalogSignal.time_slice(t - w, t + w).flatten().magnitude
                x = np.linspace(-w.magnitude, w.magnitude, y.shape[0])
                line_pos.set_data(x, y)

            if event.artist == scatter_neg:
                ix = event.ind[0]
                t = AnalogSignal.times[peak_ix_neg[ix]]
                y = AnalogSignal.time_slice(t - w, t + w).flatten().magnitude
                x = np.linspace(-w.magnitude, w.magnitude, y.shape[0])
                line_neg.set_data(x, y)

        fig.canvas.draw()

    def on_press(event):
        # store and exit
        if event.key == 'enter':
            # prep values
            if peak_mode == 'positive':
                (min_threshold_scale,) = height_line_pos.get_ydata()
                (min_prominence,) = prom_line_pos.get_xdata()
            if peak_mode == 'negative':
                (min_threshold_scale,) = height_line_neg.get_ydata()
                (min_prominence,) = prom_line_neg.get_xdata()

            # set
            Config['spike detect']['peak_mode'] = peak_mode
            Config['spike detect']['min_threshold_scale'] = '%.3f' % min_threshold_scale
            Config['spike detect']['min_prominence'] = '%.3f' % min_prominence

            logger.info(
                'updating config with values for spike amplitude, prominence, and direction'
            )
            with open(Config.filepath, 'w') as fH:
                Config.write(fH)

    def on_click(event):
        if event.button == 3:
            x, y = event.xdata, event.ydata
            if event.inaxes == axes[0, 0]:
                height_line_pos.set_ydata(y)
                prom_line_pos.set_xdata(x)
                peak_mode = 'positive'
            if event.inaxes == axes[1, 0]:
                height_line_neg.set_ydata(y)
                prom_line_neg.set_xdata(x)
                peak_mode = 'negative'

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', on_press)

    fig.suptitle(
        'left-click on scatter to visualize spike\nenter to store thresholds to config',
        size='small',
    )
    sns.despine(fig)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)


def plot_spike_detect(
    AnalogSignal, mad_thresh, min_prominence, N=4, w=0.2 * pq.s, save=None
):
    """ """

    fig, axes = plt.subplots(nrows=N, sharey=True)

    mad = sf.MAD(AnalogSignal)

    # TODO: fix unefficient ?
    max_ampl = np.max(AnalogSignal).item()

    for i in range(N):
        t_start = np.random.rand() * AnalogSignal.times[-1] - w

        # avoid negative values
        t_start = t_start * 0 if t_start < 0 else t_start
        t_stop = t_start + w

        asig = AnalogSignal.time_slice(t_start, t_stop)
        axes[i].plot(asig.times, asig, color='k', lw=1, alpha=0.7)
        axes[i].set_xticks([])
        axes[i].plot(
            asig.times,
            np.ones(asig.times.shape) * mad_thresh,
            linewidth=0.5,
            label='amplitude',
        )  # FIXME this breaks
        axes[i].plot(
            asig.times,
            np.ones(asig.times.shape) * min_prominence,
            linewidth=0.5,
            color='red',
            label='min_prominence',
        )

    for ax in axes:
        # ax.set_ylim(-5, 5)  # if zscored
        ax.set_ylim(-max_ampl, max_ampl)

        ax.axhline(0, color='gray', lw=0.5, alpha=1, zorder=-1)

    sns.despine(fig, bottom=True)
    fig.tight_layout()
    plt.legend()

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


# TODO reduce complexity and combine to plot_compare spike events
def plot_spike_events(
    Segment,
    thres,
    min_prominence,
    wsize=4,
    max_window=1,
    max_row=5,
    save=None,
    save_format='.png',
    show=False,
    st=None,
    rejs=None,
):
    plt.rcParams.update({'font.size': 5})
    for asig in Segment.analogsignals:
        max_window = int(
            max_window * asig.sampling_rate
        )  # FIX conversion from secs to points

        asig = asig.reshape(asig.shape[0])
        asig_max = np.amax(asig)
        asig_min = np.amin(asig)
        n_rows = (
            asig.shape[0] // max_window
        )  # compute number of rows needed to plot complete signal
        n_plots = n_rows // max_row + int(not (n_rows / max_row).is_integer())

        # Plot max_row rows per window, plots n_plots to plot the complete signal
        for i_fig in range(0, n_plots):
            fig, axes = plt.subplots(nrows=max_row, sharey=True)

            for idx in range(0, max_row):  # plot the max_row axes.
                # plot analog signal
                ini = i_fig * max_window * max_row + idx * max_window
                end = i_fig * max_window * max_row + idx * max_window + max_window

                end = min(end, asig.data.shape[0]) - 1
                if ini >= asig.data.shape[0]:
                    break

                # Convert memory view to NumPy array
                data_array = np.array(asig.data)
                # Convert quantity to NumPy array
                times_array = asig.times.magnitude

                axes[idx].plot(
                    times_array[ini:end], data_array[ini:end], linewidth=1, color='k'
                )

                if st is None:
                    st = Segment.spiketrains[
                        0
                    ]  # get spike trains (assuming there's only one spike train)

                t_ini = asig.times[ini]
                t_end = asig.times[end]
                # get events in this chunk
                t_events = st.times[np.where((st.times > t_ini) & (st.times < t_end))]

                # TODO: fix unefficient ?
                max_ampl = np.max(asig.data[ini:end]).item()
                a_events = np.ones(t_events.shape) * max_ampl

                if rejs is not None:
                    chunk_r = rejs[np.where((rejs > t_ini) & (rejs < t_end))]
                    axes[idx].plot(
                        chunk_r,
                        asig_max * np.ones(chunk_r.shape),
                        '|',
                        markersize=5,
                        color='r',
                        label='rejected_spikes',
                    )

                axes[idx].plot(
                    t_events,
                    a_events,
                    '|',
                    markersize=5,
                    label='detected_spikes',
                    c=[0.5, 1.0, 0.5],
                )
                axes[idx].plot(
                    asig.times[ini:end],
                    np.ones(asig.times[ini:end].shape) * thres,
                    linewidth=0.5,
                    label='amplitude',
                )
                axes[idx].plot(
                    asig.times[ini:end],
                    np.ones(asig.times[ini:end].shape) * min_prominence,
                    linewidth=0.5,
                    color='darkgreen',
                    label='min_prominence',
                )
                # axes[idx].set_ylim(asig_min*1.1,asig_max*1.1)
            if idx == 0:
                break

            # set plot info:
            fig.suptitle('Spike Detection (%d/%d)' % (i_fig + 1, n_plots))
            axes[idx].set_xlabel('Time (%s)' % str(asig.times.units).split()[-1])
            v_unit = str(asig.units).split()[-1]
            if v_unit == 'dimensionless':
                v_unit = 'V'
            axes[idx // 2].set_ylabel('Voltage (%s)' % v_unit)

            plt.tight_layout()
            plt.legend()

            if save is not None:
                fig.savefig(str(save) + '_%d' % i_fig + save_format)

            if show:
                plt.show()
            else:
                plt.close(fig)

    # return fig, axes


"""
 
 ########   #######   ######  ######## ########  ########   #######   ######  ########  ######   ######  #### ##    ##  ######   
 ##     ## ##     ## ##    ##    ##    ##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ##  ##  ###   ## ##    ##  
 ##     ## ##     ## ##          ##    ##     ## ##     ## ##     ## ##       ##       ##       ##        ##  ####  ## ##        
 ########  ##     ##  ######     ##    ########  ########  ##     ## ##       ######    ######   ######   ##  ## ## ## ##   #### 
 ##        ##     ##       ##    ##    ##        ##   ##   ##     ## ##       ##             ##       ##  ##  ##  #### ##    ##  
 ##        ##     ## ##    ##    ##    ##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ##  ##  ##   ### ##    ##  
 ##         #######   ######     ##    ##        ##     ##  #######   ######  ########  ######   ######  #### ##    ##  ######   
 
"""

# TODO replace with matplotlib.Patches.Rectangle


def plot_box(ax, box):
    bot = ax.get_ylim()[0]
    top = ax.get_ylim()[1]
    left = box[0] - box[1] / 2
    right = box[0] + box[1] / 2
    x = [left, left, right, right, left]
    y = [bot, top, top, bot, bot]
    ax.plot(x, y, 'r', lw=0.5)


def plot_spike_labels(ax, SpikeInfo, spike_label_interval):
    if spike_label_interval > 0:
        xrg = ax.get_xlim()
        lbl = SpikeInfo['id'][::spike_label_interval]
        xpo = SpikeInfo['time'][::spike_label_interval]
        lbl = lbl[np.logical_and(xpo > xrg[0], xpo < xrg[1])]
        xpo = xpo[np.logical_and(xpo > xrg[0], xpo < xrg[1])]
        fac = ax.get_ylim()[0] * 0.9
        ypo = np.ones(len(xpo)) * fac
        for x, y, s in zip(xpo, ypo, lbl):
            ax.text(x, y, str(s), ha='center', fontsize=4)


# TODO: change wsize by config param + transform
def plot_by_unit(
    ax, st, asig, Models, SpikeInfo, unit_column, unit_order=None, colors=None, wsize=40
):
    try:
        left = wsize[0]
        right = wsize[1]
    except:
        left = wsize // 2
        right = wsize // 2

    units = sf.get_units(SpikeInfo, unit_column)
    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fs = asig.sampling_rate

    for u, unit in enumerate(units):
        # St, = select_by_dict(Segment.spiketrains, unit=unit)
        fac = ax.get_ylim()[1] * (0.95 - 0.05 * u)
        times = SpikeInfo['time'][SpikeInfo[unit_column] == unit]
        ax.plot(
            times,
            np.ones(times.shape) * fac,
            '|',
            markersize=2,
            linewidth=0.5,
            label=unit + ' spikes',
            color=colors[unit],
        )
        asig_recons = np.zeros(asig.shape[0])
        asig_recons[:] = np.nan

        # inds = (St.times * fs).simplified.magnitude.astype('int32')

        # get times from SpikeInfo so units are extracted
        # from Spike and not from annotation in spiketrains
        times = SpikeInfo.groupby([unit_column]).get_group((unit))['time'].values

        # TN: I don't see the point in this & it breaks where new spikes are inserted
        # fr = (asig.times[1]-asig.times[0]).simplified.magnitude.astype('int32')
        # Inds = [np.where(np.isclose(t,np.array(st.times),atol=fr))[0][0] for t in np.array(times)]

        # inds = (st.times[Inds]*fs).simplified.magnitude.astype('int32')
        inds = (times * fs).simplified.magnitude.astype('int32')

        offset = (st.t_start * fs).simplified.magnitude.astype('int32')
        inds = inds - offset

        try:
            if type(Models).__name__ == 'dict':
                frates = (
                    SpikeInfo.groupby([unit_column])
                    .get_group((unit))['frate_fast']
                    .values
                )
                pred_spikes = [Models[unit].predict(f) for f in frates]
            else:
                Waveforms = Models
                ix = SpikeInfo.groupby([unit_column]).get_group((unit))['id']
                pred_spikes = Waveforms[:, ix].T

            for i, spike in enumerate(pred_spikes):
                try:
                    asig_recons[inds[i] - left : inds[i] + right] = spike

                except ValueError as e:
                    print('In plot by unit exception:', e.args)
                    # thrown when first or last spike smaller than reconstruction window
                    continue
            ax.plot(asig.times, asig_recons, lw=1.0, color=colors[unit], alpha=0.8)

        except KeyError:
            # thrown when no spikes are present in this segment
            pass


def plot_fitted_spikes_pp(
    Segment,
    Models,
    SpikeInfo,
    unit_column,
    unit_order=None,
    zoom=None,
    box=None,
    save=None,
    colors=None,
    wsize=40,
    rejs=None,
    spike_label_interval=0,
):
    """plot to inspect fitted spikes"""
    fig, axes = plt.subplots(
        nrows=2, sharex=True, sharey=True, num=1, clear=True, figsize=[4, 3]
    )

    asig = Segment.analogsignals[0]
    fs = asig.sampling_rate
    as_min = np.amin(asig)
    as_max = np.amax(asig)
    ylim = [1.1 * as_min, 1.1 * as_max]

    # if zoom is not None:
    # left= max(int(zoom[0]*fs),0)
    # right= min(int(zoom[1]*fs),len(asig.data))
    # else:
    # left= 0
    # right= len(asig.data)

    # axes[0].plot(asig.times[left:right], asig.data[left:right], color='k', lw=0.5)
    # axes[1].plot(asig.times[left:right], asig.data[left:right], color='k', lw=0.5)
    times = np.array(asig.times)
    data = np.array(asig.data)
    axes[0].plot(times, data, color='k', lw=0.5)
    axes[1].plot(times, data, color='k', lw=0.5)

    if zoom is not None:
        for ax in axes:
            ax.set_xlim(zoom)

    for ax in axes:
        ax.set_ylim(ylim)

    st = Segment.spiketrains[
        0
    ]  # get all spike trains (assuming there's only one spike train)
    if rejs is None:
        rejs = SpikeInfo['time'][SpikeInfo[unit_column] == '-2']
    units = sf.get_units(SpikeInfo, unit_column)
    fac = axes[1].get_ylim()[1] * (0.95 - len(units) * 0.05)
    axes[1].plot(
        rejs,
        np.ones(rejs.shape) * fac,
        '|',
        markersize=2,
        color='k',
        label='rejected spike',
    )

    plot_by_unit(
        axes[1], st, asig, Models, SpikeInfo, unit_column, unit_order, colors, wsize
    )

    if box is not None:
        plot_box(axes[0], box)
        plot_box(axes[1], box)

    plot_spike_labels(axes[0], SpikeInfo, spike_label_interval)
    plot_spike_labels(axes[1], SpikeInfo, spike_label_interval)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    # sns.despine(fig)

    fig.legend()

    if save is not None:
        fig.savefig(save)
        # plt.close(fig)

    return fig, axes


def plot_fitted_spikes_complete(
    seg,
    Models,
    SpikeInfo,
    unit_column,
    max_window,
    plots_folder,
    fig_format,
    unit_order=None,
    save=None,
    colors=None,
    wsize=40,
    extension='',
    plot_function=plot_fitted_spikes_pp,
    rejs=None,
    spike_label_interval=0,
):
    asig = seg.analogsignals[0]

    max_window = int(
        max_window * asig.sampling_rate
    )  # FIX conversion from secs to points
    n_plots = asig.shape[0] // max_window

    for n_plot in range(0, n_plots):
        outpath = plots_folder / (
            'fitted_spikes%s_%s_%d' % (extension, max_window, n_plot) + fig_format
        )
        ini = n_plot * max_window
        end = ini + max_window
        zoom = [ini, end] / asig.sampling_rate

        if rejs is None:
            rejs = SpikeInfo['time'][SpikeInfo[unit_column] == '-2']

        plot_function(
            seg,
            Models,
            SpikeInfo,
            unit_column,
            zoom=zoom,
            save=outpath,
            wsize=wsize,
            rejs=rejs,
            spike_label_interval=spike_label_interval,
            colors=colors,
        )


def plot_means(
    means, units, template_a, template_b, asigs, outpath=None, show=False, colors=None
):
    fig, axes = plt.subplots(ncols=len(units), figsize=[len(units) * 3, 4])

    if colors is None:
        colors = {'A': 'b', 'B': 'g', '?': 'r'}

    for i, (mean, unit) in enumerate(zip(means, units)):
        axes[i].plot(mean, label=unit, color='k', linewidth=0.7)
        axes[i].plot(mean, color=colors[asigs[unit]], alpha=0.3, linewidth=5)
        axes[i].plot(template_a, label='A', color=colors['A'], linewidth=0.7)
        axes[i].plot(template_b, label='B', color=colors['B'], linewidth=0.7)
        axes[i].legend()

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    if show:
        plt.show()

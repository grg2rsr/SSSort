# sys
import sys
import os
import shutil
import dill
import configparser
from pathlib import Path

# sci
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import quantities as pq

# ephys
import neo
import elephant as ele

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# own
import sssort.functions as sf
import sssort.plotters as sp
import sssio

print('This is SSSort v1.0.0')
print('author: Georg Raiser - grg2rsr@gmail.com')

"""
 
 #### ##    ## #### 
  ##  ###   ##  ##  
  ##  ####  ##  ##  
  ##  ## ## ##  ##  
  ##  ##  ####  ##  
  ##  ##   ###  ##  
 #### ##    ## #### 
 
"""

config_path = Path(os.path.abspath(sys.argv[1]))
mode = sys.argv[2]

Config = configparser.ConfigParser()
Config.read(config_path)
Config.filepath = config_path

# handling paths and creating output directory
data_path = Path(Config.get('path', 'data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path', 'experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'
detected_folder = plots_folder / 'detected_spikes'
fitted_folder = plots_folder / 'fitted_spikes'

os.makedirs(plots_folder, exist_ok=True)
os.makedirs(detected_folder, exist_ok=True)
os.makedirs(fitted_folder, exist_ok=True)

os.chdir(config_path.parent / exp_name)

log_path = (config_path.parent / exp_name / exp_name).with_suffix('.log')
logger = sssio.create_logger(filename=log_path)

logger.info(f'config file read from {config_path}')

# copy config
shutil.copyfile(config_path, config_path.parent / exp_name / config_path.name)

# read data
Blk = sssio.get_data(data_path)
for seg in Blk.segments:
    if 'filename' not in seg.annotations:
        logger.critical('segment metadata incomplete, filename missing')
        sys.exit()
Blk.name = exp_name
logger.info(f'data read from {data_path}')

# get data properies
fs = Blk.segments[0].analogsignals[0].sampling_rate
dt = (1 / fs).rescale(pq.s)

# plotting
mpl.rcParams['figure.dpi'] = Config.get('output', 'fig_dpi')
fig_format = Config.get('output', 'fig_format')

"""
 
 ########  ########  ######## ########  ########   #######   ######  ########  ######   ######  
 ##     ## ##     ## ##       ##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ## 
 ##     ## ##     ## ##       ##     ## ##     ## ##     ## ##       ##       ##       ##       
 ########  ########  ######   ########  ########  ##     ## ##       ######    ######   ######  
 ##        ##   ##   ##       ##        ##   ##   ##     ## ##       ##             ##       ## 
 ##        ##    ##  ##       ##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ## 
 ##        ##     ## ######## ##        ##     ##  #######   ######  ########  ######   ######  
 
"""

logger.info(' - preprocessing - ')

for seg in Blk.segments:
    seg.analogsignals[0].annotate(kind='original')

# highpass filter
freq = Config.getfloat('preprocessing', 'highpass_freq')
logger.info(f'highpass filtering data at {freq:.2f} Hz')
for seg in Blk.segments:
    seg.analogsignals[0] = ele.signal_processing.butter(
        seg.analogsignals[0], highpass_freq=freq
    )

if Config.getboolean('preprocessing', 'z_score'):
    logger.info('z-scoring signals')
    for seg in Blk.segments:
        seg.analogsignals = [ele.signal_processing.zscore(seg.analogsignals)]


"""
 
  ######  ########  #### ##    ## ########    ########  ######## ######## ########  ######  ######## 
 ##    ## ##     ##  ##  ##   ##  ##          ##     ## ##          ##    ##       ##    ##    ##    
 ##       ##     ##  ##  ##  ##   ##          ##     ## ##          ##    ##       ##          ##    
  ######  ########   ##  #####    ######      ##     ## ######      ##    ######   ##          ##    
       ## ##         ##  ##  ##   ##          ##     ## ##          ##    ##       ##          ##    
 ##    ## ##         ##  ##   ##  ##          ##     ## ##          ##    ##       ##    ##    ##    
  ######  ##        #### ##    ## ########    ########  ########    ##    ########  ######     ##    

"""

logger.info(' - spike detect - ')

global_mad = np.average([sf.MAD(seg.analogsignals[0]) for seg in Blk.segments])
mad_thresh = Config.getfloat('spike detect', 'min_theshold_scale')
min_prominence = Config.getfloat('spike detect', 'min_prominence_scale')
wsize = Config.getfloat('spike detect', 'wsize') * pq.ms

spike_detect_only = Config.getboolean('spike detect', 'spike_detect_only')
extense_plot = Config.getboolean('spike sort', 'plot_fitted_spikes_extense')
peak_mode = Config.get('spike detect', 'peak_mode')

# TODO general printing of used parameters in log. They are logged anyways, but still
# logger.info("min_theshold_scale was %f, global_mad is %f, used mad is: %f"%(mad_thresh, global_mad, mad_thresh*global_mad))
# logger.info("min_prominence was %f, global_mad is %f, used min_prominence is: %f"%(min_prominence, global_mad, min_prominence*global_mad))

# if only spike detection: diagnostic plot and option to continue with spike detection
if mode == 'detect':
    j = np.random.randint(len(Blk.segments))
    seg = Blk.segments[j]  # select a segment at random

    (AnalogSignal,) = sf.select_by_dict(seg.analogsignals, kind='original')

    plt.ion()
    mpl.rcParams['figure.dpi'] = Config.get('output', 'screen_dpi')
    sp.plot_spike_detect_inspect(AnalogSignal, mad_thresh, min_prominence, Config)
    # plot_spike_detect(AnalogSignal, mad_thresh, min_prominence, N=5, w=2*pq.s)
    logger.info('only spike detection - press enter to quit')
    input()  # halt terminal here
    plt.ioff()
    sys.exit()

for i, seg in enumerate(Blk.segments):
    (AnalogSignal,) = sf.select_by_dict(seg.analogsignals, kind='original')

    # spike detection
    st = sf.spike_detect(AnalogSignal, mad_thresh, min_prominence, mode=peak_mode)
    st.annotate(kind='all_spikes')

    if len(st) == 0:
        logger.critical(
            'No spikes detected, change the threshold values in the configuration file'
        )
        exit()

    # remove border spikes
    st_cut = st.time_slice(st.t_start + wsize / 2, st.t_stop - wsize / 2)
    st_cut.t_start = st.t_start
    seg.spiketrains.append(st_cut)

    if extense_plot:
        logger.info('extense plot on, plotting detected spikes...')
        # Plot detected spikes
        namepath = detected_folder / f'spike_detection_{i}'
        sp.plot_spike_events(
            seg,
            min_prominence=min_prominence,
            thres=sf.MAD(AnalogSignal) * mad_thresh,
            save=namepath,
            save_format=fig_format,
            max_window=0.4,
            max_row=3,
        )

n_spikes = np.sum([seg.spiketrains[0].shape[0] for seg in Blk.segments])
logger.info(f'total number of detected spikes: {n_spikes}')

"""
 
 ##      ##    ###    ##     ## ######## ########  #######  ########  ##     ##  ######  
 ##  ##  ##   ## ##   ##     ## ##       ##       ##     ## ##     ## ###   ### ##    ## 
 ##  ##  ##  ##   ##  ##     ## ##       ##       ##     ## ##     ## #### #### ##       
 ##  ##  ## ##     ## ##     ## ######   ######   ##     ## ########  ## ### ##  ######  
 ##  ##  ## #########  ##   ##  ##       ##       ##     ## ##   ##   ##     ##       ## 
 ##  ##  ## ##     ##   ## ##   ##       ##       ##     ## ##    ##  ##     ## ##    ## 
  ###  ###  ##     ##    ###    ######## ##        #######  ##     ## ##     ##  ######  
 
"""

logger.info(' - getting waveforms - ')

n_samples = (wsize * fs).simplified.magnitude.astype('int32')

waveforms = []
for j, seg in enumerate(Blk.segments):
    (AnalogSignal,) = sf.select_by_dict(seg.analogsignals, kind='original')
    data = AnalogSignal.magnitude.flatten()

    (SpikeTrain,) = sf.select_by_dict(seg.spiketrains, kind='all_spikes')
    inds = (SpikeTrain.times * fs).simplified.magnitude.astype('int32')

    waveforms.append(sf.get_Waveforms(data, inds, n_samples))

Waveforms = np.concatenate(waveforms, axis=1)

# waveforms to disk
outpath = results_folder / 'Waveforms.npy'
np.save(outpath, Waveforms)
logger.info(f'saving spike waveforms to {outpath}')

"""
 
  ######  ##       ##     ##  ######  ######## ######## ########  
 ##    ## ##       ##     ## ##    ##    ##    ##       ##     ## 
 ##       ##       ##     ## ##          ##    ##       ##     ## 
 ##       ##       ##     ##  ######     ##    ######   ########  
 ##       ##       ##     ##       ##    ##    ##       ##   ##   
 ##    ## ##       ##     ## ##    ##    ##    ##       ##    ##  
  ######  ########  #######   ######     ##    ######## ##     ## 
 
"""

n_clusters_init = Config.getint('spike sort', 'init_clusters')
logger.info(f'initial k-means with {n_clusters_init} clusters')

# initial clustering in the same space as subsequent spikes models
n_model_comp = Config.getint('spike model', 'n_model_comp')
pca = PCA(n_components=n_model_comp)
X = pca.fit_transform(Waveforms.T)

# the ini labels
spike_labels = KMeans(n_clusters=n_clusters_init).fit_predict(X).astype('U')

"""
 
  ######  ########  #### ##    ## ######## #### ##    ## ########  #######  
 ##    ## ##     ##  ##  ##   ##  ##        ##  ###   ## ##       ##     ## 
 ##       ##     ##  ##  ##  ##   ##        ##  ####  ## ##       ##     ## 
  ######  ########   ##  #####    ######    ##  ## ## ## ######   ##     ## 
       ## ##         ##  ##  ##   ##        ##  ##  #### ##       ##     ## 
 ##    ## ##         ##  ##   ##  ##        ##  ##   ### ##       ##     ## 
  ######  ##        #### ##    ## ######## #### ##    ## ##        #######  
 
"""
#  make a SpikeInfo dataframe
SpikeInfo = pd.DataFrame()

# count spikes
n_spikes = Waveforms.shape[1]
SpikeInfo['id'] = np.arange(n_spikes, dtype='int32')

# get all spike times
spike_times = []
for seg in Blk.segments:
    (SpikeTrain,) = sf.select_by_dict(seg.spiketrains, kind='all_spikes')
    spike_times.append(SpikeTrain.times.magnitude)
spike_times = np.concatenate(spike_times)
SpikeInfo['time'] = spike_times

# get segment labels
segment_labels = []
for i, seg in enumerate(Blk.segments):
    (SpikeTrain,) = sf.select_by_dict(seg.spiketrains, kind='all_spikes')
    segment_labels.append(SpikeTrain.shape[0] * [i])
segment_labels = np.concatenate(segment_labels)
SpikeInfo['segment'] = segment_labels

# get all labels
SpikeInfo['unit_0'] = spike_labels

# get clean waveforms
n_neighbors = Config.getint('spike model', 'template_reject')
SpikeInfo = sf.reject_spikes(Waveforms, SpikeInfo, 'unit_0', n_neighbors, verbose=True)

# unassign spikes if unit has too little good spikes
SpikeInfo = sf.reject_unit(SpikeInfo, 'unit_0', min_good=40)

"""
 
 #### ##    ## #### ######## 
  ##  ###   ##  ##     ##    
  ##  ####  ##  ##     ##    
  ##  ## ## ##  ##     ##    
  ##  ##  ####  ##     ##    
  ##  ##   ###  ##     ##    
 #### ##    ## ####    ##    
 
"""
logger.info(' - initializing algorithm - ')

# rate estimation
kernel_slow = Config.getfloat('kernels', 'sigma_slow')
kernel_fast = Config.getfloat('kernels', 'sigma_fast')
sf.calc_update_frates(SpikeInfo, 'unit_0', kernel_fast, kernel_slow)

# model
n_model_comp = Config.getint('spike model', 'n_model_comp')
Models = sf.train_Models(SpikeInfo, 'unit_0', Waveforms, n_comp=n_model_comp)
outpath = plots_folder / ('Models_ini' + fig_format)
sp.plot_Models(Models, save=outpath)

# plot waveforms
outpath = plots_folder / ('waveforms_init' + fig_format)
sp.plot_waveforms(
    Waveforms,
    SpikeInfo,
    dt.rescale(pq.ms).magnitude,
    unit_column='unit_0',
    N=100,
    save=outpath,
)

"""
 
 #### ######## ######## ########     ###    ######## ######## 
  ##     ##    ##       ##     ##   ## ##      ##    ##       
  ##     ##    ##       ##     ##  ##   ##     ##    ##       
  ##     ##    ######   ########  ##     ##    ##    ######   
  ##     ##    ##       ##   ##   #########    ##    ##       
  ##     ##    ##       ##    ##  ##     ##    ##    ##       
 ####    ##    ######## ##     ## ##     ##    ##    ######## 
 
"""

units = sf.get_units(SpikeInfo, 'unit_0')
n_units = len(units)

n_max_iter = Config.getint('spike sort', 'iterations')

force_merge = Config.getboolean('spike sort', 'force_merge')
manual_merge = Config.getboolean('spike sort', 'manual_merge')

clust_alpha = Config.getfloat('spike sort', 'clust_alpha')
n_clust_final = Config.getint('spike sort', 'n_clust_final')

reassign_penalty = Config.getfloat('spike sort', 'reassign_penalty')
noise_penalty = Config.getfloat('spike sort', 'noise_penalty')
sorting_noise = Config.getfloat('spike sort', 'f_noise')
conv_crit = Config.getfloat('spike sort', 'conv_crit')
n_hist = Config.getint('spike sort', 'history_len')

use_fr = Config.getboolean('spike sort', 'use_fr')
fr_weight = Config.getfloat('spike sort', 'fr_weight')

# hardcoded parameters
alpha_incr = 0.05
max_alpha = 100

rejected_merges = []
ScoresSum = []
AICs = []

for it in range(1, n_max_iter):
    # unit columns
    prev_unit_col = f'unit_{it - 1}'
    this_unit_col = f'unit_{it}'

    if it > 2:
        # randomly unassign a fraction of spikes
        N = int(n_spikes * sorting_noise)
        SpikeInfo.loc[SpikeInfo.sample(N).index, prev_unit_col] = '-1'

    # update rates
    sf.calc_update_frates(SpikeInfo, prev_unit_col, kernel_fast, kernel_slow)

    # train models with labels from last iteration
    Models = sf.train_Models(SpikeInfo, prev_unit_col, Waveforms, n_comp=n_model_comp)
    outpath = (plots_folder / f'Models_{prev_unit_col}').with_suffix(fig_format)
    sf.plot_Models(Models, save=outpath)

    # Score spikes with models
    Scores, units = sf.Score_spikes(
        Waveforms,
        SpikeInfo,
        prev_unit_col,
        Models,
        score_metric=sf.Rss,
        reassign_penalty=reassign_penalty,
        noise_penalty=noise_penalty,
    )

    # assign new labels
    min_ix = np.argmin(Scores, axis=1)
    SpikeInfo[this_unit_col] = np.array([units[i] for i in min_ix], dtype='U')

    # clean assignment
    SpikeInfo = sf.reject_spikes(Waveforms, SpikeInfo, this_unit_col)
    SpikeInfo = sf.reject_unit(SpikeInfo, this_unit_col)

    # plot waveforms
    outpath = (plots_folder / f'Waveforms_{this_unit_col}').with_suffix(fig_format)
    sf.plot_waveforms(
        Waveforms,
        SpikeInfo,
        dt.rescale(pq.ms).magnitude,
        this_unit_col,
        N=100,
        save=outpath,
    )

    # Model eval
    valid_ix = np.where(SpikeInfo[this_unit_col] != '-1')[0]
    Rss_sum = (
        np.sum(np.min(Scores[valid_ix], axis=1)) / valid_ix.shape[0]
    )  # Waveforms.shape[1]
    ScoresSum.append(Rss_sum)
    units = sf.get_units(SpikeInfo, this_unit_col)
    AICs.append(len(units) - 2 * np.log(Rss_sum))

    # print iteration info
    n_changes, _ = sf.get_changes(SpikeInfo, this_unit_col)
    logger.info(
        f'Iteration: {it} - Rss: {Rss_sum:.2e} - # reassigned spikes: {n_changes}'
    )

    # Plot zoom inspect for current iteration
    zoom = np.array(Config.get('output', 'zoom').split(','), dtype='float32') / 1000
    for j, Seg in enumerate(Blk.segments):
        seg_name = Path(Seg.annotations['filename']).stem
        outpath = plots_folder / (seg_name + '_fitted_spikes_' + str(it) + fig_format)
        sp.plot_fitted_spikes_pp(
            Seg, Models, SpikeInfo, this_unit_col, zoom=zoom, save=outpath
        )

    # TODO docme
    break_flag = False

    if sf.check_convergence(
        SpikeInfo, it, n_hist, conv_crit
    ):  # refactor conv_crit into 'tol'
        logger.info('convergence criterion reached')

        # check for merges - if no merge - exit
        logger.info('checking for merges')
        Avgs, Sds = sf.calculate_pairwise_distances(
            Waveforms, SpikeInfo, this_unit_col, use_fr=use_fr, w=fr_weight
        )
        merge = sf.best_merge(Avgs, Sds, units, clust_alpha, exclude=rejected_merges)

        if force_merge:  # force merge
            while len(merge) == 0:
                clust_alpha += alpha_incr
                logger.info(f'increasing alpha to: {clust_alpha:.2f}')
                merge = sf.best_merge(
                    Avgs, Sds, units, clust_alpha, exclude=rejected_merges
                )

                # bail condition in case all possible merges are rejected manually by the user
                # an min numer is not reached yet
                if clust_alpha > max_alpha:
                    logger.critical(
                        'no more good merges, quitting before reaching number of desired clusters'
                    )
                    break_flag = True
                    break
        if break_flag:
            break

        if len(merge) > 0:
            if not manual_merge:
                logger.info(f'automatic merge: {merge[0]} with {merge[1]}')
                ix = SpikeInfo.groupby(this_unit_col).get_group(merge[1])['id']
                SpikeInfo.loc[ix, this_unit_col] = merge[0]
                # reset alpha
                clust_alpha = Config.getfloat('spike sort', 'clust_alpha')

            else:
                # show plots for this merge
                units = sf.get_units(SpikeInfo, this_unit_col)
                colors = sf.get_colors(units)
                for k, v in colors.items():
                    if k not in [str(m) for m in merge]:
                        colors[k] = 'gray'
                plt.ion()
                mpl.rcParams['figure.dpi'] = Config.get('output', 'screen_dpi')
                fig, axes = sp.plot_clustering(
                    Waveforms, SpikeInfo, this_unit_col, colors=colors
                )
                fig, axes = sp.plot_compare_waveforms(
                    Waveforms, SpikeInfo, this_unit_col, dt, merge
                )

                # ask for user input
                if input(f'merge {merge[0]} with {merge[1]} (Y/N)?') == 'Y':
                    # the merge
                    ix = SpikeInfo.groupby(this_unit_col).get_group(merge[1])['id']
                    SpikeInfo.loc[ix, this_unit_col] = merge[0]
                    # reset alpha
                    clust_alpha = Config.getfloat('spike sort', 'clust_alpha')
                    logger.info(f'manually accepted merge: {merge[0]} with {merge[1]}')
                else:
                    # if no, add merge to the list of forbidden merges
                    rejected_merges.append(merge)
                    logger.info(f'manually rejected merge: {merge[0]} with {merge[1]}')

                mpl.rcParams['figure.dpi'] = Config.get('output', 'fig_dpi')
                plt.close('all')
                plt.ioff()

        else:
            if not force_merge:
                logger.info('aborting, no more merges')
                break

    if force_merge:
        if (
            len(sf.get_units(SpikeInfo, this_unit_col, remove_unassigned=True))
            == n_clust_final
        ):
            logger.info(f'aborting, desired number of {n_clust_final} clusters reached')
            break

"""
 
 ######## ######## ########  ##     ## #### ##    ##    ###    ######## ######## 
    ##    ##       ##     ## ###   ###  ##  ###   ##   ## ##      ##    ##       
    ##    ##       ##     ## #### ####  ##  ####  ##  ##   ##     ##    ##       
    ##    ######   ########  ## ### ##  ##  ## ## ## ##     ##    ##    ######   
    ##    ##       ##   ##   ##     ##  ##  ##  #### #########    ##    ##       
    ##    ##       ##    ##  ##     ##  ##  ##   ### ##     ##    ##    ##       
    ##    ######## ##     ## ##     ## #### ##    ## ##     ##    ##    ######## 
 
"""

logger.info(' - finishing up - ')

for i in range(2):
    last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
    final_unit_col = 'unit_%i' % (int(last_unit_col.split('_')[1]) + 1)

    # final calculation of frate fast
    sf.calc_update_frates(SpikeInfo, last_unit_col, kernel_fast, kernel_slow)

    # train final models
    Models = sf.train_Models(SpikeInfo, last_unit_col, Waveforms, n_comp=n_model_comp)
    outpath = (plots_folder / 'Models_final').with_suffix(fig_format)
    sp.plot_Models(Models, save=outpath)

    # final scoring and assignment
    Scores, units = sf.Score_spikes(
        Waveforms,
        SpikeInfo,
        last_unit_col,
        Models,
        score_metric=sf.Rss,
        reassign_penalty=reassign_penalty,
        noise_penalty=noise_penalty,
    )

    # assign new labels
    min_ix = np.argmin(Scores, axis=1)
    new_labels = np.array([units[i] for i in min_ix], dtype='U')
    SpikeInfo[final_unit_col] = new_labels

    # clean assignment
    SpikeInfo = sf.reject_unit(SpikeInfo, final_unit_col)
    sf.reject_spikes(Waveforms, SpikeInfo, final_unit_col)

# - algo done -
logger.info(' - saving results - ')

# plot convergence
outpath = (plots_folder / 'Convergence_Rss').with_suffix(fig_format)
sp.plot_convergence(ScoresSum, save=outpath)

outpath = (plots_folder / 'Convergence_AIC').with_suffix(fig_format)
sp.plot_convergence(AICs, save=outpath)

# plot final clustering
outpath = (plots_folder / 'Clustering_all').with_suffix(fig_format)
sp.plot_clustering(Waveforms, SpikeInfo, final_unit_col, save=outpath, N=100)
units = sf.get_units(SpikeInfo, final_unit_col)
for unit in units:
    outpath = (plots_folder / f'Clustering_{unit}').with_suffix(fig_format)
    sp.plot_clustering(
        Waveforms, SpikeInfo, final_unit_col, color_by=unit, save=outpath, N=2000
    )

# plotting waveforms
outpath = (plots_folder / 'waveforms_final').with_suffix(fig_format)
sp.plot_waveforms(
    Waveforms,
    SpikeInfo,
    dt.rescale(pq.ms).magnitude,
    final_unit_col,
    N=100,
    save=outpath,
)

# update spike labels
kernel = ele.kernels.GaussianKernel(sigma=kernel_fast * pq.s)

for i, seg in enumerate(Blk.segments):
    spike_labels = SpikeInfo.groupby(('segment')).get_group((i))[final_unit_col].values
    (SpikeTrain,) = sf.select_by_dict(seg.spiketrains, kind='all_spikes')
    SpikeTrain.annotations['unit_labels'] = list(spike_labels)

    # make spiketrains
    spike_labels = SpikeTrain.annotations['unit_labels']
    sts = [SpikeTrain]
    for unit in units:
        times = SpikeTrain.times[np.array(spike_labels) == unit]
        st = neo.core.SpikeTrain(
            times, t_start=SpikeTrain.t_start, t_stop=SpikeTrain.t_stop
        )
        st.annotate(unit=unit)
        sts.append(st)
    seg.spiketrains = sts

    # est firing rates
    asigs = [seg.analogsignals[0]]
    for unit in units:
        (St,) = sf.select_by_dict(seg.spiketrains, unit=unit)
        frate = ele.statistics.instantaneous_rate(
            St, kernel=kernel, sampling_period=1 / fs
        )
        frate.annotate(kind='frate_fast', unit=unit)
        asigs.append(frate)
    seg.analogsignals = asigs

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
logger.info(f'saving SpikeInfo to {outpath}')
SpikeInfo.to_csv(outpath)

# store Block
outpath = results_folder / 'result.dill'
logger.info(f'saving Blk as .dill to {outpath}')
sssio.blk2dill(Blk, outpath)

# store models
outpath = results_folder / 'Models.dill'
logger.info(f'saving Models as .dill to {outpath}')
with open(outpath, 'wb') as fH:
    dill.dump(Models, fH)

# output csv data
if Config.getboolean('output', 'csv'):
    logger.info('writing csv')

    # SpikeTimes
    for i, Seg in enumerate(Blk.segments):
        title = Path(Seg.annotations['filename']).stem
        for j, unit in enumerate(units):
            (St,) = sf.select_by_dict(Seg.spiketrains, unit=unit)
            outpath = results_folder / (
                f'Segment_{seg_name}_unit_{unit}_spike_times.txt'
            )
            np.savetxt(outpath, St.times.magnitude)

    # firing rates - full res
    for i, Seg in enumerate(Blk.segments):
        FratesDf = pd.DataFrame()
        title = Path(Seg.annotations['filename']).stem
        for j, unit in enumerate(units):
            (asig,) = sf.select_by_dict(Seg.analogsignals, kind='frate_fast', unit=unit)
            FratesDf['t'] = asig.times.magnitude
            FratesDf[unit] = asig.magnitude.flatten()

        outpath = results_folder / f'Segment_{seg_name}_frates.csv'
        FratesDf.to_csv(outpath)

logger.info('all data is stored')

"""
 
 ########  ##        #######  ########    #### ##    ##  ######  ########  ########  ######  ######## 
 ##     ## ##       ##     ##    ##        ##  ###   ## ##    ## ##     ## ##       ##    ##    ##    
 ##     ## ##       ##     ##    ##        ##  ####  ## ##       ##     ## ##       ##          ##    
 ########  ##       ##     ##    ##        ##  ## ## ##  ######  ########  ######   ##          ##    
 ##        ##       ##     ##    ##        ##  ##  ####       ## ##        ##       ##          ##    
 ##        ##       ##     ##    ##        ##  ##   ### ##    ## ##        ##       ##    ##    ##    
 ##        ########  #######     ##       #### ##    ##  ######  ##        ########  ######     ##    
 
"""

logger.info(' - making diagnostic plots - ')
# plot all sorted spikes
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = (plots_folder / f'{seg_name}_overview').with_suffix(fig_format)
    sp.plot_segment(Seg, units, save=outpath)

# plot all sorted spikes
zoom = np.array(Config.get('output', 'zoom').split(','), dtype='float32') / 1000
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = (plots_folder / f'{seg_name}_fitted_spikes').with_suffix(fig_format)
    sp.plot_fitted_spikes(
        Seg, j, Models, SpikeInfo, final_unit_col, wsize, zoom=zoom, save=outpath
    )

# plot final models
outpath = (plots_folder / f'{seg_name}_models_final').with_suffix(fig_format)
sp.plot_Models(Models, save=outpath)

# TODO deal with this here too
if extense_plot:
    logger.info('creating plots')
    units = sf.get_units(SpikeInfo, this_unit_col)
    colors = sf.get_colors(units)
    outpath = detected_folder / ('overview' + fig_format)
    sp.plot_segment(seg, units, save=outpath, colors=colors)
    spike_label_interval = Config.getint('output', 'spike_label_interval')
    max_window = Config.getfloat('output', 'max_window_fitted_spikes_overview')
    sp.plot_fitted_spikes_complete(
        seg,
        Models,
        SpikeInfo,
        final_unit_col,
        max_window,
        detected_folder,
        fig_format,
        wsize=n_samples,
        extension='_templates',
        spike_label_interval=spike_label_interval,
        colors=colors,
    )

logger.info('plotting done')
logger.info('all tasks done - quitting')
sys.exit()

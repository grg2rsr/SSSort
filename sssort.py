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

# ephys
import neo
import elephant as ele

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# own
from functions import *
from plotters import *
import sssio


print("This is SSSort v1.0.0")
print("author: Georg Raiser - grg2rsr@gmail.com")

"""
 
 #### ##    ## #### 
  ##  ###   ##  ##  
  ##  ####  ##  ##  
  ##  ## ## ##  ##  
  ##  ##  ####  ##  
  ##  ##   ###  ##  
 #### ##    ## #### 
 
"""

# get config
if len(sys.argv) == 1:
    config_path = Path("./example_config.ini")
else:
    config_path = Path(os.path.abspath(sys.argv[1]))

Config = configparser.ConfigParser()
Config.read(config_path)

# handling paths and creating output directory
data_path = Path(Config.get('path', 'data_path'))
if not data_path.is_absolute():
    # ToDo when starts by ~ is not absolute.
    data_path = config_path.parent / data_path

exp_name = Config.get('path', 'experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'
os.makedirs(plots_folder, exist_ok=True)
os.chdir(config_path.parent / exp_name)


logger = sssio.get_logger(exp_name)



logger.info('config file read from %s' % config_path)

# copy config
shutil.copyfile(config_path, config_path.parent / exp_name / config_path.name)

# read data
Blk = sssio.get_data(data_path)
Blk.name = exp_name
logger.info('data read from %s' % data_path)

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
logger.info("highpass filtering data at %.2f Hz" % freq)
for seg in Blk.segments:
    seg.analogsignals[0] = ele.signal_processing.butter(seg.analogsignals[0], highpass_freq=freq)

if Config.getboolean('preprocessing', 'z_score'):
    logger.info("z-scoring signals")
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

global_mad = np.average([MAD(seg.analogsignals[0]) for seg in Blk.segments])
mad_thresh = Config.getfloat('spike detect', 'amplitude')
min_prominence = Config.getfloat('spike detect', 'min_prominence')
if min_prominence == 0:
    min_prominence = None
wsize = Config.getint('spike detect', 'wsize') * pq.ms
spike_detect_only = Config.getboolean('spike detect', 'spike_detect_only')

logger.info("amplitude was %f, global_mad is %f, used mad is: %f"%(mad_thresh, global_mad, mad_thresh*global_mad))

# # if only spike detection: diagnostic plot and and quit
# if spike_detect_only:
#     j = np.random.randint(len(Blk.segments))
#     seg = Blk.segments[j]  # select a segment at random
#     AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
#     plt.ion()
#     plot_spike_detect(AnalogSignal, min_prominence, N=5, w=0.35 * pq.s)
#     logger.info("only spike detection - press enter to quit")
#     input()  # halt terminal here
#     sys.exit()

for i, seg in enumerate(Blk.segments):
    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')

    # inverting
    if Config.get('spike detect', 'peak_mode') == 'negative':
        AnalogSignal *= -1

    # spike detection
    st = spike_detect(AnalogSignal, global_mad * mad_thresh, min_prominence)
    st.annotate(kind='all_spikes')

    if len(st) == 0:
        logger.error("No spikes detected, please enter check the threshold values in the configuration file")
        exit()

    # remove border spikes
    st_cut = st.time_slice(st.t_start + wsize / 2, st.t_stop - wsize / 2)
    st_cut.t_start = st.t_start
    seg.spiketrains.append(st_cut)
   
    # if only spike detection: diagnostic plot and and quit
    if spike_detect_only:
        # j = np.random.randint(len(Blk.segments))
        # seg = Blk.segments[j]  # select a segment at random
        # AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
        # plt.ion()
        # plot_spike_detect(AnalogSignal, min_prominence, N=5, w=0.35 * pq.s)

        #Plot detected spikes
        namepath = plots_folder / ("first_spike_detection_%d"%i)
        plot_spike_events(seg, min_prominence=min_prominence, thres=MAD(AnalogSignal)*mad_thresh,save=namepath,save_format=fig_format,max_window=0.4,max_row=3)

        logger.info("detected spikes plotted")

        logger.info("only spike detection - press enter to quit")
        input()  # halt terminal here
        sys.exit()


n_spikes = np.sum([seg.spiketrains[0].shape[0] for seg in Blk.segments])
logger.info("total number of detected spikes: %i" % n_spikes)

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

    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
    data = AnalogSignal.magnitude.flatten()

    SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
    inds = (SpikeTrain.times * fs).simplified.magnitude.astype('int32')

    waveforms.append(get_Waveforms(data, inds, n_samples))

Waveforms = np.concatenate(waveforms, axis=1)

# waveforms to disk
outpath = results_folder / 'Waveforms.npy'
np.save(outpath, Waveforms)
logger.info("saving spike waveforms to %s" % outpath)

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
logger.info("initial k-means with %i clusters" % n_clusters_init)

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
    SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
    spike_times.append(SpikeTrain.times.magnitude)
spike_times = np.concatenate(spike_times)
SpikeInfo['time'] = spike_times

# get segment labels
segment_labels = []
for i, seg in enumerate(Blk.segments):
    SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
    segment_labels.append(SpikeTrain.shape[0] * [i])
segment_labels = np.concatenate(segment_labels)
SpikeInfo['segment'] = segment_labels

# get all labels
SpikeInfo['unit_0'] = spike_labels

# get clean waveforms
n_neighbors = Config.getint('spike model', 'template_reject')
SpikeInfo = reject_spikes(Waveforms, SpikeInfo, 'unit_0', n_neighbors, verbose=True)

# unassign spikes if unit has too little good spikes
SpikeInfo = reject_unit(SpikeInfo, 'unit_0')

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
calc_update_frates(SpikeInfo, 'unit_0', kernel_fast, kernel_slow)

# model
n_model_comp = Config.getint('spike model', 'n_model_comp')
Models = train_Models(SpikeInfo, 'unit_0', Waveforms, n_comp=n_model_comp)
outpath = plots_folder / ("Models_ini" + fig_format)
plot_Models(Models, save=outpath)

# plot waveforms
outpath = plots_folder / ("waveforms_init" + fig_format)
plot_waveforms(Waveforms, SpikeInfo, dt.rescale(pq.ms).magnitude, unit_column='unit_0', N=100, save=outpath)

"""
 
 #### ######## ######## ########     ###    ######## ######## 
  ##     ##    ##       ##     ##   ## ##      ##    ##       
  ##     ##    ##       ##     ##  ##   ##     ##    ##       
  ##     ##    ######   ########  ##     ##    ##    ######   
  ##     ##    ##       ##   ##   #########    ##    ##       
  ##     ##    ##       ##    ##  ##     ##    ##    ##       
 ####    ##    ######## ##     ## ##     ##    ##    ######## 
 
"""

units = get_units(SpikeInfo, 'unit_0')
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

# hardcoded parameters
alpha_incr = 0.05
max_alpha = 100

rejected_merges = []
ScoresSum = []
AICs = []

for it in range(1, n_max_iter):

    # unit columns
    prev_unit_col = 'unit_%i' % (it - 1)
    this_unit_col = 'unit_%i' % it

    if it > 2:
        # randomly unassign a fraction of spikes
        N = int(n_spikes * sorting_noise)
        SpikeInfo.loc[SpikeInfo.sample(N).index, prev_unit_col] = '-1'

    # update rates
    calc_update_frates(SpikeInfo, prev_unit_col, kernel_fast, kernel_slow)

    # train models with labels from last iteration
    Models = train_Models(SpikeInfo, prev_unit_col, Waveforms, n_comp=n_model_comp)
    outpath = plots_folder / ("Models_%s%s" % (prev_unit_col, fig_format))
    plot_Models(Models, save=outpath)

    # Score spikes with models
    Scores, units = Score_spikes(Waveforms, SpikeInfo, prev_unit_col, Models, score_metric=Rss,
                                 reassign_penalty=reassign_penalty, noise_penalty=noise_penalty)

    # assign new labels
    min_ix = np.argmin(Scores, axis=1)
    SpikeInfo[this_unit_col] = np.array([units[i] for i in min_ix], dtype='U')

    # clean assignment
    SpikeInfo = reject_spikes(Waveforms, SpikeInfo, this_unit_col)
    SpikeInfo = reject_unit(SpikeInfo, this_unit_col)

    # plot waveforms
    outpath = plots_folder / ("Waveforms_%s%s" % (this_unit_col, fig_format))
    plot_waveforms(Waveforms, SpikeInfo, dt.rescale(pq.ms).magnitude, this_unit_col, N=100, save=outpath)

    # Model eval
    valid_ix = np.where(SpikeInfo[this_unit_col] != '-1')[0]
    Rss_sum = np.sum(np.min(Scores[valid_ix], axis=1)) / valid_ix.shape[0]  # Waveforms.shape[1]
    ScoresSum.append(Rss_sum)
    units = get_units(SpikeInfo, this_unit_col)
    AICs.append(len(units) - 2 * np.log(Rss_sum))

    # print iteration info
    n_changes, _ = get_changes(SpikeInfo, this_unit_col)
    logger.info("Iteration: %i - Error: %.2e - # reassigned spikes: %s" % (it, Rss_sum, n_changes))

    if check_convergence(SpikeInfo, it, n_hist, conv_crit):  # refactor conv_crit into 'tol'

        logger.info("convergence criterion reached")

        # check for merges - if no merge - exit
        logger.info("checking for merges")
        Avgs, Sds = calculate_pairwise_distances(Waveforms, SpikeInfo, this_unit_col)
        merge = best_merge(Avgs, Sds, units, clust_alpha, exclude=rejected_merges)

        if force_merge:  # force merge
            while len(merge) == 0:
                clust_alpha += alpha_incr
                logger.info("increasing alpha to: %.2f" % clust_alpha)
                merge = best_merge(Avgs, Sds, units, clust_alpha, exclude=rejected_merges)

                # bail condition in case all possible merges are rejected manually by the user
                # an min numer is not reached yet
                if clust_alpha > max_alpha:
                    logger.critical("no more good merges, quitting before reaching number of desired clusters")
                    break

        if len(merge) > 0:
            # do_merge(merge)
            if not manual_merge:
                logger.info("automatic merge: %s with %s" % tuple(merge))
                ix = SpikeInfo.groupby(this_unit_col).get_group(merge[1])['id']
                SpikeInfo.loc[ix, this_unit_col] = merge[0]
            else:
                # show plots for this merge
                units = get_units(SpikeInfo, this_unit_col)
                colors = get_colors(units)
                for k, v in colors.items():
                    if k not in [str(m) for m in merge]:
                        colors[k] = 'gray'
                plt.ion()
                fig, axes = plot_clustering(Waveforms, SpikeInfo, this_unit_col, colors=colors)
                # fig, axes = plot_clustering(Waveforms, SpikeInfo, unit_column, color_by=None, n_components=4, N=300, save=None, colors=None, unit_order=None)
                fig, axes = plot_compare_waveforms(Waveforms, SpikeInfo, this_unit_col, dt, merge)

                # ask for user input
                if input("merge %s with %s (Y/N)?" % tuple(merge)).upper() == 'Y':
                    # the merge
                    ix = SpikeInfo.groupby(this_unit_col).get_group(merge[1])['id']
                    SpikeInfo.loc[ix, this_unit_col] = merge[0]
                    logger.info("manually accepted merge: %s with %s" % tuple(merge))
                else:
                    # if no, add merge to the list of forbidden merges
                    rejected_merges.append(merge)
                    logger.info("manually rejected merge: %s with %s" % tuple(merge))

                plt.close('all')
                plt.ioff()

        else:
            if not force_merge:
                logger.info("aborting, no more merges")
                break

    if force_merge:
        if len(get_units(SpikeInfo, this_unit_col, remove_unassigned=True)) == n_clust_final:
            logger.info("aborting, desired number of %i clusters reached" % n_clust_final)
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

logger.info(" - finishing up - ")

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
final_unit_col = 'unit_%i' % (int(last_unit_col.split('_')[1]) + 1)

# final calculation of frate fast
calc_update_frates(SpikeInfo, last_unit_col, kernel_fast, kernel_slow)

# train final models
Models = train_Models(SpikeInfo, last_unit_col, Waveforms, n_comp=n_model_comp)
outpath = plots_folder / ("Models_final%s" % fig_format)
plot_Models(Models, save=outpath)

# final scoring and assingment
Scores, units = Score_spikes(Waveforms, SpikeInfo, last_unit_col, Models, score_metric=Rss,
                             reassign_penalty=reassign_penalty, noise_penalty=noise_penalty)

# assign new labels
min_ix = np.argmin(Scores, axis=1)
new_labels = np.array([units[i] for i in min_ix], dtype='U')
SpikeInfo[final_unit_col] = new_labels

# clean assignment
SpikeInfo = reject_unit(SpikeInfo, final_unit_col)
reject_spikes(Waveforms, SpikeInfo, final_unit_col)

# - algo done -

logger.info(" - saving results - ")

# plot convergence
outpath = plots_folder / ("Convergence_Rss" + fig_format)
plot_convergence(ScoresSum, save=outpath)

outpath = plots_folder / ("Convergence_AIC" + fig_format)
plot_convergence(AICs, save=outpath)

# plot final clustering
outpath = plots_folder / ("Clustering_all" + fig_format)
plot_clustering(Waveforms, SpikeInfo, final_unit_col, save=outpath, N=2000)
units = get_units(SpikeInfo, final_unit_col)
for unit in units:
    outpath = plots_folder / ("Clustering_%s%s" % (unit, fig_format))
    plot_clustering(Waveforms, SpikeInfo, final_unit_col, color_by=unit, save=outpath, N=2000)

# plotting waveforms
outpath = plots_folder / ("waveforms_final" + fig_format)
plot_waveforms(Waveforms, SpikeInfo, dt.rescale(pq.ms).magnitude, this_unit_col, N=100, save=outpath)

# update spike labels
kernel = ele.kernels.GaussianKernel(sigma=kernel_fast * pq.s)

for i, seg in enumerate(Blk.segments):
    spike_labels = SpikeInfo.groupby(('segment')).get_group((i))[final_unit_col].values
    SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
    SpikeTrain.annotations['unit_labels'] = list(spike_labels)

    # make spiketrains
    spike_labels = SpikeTrain.annotations['unit_labels']
    sts = [SpikeTrain]
    for unit in units:
        times = SpikeTrain.times[np.array(spike_labels) == unit]
        st = neo.core.SpikeTrain(times, t_start=SpikeTrain.t_start, t_stop=SpikeTrain.t_stop)
        st.annotate(unit=unit)
        sts.append(st)
    seg.spiketrains = sts

    # est firing rates
    asigs = [seg.analogsignals[0]]
    for unit in units:
        St, = select_by_dict(seg.spiketrains, unit=unit)
        frate = ele.statistics.instantaneous_rate(St, kernel=kernel, sampling_period=1 / fs)
        frate.annotate(kind='frate_fast', unit=unit)
        asigs.append(frate)
    seg.analogsignals = asigs

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
logger.info("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)

# store Block
outpath = results_folder / 'result.dill'
logger.info("saving Blk as .dill to %s" % outpath)
sssio.blk2dill(Blk, outpath)

# store models
outpath = results_folder / 'Models.dill'
logger.info("saving Models as .dill to %s" % outpath)
with open(outpath, 'wb') as fH:
    dill.dump(Models, fH)

# output csv data
if Config.getboolean('output', 'csv'):
    logger.info("writing csv")

    # SpikeTimes
    for i, Seg in enumerate(Blk.segments):
        try:
            title = Path(Seg.annotations['filename']).stem
        except:
            title = 'Segment %s'%(Seg.name)
        for j, unit in enumerate(units):
            St, = select_by_dict(Seg.spiketrains, unit=unit)
            outpath = results_folder / ("Segment_%s_unit_%s_spike_times.txt" % (seg_name, unit))
            np.savetxt(outpath, St.times.magnitude)

    # firing rates - full res
    for i, Seg in enumerate(Blk.segments):
        FratesDf = pd.DataFrame()
        try:
            title = Path(Seg.annotations['filename']).stem
        except:
            title = 'Segment %s'%(Seg.name)
        for j, unit in enumerate(units):
            asig, = select_by_dict(Seg.analogsignals, kind='frate_fast', unit=unit)
            FratesDf['t'] = asig.times.magnitude
            FratesDf[unit] = asig.magnitude.flatten()

        outpath = results_folder / ("Segment_%s_frates.csv" % seg_name)
        FratesDf.to_csv(outpath)

logger.info("all data is stored")

"""
 
 ########  ##        #######  ########    #### ##    ##  ######  ########  ########  ######  ######## 
 ##     ## ##       ##     ##    ##        ##  ###   ## ##    ## ##     ## ##       ##    ##    ##    
 ##     ## ##       ##     ##    ##        ##  ####  ## ##       ##     ## ##       ##          ##    
 ########  ##       ##     ##    ##        ##  ## ## ##  ######  ########  ######   ##          ##    
 ##        ##       ##     ##    ##        ##  ##  ####       ## ##        ##       ##          ##    
 ##        ##       ##     ##    ##        ##  ##   ### ##    ## ##        ##       ##    ##    ##    
 ##        ########  #######     ##       #### ##    ##  ######  ##        ########  ######     ##    
 
"""

logger.info(" - making diagnostic plots - ")
# plot all sorted spikes
for j, Seg in enumerate(Blk.segments):
    try:
        seg_name = Path(Seg.annotations['filename']).stem
    except:
        seg_name = 'Segment %s'%(Seg.name)
    outpath = plots_folder / (seg_name + '_overview' + fig_format)
    plot_segment(Seg, units, save=outpath)

# plot all sorted spikes
zoom = np.array(Config.get('output', 'zoom').split(','), dtype='float32') / 1000
for j, Seg in enumerate(Blk.segments):
    try:
        seg_name = Path(Seg.annotations['filename']).stem
    except:
        seg_name = 'Segment %s'%(Seg.name)
    outpath = plots_folder / (seg_name + '_fitted_spikes' + fig_format)
    plot_fitted_spikes(Seg, j, Models, SpikeInfo, final_unit_col, wsize, zoom=zoom, save=outpath)

# plot final models
outpath = plots_folder / (seg_name + '_models_final' + fig_format)
plot_Models(Models, save=outpath)
logger.info("all plotting done")


logger.info("all tasks done - quitting")
sys.exit()

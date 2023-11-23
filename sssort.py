# sys
import sys
import os
import copy
import dill
import shutil
import configparser
from pathlib import Path
from tqdm import tqdm

# sci
import scipy as sp
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
import seaborn as sns

# own
from functions import *
from plotters import *
import sssio

# logging
import logging

log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
date_fmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(log_fmt, datefmt=date_fmt)

# for printing to stdout
logger = logging.getLogger() # get all loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('functions').setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

# logging unhandled exceptions
def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    # TODO make this cleaner that it doesn't use global namespace
    logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_unhandled_exception

# banner
# if os.name == "posix":
#     tp.banner("This is SSSort v1.0.0", 78)
#     tp.banner("author: Georg Raiser - grg2rsr@gmail.com", 78)
# else:
#     print("This is SSSort v1.0.0")
#     print("author: Georg Raiser - grg2rsr@gmail.com")

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
# config_path = Path(os.path.abspath(sys.argv[1]))
config_path = Path("/home/georg/code/SSSort/example_config_testing_termin_criteria.ini")
Config = configparser.ConfigParser()
Config.read(config_path)
logger.info('config file read from %s' % config_path)

# handling paths and creating output directory
data_path = Path(Config.get('path','data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path','experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'
os.makedirs(plots_folder, exist_ok=True)
os.chdir(config_path.parent / exp_name)

# config logger for writing to file
file_handler = logging.FileHandler(filename="%s.log" % exp_name, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('config file read from %s' % config_path)

# copy config
shutil.copyfile(config_path, config_path.parent / exp_name / config_path.name)

# read data
Blk = sssio.get_data(data_path)
Blk.name = exp_name
logger.info('data read from %s' % data_path)

# get data properies
fs = Blk.segments[0].analogsignals[0].sampling_rate
dt = (1/fs).rescale(pq.s)

# plotting
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')

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
freq = Config.getfloat('preprocessing','highpass_freq')
logger.info("highpass filtering data at %.2f Hz" % freq)
for seg in Blk.segments:
    seg.analogsignals[0] = ele.signal_processing.butter(seg.analogsignals[0], highpass_freq=freq)

if Config.getboolean('preprocessing','z_score'):
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

# global mad
global_mad = np.average([MAD(seg.analogsignals[0]) for seg in Blk.segments])
mad_thresh = Config.getfloat('spike detect', 'amplitude')
min_prominence = Config.getfloat('spike detect', 'min_prominence')
wsize = Config.getfloat('spike detect', 'wsize') * pq.ms
spike_detect_only = Config.getboolean('spike detect','spike_detect_only')

min_prominence = 5

if spike_detect_only:
    outpath = None
    plt.ion()
    seg = Blk.segments[0]
    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
    plot_spike_detect(AnalogSignal, min_prominence, N=5, w=0.35*pq.s, save=None)
    logger.info("only spike detection - press enter to quit")
    input()
    sys.exit()

# outpath = plots_folder / ("spike_detect" + fig_format)
# plot_spike_detect(AnalogSignal, SpikeTrain, 5, w=30*pq.ms, save=outpath)
# plt.show()
# sys.exit()



for i, seg in enumerate(Blk.segments):
    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')

    # inverting
    if Config.get('spike detect','peak_mode') == 'negative':
        AnalogSignal *= -1

    # spike detection
    st = spike_detect(AnalogSignal, global_mad*mad_thresh, min_prominence)
    st.annotate(kind='all_spikes')

    # remove border spikes
    st_cut = st.time_slice(st.t_start + wsize/2, st.t_stop - wsize/2)
    st_cut.t_start = st.t_start
    seg.spiketrains.append(st_cut)

n_spikes = np.sum([seg.spiketrains[0].shape[0] for seg in Blk.segments])
logger.info("total number of detected spikes: %i" % n_spikes)

seg = Blk.segments[0]
AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')

"""
 
 ######## ######## ##     ## ########  ##          ###    ######## ########  ######  
    ##    ##       ###   ### ##     ## ##         ## ##      ##    ##       ##    ## 
    ##    ##       #### #### ##     ## ##        ##   ##     ##    ##       ##       
    ##    ######   ## ### ## ########  ##       ##     ##    ##    ######    ######  
    ##    ##       ##     ## ##        ##       #########    ##    ##             ## 
    ##    ##       ##     ## ##        ##       ##     ##    ##    ##       ##    ## 
    ##    ######## ##     ## ##        ######## ##     ##    ##    ########  ######  
 
"""

logger.info(' - getting templates - ')

fs = Blk.segments[0].analogsignals[0].sampling_rate
n_samples = (wsize * fs).simplified.magnitude.astype('int32')

templates = []
for j, seg in enumerate(Blk.segments):

    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
    data = AnalogSignal.magnitude.flatten()

    SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
    inds = (SpikeTrain.times * fs).simplified.magnitude.astype('int32')

    templates.append(get_Templates(data, inds, n_samples))

Templates = np.concatenate(templates, axis=1)

# templates to disk
outpath = results_folder / 'Templates.npy'
np.save(outpath, Templates)
logger.info("saving Templates to %s" % outpath)

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
logger.info("initial kmeans with %i clusters" % n_clusters_init)

# initial clustering in the same space as subsequent spikes models
n_model_comp = Config.getint('spike model', 'n_model_comp')
pca = PCA(n_components=n_model_comp)
X = pca.fit_transform(Templates.T)
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
n_spikes = Templates.shape[1]
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
SpikeInfo['unit'] = spike_labels

# get clean templates
n_neighbors = Config.getint('spike model', 'template_reject')
reject_spikes(Templates, SpikeInfo, 'unit', n_neighbors, verbose=True)

# unassign spikes if unit has too little good spikes
SpikeInfo = reject_unit(SpikeInfo, 'unit')

outpath = plots_folder / ("templates_init" + fig_format)
# fs = Blk.segments[0].analogsignals[0].sampling_rate
# dt = (1/fs).rescale(pq.ms).magnitude
plot_templates(Templates, SpikeInfo, dt.rescale(pq.ms).magnitude, N=100, save=outpath)


"""
 
 #### ##    ## #### ######## 
  ##  ###   ##  ##     ##    
  ##  ####  ##  ##     ##    
  ##  ## ## ##  ##     ##    
  ##  ##  ####  ##     ##    
  ##  ##   ###  ##     ##    
 #### ##    ## ####    ##    
 
"""
# first ini run
logger.info(' - initializing algorithm - ')

# rate est
kernel_slow = Config.getfloat('kernels', 'sigma_slow')
kernel_fast = Config.getfloat('kernels', 'sigma_fast')
calc_update_frates(SpikeInfo, 'unit', kernel_fast, kernel_slow)

# model
n_model_comp = Config.getint('spike model','n_model_comp')
Models = train_Models(SpikeInfo, 'unit', Templates, n_comp=n_model_comp)
outpath = plots_folder / ("Models_ini" + fig_format)
plot_Models(Models, save=outpath)

"""
 
 #### ######## ######## ########     ###    ######## ######## 
  ##     ##    ##       ##     ##   ## ##      ##    ##       
  ##     ##    ##       ##     ##  ##   ##     ##    ##       
  ##     ##    ######   ########  ##     ##    ##    ######   
  ##     ##    ##       ##   ##   #########    ##    ##       
  ##     ##    ##       ##    ##  ##     ##    ##    ##       
 ####    ##    ######## ##     ## ##     ##    ##    ######## 
 
"""

# reset
SpikeInfo['unit_0'] = SpikeInfo['unit'] # the init
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
    prev_unit_col = 'unit_%i' % (it-1)
    this_unit_col = 'unit_%i' % it

    if it > 2:
        # randomly unassign a fraction of spikes
        N = int(n_spikes * sorting_noise)
        SpikeInfo.loc[SpikeInfo.sample(N).index, prev_unit_col] = '-1'

    # update rates
    calc_update_frates(SpikeInfo, prev_unit_col, kernel_fast, kernel_slow)

    # train models with labels from last iteration
    Models = train_Models(SpikeInfo, prev_unit_col, Templates, n_comp=n_model_comp)
    outpath = plots_folder / ("Models_%s%s" % (prev_unit_col, fig_format))
    plot_Models(Models, save=outpath)

    # Score spikes with models
    Scores, units = Score_spikes(Templates, SpikeInfo, prev_unit_col, Models, score_metric=Rss,
                                 reassign_penalty=reassign_penalty, noise_penalty=noise_penalty)

    # assign new labels
    min_ix = np.argmin(Scores, axis=1)
    SpikeInfo[this_unit_col] = np.array([units[i] for i in min_ix], dtype='U')

    # clean assignment
    SpikeInfo = reject_spikes(Templates, SpikeInfo, this_unit_col)
    SpikeInfo = reject_unit(SpikeInfo, this_unit_col)
    
    # plot templates
    outpath = plots_folder / ("Templates_%s%s" % (this_unit_col, fig_format))
    plot_templates(Templates, SpikeInfo, dt.rescale(pq.ms).magnitude, this_unit_col, save=outpath)

    # Model eval
    valid_ix = np.where(SpikeInfo[this_unit_col] != '-1')[0]
    Rss_sum = np.sum(np.min(Scores[valid_ix], axis=1)) / valid_ix.shape[0] #Templates.shape[1]
    ScoresSum.append(Rss_sum)
    units = get_units(SpikeInfo, this_unit_col)
    AICs.append(len(units) - 2 * np.log(Rss_sum))

    # print iteration info
    n_changes, _ = get_changes(SpikeInfo, this_unit_col)
    logger.info("Iteration: %i - Error: %.2e - # reassigned spikes: %s" % (it, Rss_sum, n_changes))

    if check_convergence(SpikeInfo, it, n_hist, conv_crit): # refactor conv_crit into 'tol'

        logger.info("convergence criterion reached")

        # check for merges - if no merge - exit
        logger.info("checking for merges")
        Avgs, Sds = calculate_pairwise_distances(Templates, SpikeInfo, this_unit_col)
        merge = best_merge(Avgs, Sds, units, clust_alpha, exclude=rejected_merges)

        if force_merge:# force merge
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
                for k,v in colors.items():
                    if k not in [str(m) for m in merge]:
                        colors[k] = 'gray'
                plt.ion()
                fig, axes = plot_clustering(Templates, SpikeInfo, this_unit_col, colors=colors)
                fig, axes = plot_compare_templates(Templates, SpikeInfo, this_unit_col, dt, merge)

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
        if len(get_units(SpikeInfo, this_unit_col, remove_unassinged=True)) == n_clust_final:
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

# final scoring and assingment
Scores, units = Score_spikes(Templates, SpikeInfo, last_unit_col, Models, score_metric=Rss,
                                reassign_penalty=reassign_penalty, noise_penalty=noise_penalty)

# assign new labels
min_ix = np.argmin(Scores, axis=1)
new_labels = np.array([units[i] for i in min_ix], dtype='U')
SpikeInfo[final_unit_col] = new_labels

# clean assignment
SpikeInfo = reject_unit(SpikeInfo, final_unit_col)
reject_spikes(Templates, SpikeInfo, final_unit_col)

# - algo done - 

logger.info(" - saving results - ")

# plot
outpath = plots_folder / ("Convergence_Rss" + fig_format)
plot_convergence(ScoresSum, save=outpath)

outpath = plots_folder / ("Convergence_AIC" + fig_format)
plot_convergence(AICs, save=outpath)

outpath = plots_folder / ("Clustering" + fig_format)
plot_clustering(Templates, SpikeInfo, final_unit_col, save=outpath)

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
        st = neo.core.SpikeTrain(times, t_start = SpikeTrain.t_start, t_stop=SpikeTrain.t_stop)
        st.annotate(unit=unit)
        sts.append(st)
    seg.spiketrains = sts

    # est firing rates
    asigs = [seg.analogsignals[0]]
    for unit in units:
        St, = select_by_dict(seg.spiketrains, unit=unit)
        frate = ele.statistics.instantaneous_rate(St, kernel=kernel, sampling_period=1/fs)
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
if Config.getboolean('output','csv'):
    logger.info("writing csv")

    # SpikeTimes
    for i, Seg in enumerate(Blk.segments):
        seg_name = Path(Seg.annotations['filename']).stem
        for j, unit in enumerate(units):
            St, = select_by_dict(Seg.spiketrains, unit=unit)
            outpath = results_folder / ("Segment_%s_unit_%s_spike_times.txt" % (seg_name, unit))
            np.savetxt(outpath, St.times.magnitude)

    # firing rates - full res
    for i, Seg in enumerate(Blk.segments):
        FratesDf = pd.DataFrame()
        seg_name = Path(Seg.annotations['filename']).stem
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
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_overview' + fig_format)
    plot_segment(Seg, units, save=outpath)

# plot all sorted spikes
zoom = np.array(Config.get('output','zoom').split(','),dtype='float32') / 1000
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_fitted_spikes' + fig_format)
    plot_fitted_spikes(Seg, j, Models, SpikeInfo, final_unit_col, zoom=zoom, save=outpath)

# plot final models
outpath = plots_folder / (seg_name + '_models_final' + fig_format)
plot_Models(Models, save=outpath)
logger.info("all plotting done")


logger.info("all tasks done - quitting")
sys.exit()

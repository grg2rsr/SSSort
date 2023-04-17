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

# own
from functions import *
from plotters import *
import sssio

# banner
if os.name == "posix":
    tp.banner("This is SSSort v1.0.0", 78)
    tp.banner("author: Georg Raiser - grg2rsr@gmail.com", 78)
else:
    print("This is SSSort v1.0.0")
    print("author: Georg Raiser - grg2rsr@gmail.com")

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


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
config_path = Path(os.path.abspath(sys.argv[1]))
Config = configparser.ConfigParser()
Config.read(config_path)
print_msg('config file read from %s' % config_path)

# handling paths and creating output directory
data_path = Path(Config.get('path','data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path','experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'
os.makedirs(plots_folder, exist_ok=True)
os.chdir(config_path.parent / exp_name)

# copy config
shutil.copyfile(config_path, config_path.parent / exp_name / config_path.name)

# read data
Blk = sssio.get_data(data_path)
Blk.name = exp_name
print_msg('data read from %s' % data_path)

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

print_msg(' - preprocessing - ')

for seg in Blk.segments:
    seg.analogsignals[0].annotate(kind='original')

# highpass filter
freq = Config.getfloat('preprocessing','highpass_freq')
print_msg("highpass filtering data at %.2f Hz" % freq)
for seg in Blk.segments:
    seg.analogsignals[0] = ele.signal_processing.butter(seg.analogsignals[0], highpass_freq=freq)

if Config.getboolean('preprocessing','z_score'):
    print_msg("z-scoring signals")
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

print_msg('- spike detect - ')

# detecting all spikes by MAD thresholding
mad_thresh = Config.getfloat('spike detect', 'mad_thresh')

bad_segments = []
for i, seg in enumerate(Blk.segments):
    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')

    # invert
    if Config.get('spike detect','peak_mode') == 'negative': # TODO peak mode should be a spike detect parameter
        bounds = [-np.inf, MAD(AnalogSignal) * -mad_thresh]
    else:
        bounds = [MAD(AnalogSignal) * mad_thresh, np.inf]
        
    st = spike_detect(AnalogSignal, bounds * AnalogSignal.units)

    if st.times.shape[0] == 0:
        seg_name = Path(seg.annotations['filename']).stem
        print_msg("no spikes found for segment %i:%s" % (i, seg_name))
        bad_segments.append(i)
    st.annotate(kind='all_spikes')

    # remove border spikes
    wsize = Config.getfloat('spike detect', 'wsize') * pq.ms
    st_cut = st.time_slice(st.t_start + wsize/2, st.t_stop - wsize/2)
    st_cut.t_start = st.t_start
    seg.spiketrains.append(st_cut)

n_spikes = np.sum([seg.spiketrains[0].shape[0] for seg in Blk.segments])
print_msg("total number of spikes found: %s" % n_spikes)

"""
 
 ######## ######## ##     ## ########  ##          ###    ######## ########  ######  
    ##    ##       ###   ### ##     ## ##         ## ##      ##    ##       ##    ## 
    ##    ##       #### #### ##     ## ##        ##   ##     ##    ##       ##       
    ##    ######   ## ### ## ########  ##       ##     ##    ##    ######    ######  
    ##    ##       ##     ## ##        ##       #########    ##    ##             ## 
    ##    ##       ##     ## ##        ##       ##     ##    ##    ##       ##    ## 
    ##    ######## ##     ## ##        ######## ##     ##    ##    ########  ######  
 
"""

print_msg(' - getting templates - ')

fs = Blk.segments[0].analogsignals[0].sampling_rate
n_samples = (wsize * fs).simplified.magnitude.astype('int32')

templates = []
for j, seg in enumerate(Blk.segments):
    data = seg.analogsignals[0].magnitude.flatten()
    inds = (seg.spiketrains[0].times * fs).simplified.magnitude.astype('int32')
    templates.append(get_Templates(data, inds, n_samples))

Templates = np.concatenate(templates,axis=1)

# templates to disk
outpath = results_folder / 'Templates.npy'
np.save(outpath, Templates)
print_msg("saving Templates to %s" % outpath)

"""
 
  ######  ##       ##     ##  ######  ######## ######## ########  
 ##    ## ##       ##     ## ##    ##    ##    ##       ##     ## 
 ##       ##       ##     ## ##          ##    ##       ##     ## 
 ##       ##       ##     ##  ######     ##    ######   ########  
 ##       ##       ##     ##       ##    ##    ##       ##   ##   
 ##    ## ##       ##     ## ##    ##    ##    ##       ##    ##  
  ######  ########  #######   ######     ##    ######## ##     ## 
 
"""

n_clusters_init = Config.getint('spike sort','init_clusters')
print_msg("initial kmeans with %i clusters" % n_clusters_init)

# initial clustering in the same space as subsequent spikes models
n_model_comp = Config.getint('spike model','n_model_comp')
pca = PCA(n_components=n_model_comp)
X = pca.fit_transform(Templates.T)
kmeans_labels = KMeans(n_clusters=n_clusters_init).fit_predict(X)
spike_labels = kmeans_labels.astype('U')

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
SpikeInfo['id'] = np.arange(n_spikes,dtype='int32')

# get all spike times
spike_times = np.concatenate([seg.spiketrains[0].times.magnitude for seg in Blk.segments])
SpikeInfo['time'] = spike_times

# get segment labels
segment_labels = []
for i, seg in enumerate(Blk.segments):
    segment_labels.append(seg.spiketrains[0].shape[0] * [i])
segment_labels = np.concatenate(segment_labels)
SpikeInfo['segment'] = segment_labels

# get all labels
SpikeInfo['unit'] = spike_labels

# get clean templates
n_neighbors = Config.getint('spike model','template_reject')
reject_spikes(Templates, SpikeInfo, 'unit', n_neighbors, verbose=True)

# unassign spikes if unit has too little good spikes
SpikeInfo = unassign_spikes(SpikeInfo, 'unit')

outpath = plots_folder / ("templates_init" + fig_format)
plot_templates(Templates, SpikeInfo, N=100, save=outpath)


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
print_msg('- initializing algorithm: calculating all initial firing rates')

# rate est
kernel_slow = Config.getfloat('kernels','sigma_slow')
kernel_fast = Config.getfloat('kernels','sigma_fast')
calc_update_frates(Blk.segments, SpikeInfo, 'unit', kernel_fast, kernel_slow)

# model
n_model_comp = Config.getint('spike model','n_model_comp')
Models = train_Models(SpikeInfo, 'unit', Templates, n_comp=n_model_comp, verbose=False)
outpath = plots_folder / ("Models_ini" + fig_format)
plot_Models(Models, save=outpath)

"""
 
 ########  ##     ## ##    ## 
 ##     ## ##     ## ###   ## 
 ##     ## ##     ## ####  ## 
 ########  ##     ## ## ## ## 
 ##   ##   ##     ## ##  #### 
 ##    ##  ##     ## ##   ### 
 ##     ##  #######  ##    ## 
 
"""

# reset
SpikeInfo['unit_0'] = SpikeInfo['unit'] # the init

its = Config.getint('spike sort','iterations')
it_merge = Config.getint('spike sort','it_merge')
first_merge = Config.getint('spike sort','first_merge')
clust_alpha = Config.getfloat('spike sort','clust_alpha')
units = get_units(SpikeInfo, 'unit_0')
n_units = len(units)
penalty = Config.getfloat('spike sort','penalty')
sorting_noise = Config.getfloat('spike sort','f_noise')
ScoresSum = []
AICs = []

spike_ids = SpikeInfo['id'].values

for it in range(1,its):
    # unit columns
    prev_unit_col = 'unit_%i' % (it-1)
    this_unit_col = 'unit_%i' % it

    # update rates
    calc_update_frates(Blk.segments, SpikeInfo, prev_unit_col, kernel_fast, kernel_slow)

    # train models with labels from last iteration
    Models = train_Models(SpikeInfo, prev_unit_col, Templates, verbose=False, n_comp=n_model_comp)
    outpath = plots_folder / ("Models_%s%s" % (prev_unit_col, fig_format))
    plot_Models(Models, save=outpath)

    # Score spikes with models
    if it == its-1: # the last
        penalty = 0
    Scores, units = Score_spikes(Templates, SpikeInfo, prev_unit_col, Models, score_metric=Rss, penalty=penalty)

    # assign new labels
    min_ix = np.argmin(Scores, axis=1)
    new_labels = np.array([units[i] for i in min_ix],dtype='object')
    SpikeInfo[this_unit_col] = new_labels

    # clean assignment
    SpikeInfo = unassign_spikes(SpikeInfo, this_unit_col)
    reject_spikes(Templates, SpikeInfo, this_unit_col)

    # randomly unassign a fraction of spikes
    if it != its-1: # the last
        N = int(n_spikes * sorting_noise)
        SpikeInfo.loc[SpikeInfo.sample(N).index,this_unit_col] = '-1'
    
    # plot templates
    outpath = plots_folder / ("Templates_%s%s" % (this_unit_col, fig_format))
    plot_templates(Templates, SpikeInfo, this_unit_col, save=outpath)

    # every n iterations, merge
    if (it > first_merge) and (it % it_merge) == 0:
        print_msg("check for merges ... ")
        Avgs, Sds = calculate_pairwise_distances(Templates, SpikeInfo, this_unit_col)
        merge = best_merge(Avgs, Sds, units, clust_alpha)
        if len(merge) > 0:
            print_msg("merging: " + ' '.join(merge))
            ix = SpikeInfo.groupby(this_unit_col).get_group(merge[1])['id']
            SpikeInfo.loc[ix, this_unit_col] = merge[0]

    # Model eval
    n_changes = np.sum(~(SpikeInfo[this_unit_col] == SpikeInfo[prev_unit_col]).values)
    valid_ix = np.where(SpikeInfo[this_unit_col] != '-1')[0]
    
    Rss_sum = np.sum(np.min(Scores[valid_ix], axis=1)) / Templates.shape[1]
    ScoresSum.append(Rss_sum)
    units = get_units(SpikeInfo, this_unit_col)
    AICs.append(len(units) - 2 * np.log(Rss_sum))

    # print iteration info
    print_msg("It:%i - Rss sum: %.3e - # reassigned spikes: %s" % (it, Rss_sum, n_changes))

print_msg("algorithm run is done")

"""
 
 ######## #### ##    ## ####  ######  ##     ## 
 ##        ##  ###   ##  ##  ##    ## ##     ## 
 ##        ##  ####  ##  ##  ##       ##     ## 
 ######    ##  ## ## ##  ##   ######  ######### 
 ##        ##  ##  ####  ##        ## ##     ## 
 ##        ##  ##   ###  ##  ##    ## ##     ## 
 ##       #### ##    ## ####  ######  ##     ## 
 
"""

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]

# plot
outpath = plots_folder / ("Convergence_Rss" + fig_format)
plot_convergence(ScoresSum, save=outpath)

outpath = plots_folder / ("Convergence_AIC" + fig_format)
plot_convergence(AICs, save=outpath)

outpath = plots_folder / ("Clustering" + fig_format)
plot_clustering(Templates, SpikeInfo, last_unit_col, save=outpath)

# update spike labels
kernel = ele.kernels.GaussianKernel(sigma=kernel_fast * pq.s)
it = its-1 # the last

for i, seg in tqdm(enumerate(Blk.segments),desc="populating block for output"):
    spike_labels = SpikeInfo.groupby(('segment')).get_group((i))['unit_%i' % it].values
    seg.spiketrains[0].annotations['unit_labels'] = list(spike_labels)

    # make spiketrains
    St = seg.spiketrains[0]
    spike_labels = St.annotations['unit_labels']
    sts = [St]
    for unit in units:
        times = St.times[np.array(spike_labels) == unit]
        st = neo.core.SpikeTrain(times, t_start = St.t_start, t_stop=St.t_stop)
        st.annotate(unit=unit)
        sts.append(st)
    seg.spiketrains=sts

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
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)

# store Block
outpath = results_folder / 'result.dill'
print_msg("saving Blk as .dill to %s" % outpath)
sssio.blk2dill(Blk, outpath)

# store models
outpath = results_folder / 'Models.dill'
print_msg("saving Models as .dill to %s" % outpath)
with open(outpath, 'wb') as fH:
    dill.dump(Models, fH)

print_msg("data is stored")

# output csv data
if Config.getboolean('output','csv'):
    print_msg("writing csv")

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
            
        # firing rates - downsampled
        # tbins = np.arange(0,12.2,0.1)
        # FratesDf_ds = pd.DataFrame(columns=FratesDf.columns)
        # for i in range(1,tbins.shape[0]):
        #     t0 = tbins[i-1]
        #     t1 = tbins[i]
        #     ix = np.logical_and(FratesDf['t'] > t0,FratesDf['t'] < t1)
        #     FratesDf_ds = FratesDf_ds.append(FratesDf.iloc[ix.values].mean(axis=0),ignore_index=True)
        # FratesDf_ds['t'] = tbins[:-1]

        # outpath = results_folder / ("Segment_%s_frates_downsampled.csv" % seg_name)
        # FratesDf_ds.to_csv(outpath)
    

"""
 
 ########  ##        #######  ########    #### ##    ##  ######  ########  ########  ######  ######## 
 ##     ## ##       ##     ##    ##        ##  ###   ## ##    ## ##     ## ##       ##    ##    ##    
 ##     ## ##       ##     ##    ##        ##  ####  ## ##       ##     ## ##       ##          ##    
 ########  ##       ##     ##    ##        ##  ## ## ##  ######  ########  ######   ##          ##    
 ##        ##       ##     ##    ##        ##  ##  ####       ## ##        ##       ##          ##    
 ##        ##       ##     ##    ##        ##  ##   ### ##    ## ##        ##       ##    ##    ##    
 ##        ########  #######     ##       #### ##    ##  ######  ##        ########  ######     ##    
 
"""

# plot all sorted spikes
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_overview' + fig_format)
    plot_segment(Seg, units, save=outpath)

# plot all sorted spikes
zoom = np.array(Config.get('output','zoom').split(','),dtype='float32') / 1000
unit_column = 'unit_%i' % it
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_fitted_spikes' + fig_format)
    plot_fitted_spikes(Seg, j, Models, SpikeInfo, unit_column, zoom=zoom, save=outpath)

print_msg("plotting done")
print_msg("all done - quitting")

sys.exit()

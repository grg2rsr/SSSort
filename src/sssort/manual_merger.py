# sys
import sys
import os
import dill
import configparser
from pathlib import Path

# sci
import numpy as np
import pandas as pd
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
config_path = Path(sys.argv[1])
Config = configparser.ConfigParser()
Config.read(config_path)
# print('config file read from %s' % config_path)

# optional extra arg: unit column
if len(sys.argv) == 3:
    unit_column = sys.argv[2]

Config = configparser.ConfigParser()
Config.read(config_path)

# handling paths and creating output directory
data_path = Path(Config.get('path', 'data_path'))
# if not data_path.is_absolute():
#     data_path = config_path.parent / data_path

exp_name = Config.get('path', 'experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'

logger = sssio.get_logger(
    (config_path.parent / exp_name / exp_name).with_suffix('.log')
)
logger.info(' - manual merging - ')

os.chdir(results_folder)

"""
 
 ########  ########    ###    ########     ########     ###    ########    ###    
 ##     ## ##         ## ##   ##     ##    ##     ##   ## ##      ##      ## ##   
 ##     ## ##        ##   ##  ##     ##    ##     ##  ##   ##     ##     ##   ##  
 ########  ######   ##     ## ##     ##    ##     ## ##     ##    ##    ##     ## 
 ##   ##   ##       ######### ##     ##    ##     ## #########    ##    ######### 
 ##    ##  ##       ##     ## ##     ##    ##     ## ##     ##    ##    ##     ## 
 ##     ## ######## ##     ## ########     ########  ##     ##    ##    ##     ## 
 
"""
# read data
Blk = sssio.dill2blk(results_folder / 'result.dill')

# get data properies
fs = Blk.segments[0].analogsignals[0].sampling_rate
dt = (1 / fs).rescale(pq.s)

# loading SpikeInfo
SpikeInfo = pd.read_csv(results_folder / 'SpikeInfo.csv')
unit_columns = [col for col in SpikeInfo.columns if col.startswith('unit_')]
SpikeInfo[unit_columns] = SpikeInfo[unit_columns].astype(str)

# loading Models
with open(results_folder / 'Models.dill', 'rb') as fH:
    Models = dill.load(fH)

# loading Waveforms
Waveforms = np.load(results_folder / 'Waveforms.npy')

# plotting
mpl.rcParams['figure.dpi'] = Config.get('output', 'fig_dpi')
fig_format = Config.get('output', 'fig_format')

"""
 
 #### ######## ######## ########     ###    ######## ######## 
  ##     ##    ##       ##     ##   ## ##      ##    ##       
  ##     ##    ##       ##     ##  ##   ##     ##    ##       
  ##     ##    ######   ########  ##     ##    ##    ######   
  ##     ##    ##       ##   ##   #########    ##    ##       
  ##     ##    ##       ##    ##  ##     ##    ##    ##       
 ####    ##    ######## ##     ## ##     ##    ##    ######## 
 
"""
last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
if len(sys.argv) <= 2:
    this_unit_col = last_unit_col
else:
    this_unit_col = sys.argv[2]
    ix = np.where(SpikeInfo.columns == this_unit_col)[0][0]
    SpikeInfo = SpikeInfo.iloc[:, : ix + 1]

units = sf.get_units(SpikeInfo, this_unit_col)
n_units = len(units)

clust_alpha = Config.getfloat('spike sort', 'clust_alpha')
n_clust_final = Config.getint('spike sort', 'n_clust_final')

# hardcoded parameters
alpha_incr = 0.1
max_alpha = 1000

rejected_merges = []

n_clust = len(sf.get_units(SpikeInfo, this_unit_col, remove_unassigned=True))
break_flag = False

while n_clust < n_clust_final or clust_alpha < max_alpha:
    # check for merges - if no merge - exit
    units = sf.get_units(SpikeInfo, this_unit_col)
    Avgs, Sds = sf.calculate_pairwise_distances(Waveforms, SpikeInfo, this_unit_col)
    merge = sf.best_merge(Avgs, Sds, units, clust_alpha, exclude=rejected_merges)

    while len(merge) == 0:
        clust_alpha += alpha_incr
        logger.info('increasing alpha to: %.2f' % clust_alpha)
        merge = sf.best_merge(Avgs, Sds, units, clust_alpha, exclude=rejected_merges)

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
            logger.info('manually accepted merge: {merge[0]} with {merge[1]}')
            # resetting alpha
            clust_alpha = Config.getfloat('spike sort', 'clust_alpha')
        else:
            # if no, add merge to the list of forbidden merges
            rejected_merges.append(merge)
            logger.info(f'manually rejected merge: {merge[0]} with {merge[1]}')

        mpl.rcParams['figure.dpi'] = Config.get('output', 'fig_dpi')
        plt.close('all')
        plt.ioff()

    # exit condition if n_clusters is reached
    n_clust = len(sf.get_units(SpikeInfo, this_unit_col, remove_unassigned=True))
    if n_clust == n_clust_final:
        print(f'desired number of {n_clust_final} clusters reached')
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

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
final_unit_col = 'unit_%i' % (int(last_unit_col.split('_')[1]) + 1)

# final calculation of frate fast
kernel_slow = Config.getfloat('kernels', 'sigma_slow')
kernel_fast = Config.getfloat('kernels', 'sigma_fast')
sf.calc_update_frates(SpikeInfo, last_unit_col, kernel_fast, kernel_slow)

# train final models
n_model_comp = Config.getint('spike model', 'n_model_comp')
Models = sf.train_Models(SpikeInfo, last_unit_col, Waveforms, n_comp=n_model_comp)
outpath = (plots_folder / 'Models_final').with_suffix(fig_format)
sp.plot_Models(Models, save=outpath)

# final scoring and assignment
rescore = False  # FUTURE TODO - this might make sense under some conditions
# make this a flag in the config - explore when and when not this is better
if rescore:
    reassign_penalty = Config.getfloat('spike sort', 'reassign_penalty')
    noise_penalty = Config.getfloat('spike sort', 'noise_penalty')
    Scores, units = sf.Score_spikes(  # TODO refactor this one
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
else:
    SpikeInfo[final_unit_col] = SpikeInfo[last_unit_col]

# - algo done -
logger.info(' - saving results - ')

results_folder = results_folder.parent / 'results_manual'
plots_folder = results_folder / 'plots'
os.makedirs(plots_folder, exist_ok=True)


# waveforms to disk
outpath = results_folder / 'Waveforms.npy'
np.save(outpath, Waveforms)
logger.info(f'saving spike waveforms to {outpath}')

# plot final clustering
outpath = (plots_folder / 'Clustering_all').with_suffix(fig_format)
sp.plot_clustering(Waveforms, SpikeInfo, final_unit_col, save=outpath, N=2000)
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
        seg_name = Path(Seg.annotations['filename']).stem
        for j, unit in enumerate(units):
            (St,) = sf.select_by_dict(Seg.spiketrains, unit=unit)
            outpath = results_folder / (
                f'Segment_{seg_name}_unit_{unit}_spike_times.txt'
            )
            np.savetxt(outpath, St.times.magnitude)

    # firing rates - full res
    for i, Seg in enumerate(Blk.segments):
        FratesDf = pd.DataFrame()
        seg_name = Path(Seg.annotations['filename']).stem
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
    outpath = plots_folder / (seg_name + '_fitted_spikes' + fig_format)
    wsize = Config.getint('spike detect', 'wsize') * pq.ms
    sp.plot_fitted_spikes(
        Seg, j, Models, SpikeInfo, final_unit_col, wsize, zoom=zoom, save=outpath
    )

# plot final models
outpath = (plots_folder / f'{seg_name}_models_final').with_suffix(fig_format)
sp.plot_Models(Models, save=outpath)
logger.info('all plotting done')
logger.info('all tasks done - quitting')
sys.exit()

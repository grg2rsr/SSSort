# %%
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
# config_path = Path(os.path.abspath(sys.argv[1]))
config_path = Path(os.path.abspath("/home/georg/code/SSSort/example_config.ini"))
Config = configparser.ConfigParser()
Config.read(config_path)
print_msg('config file read from %s' % config_path)

# handling paths and creating output directory
data_path = Path(Config.get('path','data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path','experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots_manual'
os.makedirs(plots_folder, exist_ok=True)
os.chdir(config_path.parent / exp_name)

# copy config
shutil.copyfile(config_path, config_path.parent / exp_name / config_path.name)

# plotting
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')
plt.ion()

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
fs = Blk.segments[0].analogsignals[0].sampling_rate
dt = (1/fs).rescale(pq.ms).magnitude

# loading SpikeInfo
SpikeInfo = pd.read_csv(results_folder / 'SpikeInfo.csv')
unit_columns = [col for col in SpikeInfo.columns if col.startswith('unit_')]
SpikeInfo[unit_columns] = SpikeInfo[unit_columns].astype(str)

# loading Models
with open(results_folder / "Models.dill", 'rb') as fH:
    Models = dill.load(fH)

# loading Templates
Templates = np.load(results_folder / "Templates.npy")

"""
 
 ##     ## ######## ########   ######   #### ##    ##  ######      ##        #######   #######  ########  
 ###   ### ##       ##     ## ##    ##   ##  ###   ## ##    ##     ##       ##     ## ##     ## ##     ## 
 #### #### ##       ##     ## ##         ##  ####  ## ##           ##       ##     ## ##     ## ##     ## 
 ## ### ## ######   ########  ##   ####  ##  ## ## ## ##   ####    ##       ##     ## ##     ## ########  
 ##     ## ##       ##   ##   ##    ##   ##  ##  #### ##    ##     ##       ##     ## ##     ## ##        
 ##     ## ##       ##    ##  ##    ##   ##  ##   ### ##    ##     ##       ##     ## ##     ## ##        
 ##     ## ######## ##     ##  ######   #### ##    ##  ######      ########  #######   #######  ##        
 
"""


clust_alpha = Config.getfloat('spike sort','clust_alpha')
n_clust_final = Config.getint('spike sort','n_clust_final')

max_it = 20
illegal_merges = []
unit_col = unit_columns[-1]
n_merge = 0
n_clust = len(get_units(SpikeInfo, unit_col, remove_unassinged=True))


i = 0
while n_clust < n_clust_final or i < max_it:
    i += 1

    # calculate best merge
    units = get_units(SpikeInfo, unit_col)
    Avgs, Sds = calculate_pairwise_distances(Templates, SpikeInfo, unit_col)
    merge = best_merge(Avgs, Sds, units, clust_alpha)

    if len(merge) > 0 and merge not in illegal_merges:

        # show plots for this merge
        colors = get_colors(units)
        for k,v in colors.items():
            if k not in [str(m) for m in merge]:
                colors[k] = 'gray'
        fig, axes = plot_clustering(Templates, SpikeInfo, unit_col, colors=colors)
        fig, axes = plot_compare_templates(Templates, SpikeInfo, dt, merge)

        # ask for user input
        if input("do merge? (Y/N)?").upper() == 'Y':
            # prep merge
            n_merge += 1
            next_unit_col = unit_col + '_%i' % n_merge
            SpikeInfo[next_unit_col] = SpikeInfo[unit_col]

            # the merge
            ix = SpikeInfo.groupby(next_unit_col).get_group(merge[1])['id']
            SpikeInfo.loc[ix, next_unit_col] = merge[0]
            print_msg("merging: " + ' '.join(merge))

            unit_col = next_unit_col
        else:
            # if no, add merge to the list of forbidden merges
            illegal_merges.append(merge)

        plt.close('all')

    else:
        clust_alpha += 0.05
        print_msg("no merges, increasing alpha: %.2f" % clust_alpha)

    # exit condition if n_clusters is reached
    n_clust = len(get_units(SpikeInfo, unit_col, remove_unassinged=True))
    if n_clust == n_clust_final:
        print_msg("desired number of %i clusters reached" % n_clust_final)
        break
    
"""
 
 ######## #### ##    ## ####  ######  ##     ## 
 ##        ##  ###   ##  ##  ##    ## ##     ## 
 ##        ##  ####  ##  ##  ##       ##     ## 
 ######    ##  ## ## ##  ##   ######  ######### 
 ##        ##  ##  ####  ##        ## ##     ## 
 ##        ##  ##   ###  ##  ##    ## ##     ## 
 ##       #### ##    ## ####  ######  ##     ## 
 
"""
kernel_slow = Config.getfloat('kernels','sigma_slow')
kernel_fast = Config.getfloat('kernels','sigma_fast')
last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]

# final calculation of frate fast
calc_update_frates(Blk.segments, SpikeInfo, last_unit_col, kernel_fast, kernel_slow)

outpath = plots_folder / ("Clustering" + fig_format)
plot_clustering(Templates, SpikeInfo, last_unit_col, save=outpath)

# update spike labels
kernel = ele.kernels.GaussianKernel(sigma=kernel_fast * pq.s)

units = get_units(SpikeInfo, last_unit_col)

for i, seg in tqdm(enumerate(Blk.segments),desc="populating block for output"):
    spike_labels = SpikeInfo.groupby(('segment')).get_group((i))[last_unit_col].values
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
outpath = results_folder / 'SpikeInfo_manual.csv'
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)

# store Block
outpath = results_folder / 'result_manual.dill'
print_msg("saving Blk as .dill to %s" % outpath)
sssio.blk2dill(Blk, outpath)

# store models
outpath = results_folder / 'Models_manual.dill'
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
            outpath = results_folder / ("Segment_%s_unit_%s_spike_times_manual.txt" % (seg_name, unit))
            np.savetxt(outpath, St.times.magnitude)

    # firing rates - full res
    for i, Seg in enumerate(Blk.segments):
        FratesDf = pd.DataFrame()
        seg_name = Path(Seg.annotations['filename']).stem
        for j, unit in enumerate(units):
            asig, = select_by_dict(Seg.analogsignals, kind='frate_fast', unit=unit)
            FratesDf['t'] = asig.times.magnitude
            FratesDf[unit] = asig.magnitude.flatten()

        outpath = results_folder / ("Segment_%s_frates_manual.csv" % seg_name)
        FratesDf.to_csv(outpath)    

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
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_fitted_spikes' + fig_format)
    plot_fitted_spikes(Seg, j, Models, SpikeInfo, last_unit_col, zoom=zoom, save=outpath)

print_msg("plotting done")
print_msg("all done - quitting")

sys.exit()
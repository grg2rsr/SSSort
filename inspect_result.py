# sys
import sys
import os
import dill
import configparser
from pathlib import Path

# sci
import numpy as np
import pandas as pd

# ephys

# own
from functions import *
from plotters import *
import sssio

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt


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
print("config file read from %s" % config_path)

# handling paths and creating output directory
exp_name = Config.get("path", "experiment_name")
data_path = Path(Config.get("path", "data_path"))
if not data_path.is_absolute():
    data_path = config_path.parent / exp_name / data_path

if (config_path.parent / exp_name / "results_manual").exists():
    print("using manually curated results")
    results_folder = config_path.parent / exp_name / "results_manual"
else:
    results_folder = config_path.parent / exp_name / "results"

os.chdir(config_path.parent)

# plotting
mpl.rcParams["figure.dpi"] = Config.get("output", "screen_dpi")
# mpl.rcParams['figure.dpi'] = 166 # your screen dpi
# fig_format = Config.get('output','fig_format')
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
Blk = sssio.dill2blk(results_folder / "result.dill")
fs = Blk.segments[0].analogsignals[0].sampling_rate
dt = (1 / fs).rescale(pq.ms).magnitude

# loading SpikeInfo
SpikeInfo = pd.read_csv(results_folder / "SpikeInfo.csv")
unit_columns = [col for col in SpikeInfo.columns if col.startswith("unit_")]
SpikeInfo[unit_columns] = SpikeInfo[unit_columns].astype(str)

# loading Models
with open(results_folder / "Models.dill", "rb") as fH:
    Models = dill.load(fH)

# loading Waveforms
Waveforms = np.load(results_folder / "Waveforms.npy")

# prep for plotting
last_unit_col = [col for col in SpikeInfo.columns if col.startswith("unit")][-1]

# plot inspecting
# ask for trial index
if len(Blk.segments) > 1:
    # TODO could print a list of available trials
    print("enter the trial number you would like to inspect")
    j = int(input())
else:
    j = 0

# get d
Seg = Blk.segments[j]
unit_col = last_unit_col
wsize = Config.getint("spike detect", "wsize") * pq.ms
plot_fitted_spikes(Seg, j, Models, SpikeInfo, unit_col, wsize)
plot_waveforms(Waveforms, SpikeInfo, 2 * pq.ms, unit_col, N=100)
plot_Models(Models)

input()

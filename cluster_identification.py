########################################################################################################
# 
#  Loads neurons template and labels given clusters based on similarity to the templates
# 
########################################################################################################

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

# ephys
import neo
import elephant as ele

# own
from functions import *
from plotters import *
from functions_post_processing import *
from sssio import *
import sssio

import matplotlib.pyplot as plt


# Load file

# get config
config_path = Path(os.path.abspath(sys.argv[1]))
sssort_path = os.path.dirname(os.path.abspath(sys.argv[0]))
Config = configparser.ConfigParser()
Config.read(config_path)
logger.info('config file read from %s' % config_path)

mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output', 'fig_format')

# get segment to analyse
seg_no = Config.getint('postprocessing', 'segment_number')

# handling paths and creating output directory
data_path = Path(Config.get('path', 'data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path', 'experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots_post'

os.makedirs(plots_folder, exist_ok=True)

# create and config logger for writing to file
sssio.get_logger(exp_name)

# Load clustering data
Blk = get_data(results_folder / "result.dill")

SpikeInfo = pd.read_csv(results_folder / "SpikeInfo.csv")

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo, unit_column)

# Load Waveforms
Waveforms = np.load(results_folder / "Waveforms.npy")
fs = Blk.segments[seg_no].analogsignals[0].sampling_rate
n_samples = np.array(Config.get('postprocessing', 'template_window').split(','), dtype='float32')/1000.0
n_samples = np.array(n_samples * fs, dtype=int)

new_column = 'unit_labeled'

if new_column in SpikeInfo.keys():
    logger.info("Clusters already assigned")
    print(SpikeInfo[unit_column].value_counts())
    exit()

if len(units) != 3:
    print("Three units needed, %d found in SpikeInfo"%len(units))
    exit()

# Load model templates
template_A = np.load(os.path.join(sssort_path, "templates/template_A.npy"))
template_B = np.load(os.path.join(sssort_path, "templates/template_B.npy"))

# Adapt templates to negative detection.
if Config.get('spike detect', 'peak_mode') == 'negative':
    org_v_base = (np.min(template_A), np.min(template_B))
    template_A *= -1
    template_B *= -1

    # align back to negative values
    # (fix: outcome from sssort is negative ¿?¿)
    template_A -= abs(np.min(template_A)-org_v_base[0])
    template_B -= abs(np.min(template_B)-org_v_base[1])

# templates and waveforms need to be put on comparable shape and size
template_A, template_B, Waveforms = resize_waveforms(template_A, template_B, Waveforms, n_samples)

logger.info("Current units: %s" % units)

distances_a = []
distances_b = []

mode = 'peak'

logger.info("Computing best assignment")

# Average waveforms
mean_waveforms = get_aligned_wmean_by_unit(Waveforms, SpikeInfo, units,
                                           unit_column, mode)

# normalize waveforms
# get maximum amplitude for norm_factor
max_ampl = np.max([np.max(mean_wave)-np.min(mean_wave)
                   for unit, mean_wave in mean_waveforms.items()])

norm_factor = (np.max(template_A)-np.min(template_A))/max_ampl
normalized_means = [mean_wave*norm_factor for unit, mean_wave
                    in mean_waveforms.items()]

# Compare waveform units to templates
distances_a = [np.linalg.norm(mean-template_A) for mean in normalized_means]
distances_b = [np.linalg.norm(mean-template_B) for mean in normalized_means]

logger.info("Distances to a: ")
logger.info("\t\t%s" % str(units))
logger.info("\t\t%s" % str(distances_a))
logger.info("Distances to b: ")
logger.info("\t\t%s" % str(units))
logger.info("\t\t%s" % str(distances_b))

# Get best assignments
a_unit = units[np.argmin(distances_a)]
b_unit = units[np.argmin(distances_b)]
if len(units) > 2:
    non_unit = [unit for unit in units if a_unit not in unit and b_unit not in unit][0]

asigs = {a_unit: 'A', b_unit: 'B'}
if len(units) > 2:
    asigs[non_unit] = '?'
logger.info("Final assignation: %s" % asigs)

# plot assignments
outpath = plots_folder / ("cluster_reassignments" + fig_format)
plot_means(normalized_means, units, template_A, template_B, asigs=asigs, outpath=outpath)

# create new column with reassigned labels
SpikeInfo[new_column] = copy.deepcopy(SpikeInfo[unit_column].values)
if len(units) > 2:
    non_unit_rows = SpikeInfo.groupby(new_column).get_group(non_unit)
    SpikeInfo.loc[non_unit_rows.index, new_column] = '-2'

# Relabel column to A/B
a_unit_rows = SpikeInfo.groupby(new_column).get_group(a_unit)
SpikeInfo.loc[a_unit_rows.index, new_column] = 'A'
b_unit_rows = SpikeInfo.groupby(new_column).get_group(b_unit)
SpikeInfo.loc[b_unit_rows.index, new_column] = 'B'

# store SpikeInfo
outpath = results_folder / 'SpikeInfo_post.csv'
logger.info("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath, index=False)

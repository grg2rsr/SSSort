import matplotlib.pyplot as plt
import pandas as pd
import sssio
from plotters import *
from functions import *
from functions_post_processing import *
from sys import path

import configparser


plt.rcParams.update({'font.size': 6})

"""
########  #######    #######   ##       ######
   ##    ##     ##  ##     ##  ##      ##    ##
   ##    ##     ##  ##     ##  ##      ##
   ##    ##     ##  ##     ##  ##       ######
   ##    ##     ##  ##     ##  ##            ##
   ##    ##     ##  ##     ##  ##      ##    ##
   ##     #######    #######   #######  ######
"""


# insert row *after* idx
def insert_row(df, idx, df_insert):
    dfA = df.iloc[:idx+1, ].copy()
    dfB = df.iloc[idx+1:, ].copy()
    df_insert = pd.DataFrame([df_insert], columns=df.columns)

    # Iterate over the columns and force the same data type as in df_insert
    for column in df.columns:
        df_insert[column] = df_insert[column].astype(df[column].dtype)

    df = pd.concat([dfA, df_insert, dfB]).reset_index(drop=True)
    return df


# delete row at idx
def delete_row(df, idx):
    return df.iloc[:idx, ].append(df.iloc[idx+1:]).reset_index(drop = True)


# insert a new spike entry after row idx
def insert_spike(SpikeInfo, new_column, idx, o_spike_time, o_spike_unit):
    SpikeInfo = insert_row(SpikeInfo, idx, SpikeInfo.iloc[idx, ]) # insert a copy of row idx 
    SpikeInfo[new_column][idx+1] = o_spike_unit
    SpikeInfo['id'][idx+1] = str(SpikeInfo['id'][idx])+'B'
    SpikeInfo['time'][idx+1] = o_spike_time
    SpikeInfo['good'][idx+1] = False   # do not use for building templates!
    SpikeInfo['frate_fast'][idx+1] = SpikeInfo['frate_'+o_spike_unit][idx+1]   # update rate_fast to the correct rate for the nwe spike's identity
    return SpikeInfo

    
"""
##       ######      ###     ########      ######  ##      ##     ##   ######  ######## ######## ########  
##      ##    ##    ## ##    ##     ##    ##    ## ##      ##     ##  ##    ##    ##    ##       ##     ## 
##      ##    ##   ##   ##   ##     ##    ##       ##      ##     ##  ##          ##    ##       ##     ## 
##      ##    ##  ##     ##  ##     ##    ##       ##      ##     ##   ######     ##    ######   ########  
##      ##    ##  #########  ##     ##    ##       ##      ##     ##        ##    ##    ##       ##   ##   
##      ##    ##  ##     ##  ##     ##    ##    ## ##      ##     ##  ##    ##    ##    ##       ##    ## 
#######  ######   ##     ##  ########      ######  #######  #######    ######     ##    ######## ##     ##  
"""


# get config
config_path = Path(os.path.abspath(sys.argv[1]))
sssort_path = os.path.dirname(os.path.abspath(sys.argv[0]))
Config = configparser.ConfigParser()
Config.read(config_path)
logger.info('config file read from %s' % config_path)

# get segment to analyse
seg_no = Config.getint('postprocessing', 'segment_number')

# handling paths and creating outunits.sort()put directory
data_path = Path(Config.get('path', 'data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path', 'experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots_post' / 'post_process_all' 

os.makedirs(plots_folder, exist_ok=True)

# create and config logger for writing to file
sssio.get_logger(exp_name)

# plot config
# plotting_changes = Config.getboolean('postprocessing','plot_changes')
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output', 'fig_format')


# Load necessary data

# Load block
Blk = get_data(results_folder / "result.dill")

# Load SpikeInfo
error_msg = "It appears that you have not yet labeled the spike clusters. Run cluster_identification.py first"
try:
    SpikeInfo = pd.read_csv(results_folder / "SpikeInfo_post.csv")
except:
    logger.info(error_msg)
    exit()

if 'unit_labeled' not in SpikeInfo.columns:
    logger.info(error_msg)
    exit()

unit_column = 'unit_labeled'
SpikeInfo = SpikeInfo.astype({'id': str, unit_column: str})
units = get_units(SpikeInfo, unit_column)

stimes = SpikeInfo['time']
seg = Blk.segments[seg_no]
fs = np.float64(seg.analogsignals[0].sampling_rate)
ifs = int(fs/1000)   # sampling rate in kHz as integer value to convert ms to bins NOTE: assumes sampling rate divisible by 1000

### Train models

# recalculate the latest firing rates according to spike assignments in unit_column
# kernel_slow = Config.getfloat('kernels','sigma_slow')
kernel_fast = Config.getfloat('kernels', 'sigma_fast')
calc_update_final_frates(SpikeInfo, unit_column, kernel_fast)

waveforms_path = config_path.parent / results_folder / "Waveforms.npy"
Waveforms = np.load(waveforms_path)
logger.info('templates read from %s' % waveforms_path)
n_model_comp = Config.getint('spike model', 'n_model_comp')


spike_model_type = Config.get('postprocessing', 'spike_model_type')

spike_model = Spike_Model if spike_model_type == "individual" else Spike_Model_Nlin 
Models = train_Models(SpikeInfo, unit_column, Waveforms, n_comp=n_model_comp, model_type=spike_model)

unit_ids = SpikeInfo[unit_column]
units = get_units(SpikeInfo, unit_column)
frate = {}
for unit in units:
    frate[unit] = SpikeInfo['frate_'+unit]

##########################################################################################

sz_wd = Config.getfloat('postprocessing', 'spike_window_width')
align_mode = Config.get('postprocessing', 'vertical_align_mode')
same_spike_tolerance = Config.getfloat('postprocessing', 'spike_position_tolerance')
same_spike_tolerance = int(same_spike_tolerance*ifs) # in time steps
d_accept = Config.getfloat('postprocessing', 'max_dist_for_auto_accept')
d_reject = Config.getfloat('postprocessing', 'min_dist_for_auto_reject')
min_diff = Config.getfloat('postprocessing', 'min_diff_for_auto_accept')
wsize = Config.getfloat('spike detect', 'wsize')
max_spike_diff = int(Config.getfloat('postprocessing', 'max_compound_spike_diff') * ifs)
n_samples = np.array(Config.get('postprocessing', 'template_window').split(','), dtype='float32')/1000.0
n_samples = np.array(n_samples*fs, dtype=int)

try:
    spkr = Config.get('postprocessing', 'spike_range').replace(' ', '').split(',')
    spike_range = range(pd.Index(SpikeInfo['id']).get_loc(spkr[0]),
                        pd.Index(SpikeInfo['id']).get_loc(spkr[1]))
except:
    logger.info("spike index range not valid, reverting to processing all")
    spike_range = range(1, len(unit_ids)-1)
spike_label_interval = Config.getint('output', 'spike_label_interval')

asig = seg.analogsignals[0]
asig = asig.reshape(asig.shape[0])
as_min = np.amin(asig)
as_max = np.amax(asig)
y_lim = [1.05*as_min, 1.05*as_max]
n_wd = int(sz_wd*ifs)
n_wdh = n_wd//2

"""
##     ##      ###     ##   ##    ##    ##        #######    #######   ########   
###   ###     ## ##    ##   ###   ##    ##       ##     ##  ##     ##  ##     ## 
#### ####    ##   ##   ##   ####  ##    ##       ##     ##  ##     ##  ##     ## 
## ### ##   ##     ##  ##   ## ## ##    ##       ##     ##  ##     ##  ########  
##     ##   #########  ##   ##  ####    ##       ##     ##  ##     ##  ##    
##     ##   ##     ##  ##   ##   ###    ##       ##     ##  ##     ##  ##    
##     ##   ##     ##  ##   ##    ##    ########  #######    #######   ##    
"""

new_column = 'unit_final'
if new_column not in SpikeInfo.keys():
    SpikeInfo[new_column] = SpikeInfo['unit_labeled']
offset = 0   # will keep track of shifts due to inserted and deleted spikes 
# don't consider first and last spike to avoid corner cases; these do not matter in practice anyway
# tracemalloc.start()
skip = False
for i in spike_range:
    if skip:
        skip = False
        continue
    start = int((float(stimes[i])*1000-sz_wd/2)*ifs)
    stop = start+n_wd

    if not ((start > 0) and (stop < len(asig))):   # only do something if the spike is not too close to the start or end of the recording, otherwise ignore
        continue

    # spikes are not borders:
    v = np.array(asig[start:stop], dtype=float)
    v = align_to(v, align_mode)
    d = []
    sh = []
    un = []
    templates = {}
    for unit in units[:2]:
        templates[unit] = make_single_template(Models[unit], frate[unit][i])
        templates[unit] = align_to(templates[unit], align_mode)
        for pos in range(n_wdh-same_spike_tolerance, n_wdh + same_spike_tolerance):
            d.append(dist(v, templates[unit], n_samples, pos))
            sh.append(pos)
            un.append(unit)
    d2 = []
    sh2 = []
    for pos1 in range(n_wd):
        for pos2 in range(n_wd):
            # one of the spikes must be close to the spike time under consideration
            if ((abs(pos1-n_wdh) <= same_spike_tolerance) or (abs(pos2-n_wdh) <= same_spike_tolerance)) and (abs(pos1-pos2) < max_spike_diff):
                d2.append(compound_dist(v, templates['A'], templates['B'], n_samples, pos1, pos2))
                sh2.append((pos1, pos2))

    # work out the final decision
    best = np.argmin(d)
    best2 = np.argmin(d2)
    d_min = min(d[best], d2[best2])
    choice = 1 if d[best] <= d2[best2] else 2
    d_diff = abs(d[best]-d2[best2])

    logger.info("Spike {}: Single spike d={}, compound spike d={}, difference={}".format(SpikeInfo['id'][i+offset], ('%.4f' % d[best]), ('%.4f' % d2[best2]), ('%.4f' % d_diff)))

    # plot params
    zoom = (float(stimes[i])-sz_wd/1000*20, float(stimes[i])+sz_wd/1000*20)
    colors = get_colors(['A', 'B'], keep=False)

    if d_min >= d_accept or 200*d_diff/(d[best]+d2[best2]) < min_diff:
        # make plots and save them
        fig2, ax2 = plot_fitted_spikes_pp(seg, Models, SpikeInfo, new_column, zoom=zoom, box=(float(stimes[i]),sz_wd/1000), wsize=n_samples, spike_label_interval=spike_label_interval, colors=colors)
        outpath = plots_folder / (str(SpikeInfo['id'][i+offset]) + '_context_plot' + fig_format)
        ax2[1].plot(stimes[i], 1, '.', color='y')

        fig2.savefig(outpath)

        fig, ax = plt.subplots(ncols=2, sharey= True, figsize=[4, 2])
        dist(v, templates[un[best]], n_samples, sh[best], unit=un[best], ax=ax[0])
        ax[0].set_ylim(y_lim)
        compound_dist(v, templates['A'], templates['B'], n_samples, sh2[best2][0], sh2[best2][1], ax[1])
        ax[1].set_ylim(y_lim)
        outpath = plots_folder / (str(SpikeInfo['id'][i+offset]) + '_template_matches' + fig_format)
        fig.savefig(outpath)
        if d_min > d_reject:
            choice = 0
        else:
            # show some plots first
            fig2.show()
            fig.show()
            # ask user
            if (200*d_diff/(d[best]+d2[best2]) <= min_diff):
                reason = "two very close matches"
            elif d_min >= d_accept:
                reason = "no good match but not bad enough to reject"
            print("User feedback required: "+reason)
            choice = " "
            while choice not in ["0", "1", "2"]:
                choice = input("Single spike (1), Compound spike (2), no spike (0)? ")
            choice = int(choice)

        plt.close(fig2)
        plt.close(fig)

    # apply choice
    if choice == 1:
        # it's a single spike - choose the appropriate single spike unit
        peak_pos = np.argmax(templates[un[best]])
        peak_diff = peak_pos-n_samples[0]   # difference in actual peak pos compared where it should be
        spike_time = stimes[i]+np.float64((sh[best]-n_wdh+peak_diff))/fs  # spike time in seconds
        if (abs(stimes[i+1]-spike_time)*fs < max_spike_diff):
            skip = True
            # this spike was recorded within compound spike distance before
            if 'B' in str(SpikeInfo['id'][i+1+offset]):
                # this is a spike entry that was previously created by DroSort, delete
                logger.info("Spike {}: time= {}: Single spike, was type {} now of type {}, time {}. Conflicting spike {}; deleted {}".format(SpikeInfo['id'][i+offset],('%.4f' % stimes[i]),SpikeInfo[unit_column][i+offset],un[best],('%.4f' % spike_time),SpikeInfo['id'][i+1+offset],SpikeInfo['id'][i+1+offset]))
                SpikeInfo = delete_row(SpikeInfo, i+1+offset)
                offset -= 1
            else:
                # this is a detected spike, keep for further reference
                logger.info("Spike {}: time= {}: Single spike, was type {} now of type {}, time {}. Conflicting spike {}; marked {} for deletion (-2)".format(SpikeInfo['id'][i+offset],('%.4f' % stimes[i]),SpikeInfo[unit_column][i+offset],un[best],('%.4f' % spike_time),SpikeInfo['id'][i+1+offset],SpikeInfo['id'][i+1+offset]))
                SpikeInfo[new_column][i+offset]= '-2'
                SpikeInfo['good'][i+offset]= False
                SpikeInfo['frate_fast'][i+offset]= SpikeInfo['frate_'+un[best]][i+offset]
        else:
            # spike isn't duplicated, normal assignment of a single spike
            logger.info("Spike {}: time= {}: Single spike, was type {}, now  of type {}, time= {}".format(SpikeInfo['id'][i+offset],('%.4f' % stimes[i]),SpikeInfo[unit_column][i+offset],un[best],('%.4f' % spike_time)))
            SpikeInfo[new_column][i+offset] = un[best]
            SpikeInfo['time'][i+offset] = spike_time
            SpikeInfo['frate_fast'][i+offset] = SpikeInfo['frate_'+un[best]][i+offset]
    elif choice == 2:
        # it's a compound spike - choose the appropriate spike unit and handle second spike
        orig_spike = np.argmin(abs(np.array(sh2[best2])-n_wdh))
        other_spike = 1-orig_spike
        spike_unit = 'A' if orig_spike == 0 else 'B'
        peak_pos = np.argmax(templates[spike_unit])
        peak_diff = peak_pos-n_samples[0]   # difference in actual peak pos compared where it should be
        spike_time = stimes[i]+np.float64(sh2[best2][orig_spike]-n_wdh+peak_diff)/fs  # spike time in seconds
        logger.info("Spike {}: time= {}: Compound spike, first spike of type {}, time= {}".format(SpikeInfo['id'][i+offset],('%.4f' % SpikeInfo['time'][i+offset]),spike_unit,('%.4f' % spike_time)))
        SpikeInfo[new_column][i+offset] = spike_unit
        SpikeInfo['time'][i+offset] = spike_time
        SpikeInfo['good'][i+offset] = False   # do not use compound spikes for Model building
        SpikeInfo['frate_fast'][i+offset] = SpikeInfo['frate_'+spike_unit][i+offset]
        o_spike_id = i+1
        o_spike_unit = 'A' if other_spike == 0 else 'B'
        peak_pos = np.argmax(templates[o_spike_unit])
        peak_diff = peak_pos-n_samples[0]   # difference in actual peak pos compared where it should be
        o_spike_time = stimes[i]+float(sh2[best2][other_spike]-n_wdh+peak_diff)/fs  # spike time in seconds
        found = False
        for j in [i-1, i+1]:
            if abs(stimes[j]-o_spike_time)*fs < same_spike_tolerance:
                # the other spike coincides with the previous spike in the original list
                # make sure that the previous decision is consistent with the current one
                logger.info("Spike {}: time= {}: Compound spike, second spike was known as {}, now of type {}, time= {}".format(SpikeInfo['id'][i+offset],('%.4f' % SpikeInfo['time'][i+offset]),SpikeInfo[unit_column][o_spike_id+offset],o_spike_unit,('%.4f' % o_spike_time)))
                SpikeInfo[new_column][o_spike_id+offset] = o_spike_unit
                SpikeInfo['good'][o_spike_id+offset] = False   # do not use compound spikes for Model building
                SpikeInfo['frate_fast'][o_spike_id+offset] = SpikeInfo['frate_'+o_spike_unit][o_spike_id+offset]
                found = True
                if j == i+1:
                    skip = True
                break
        if not found:
            # the other spike does not yet exist in the list: insert new row
            logger.info("Spike {}: Compound spike, second spike was undetected, inserted new spike of type {}, time= {}".format(SpikeInfo['id'][i+offset],o_spike_unit,o_spike_time))
            SpikeInfo = insert_spike(SpikeInfo, new_column, i+offset, o_spike_time, o_spike_unit)
            offset += 1
        
    else:
        # it's a non-spike - delete it or mark for deletion
        if 'B' in str(SpikeInfo['id'][i+offset]):   # this is a spike entry that was previously created by DroSort, delete
            SpikeInfo= delete_row(SpikeInfo, i+offset)
            logger.info("Spike {}: Not a spike, inserted by DroSort previously, row removed".format(SpikeInfo['id'][i+offset]))
            offset -= 1
        else:   # this is a detected spike, keep for further reference
            SpikeInfo[new_column][i+offset] = '-2'
            SpikeInfo['good'][i+offset] = False   # definitively do not use for model building
            logger.info("Spike {}: Not a spike, marked for deletion (-2)".format(SpikeInfo['id'][i+offset]))
            
calc_update_final_frates(SpikeInfo, unit_column, kernel_fast)

# Saving
kernel = ele.kernels.GaussianKernel(sigma=kernel_fast * pq.s)
fs = seg.analogsignals[0].sampling_rate
spike_labels = SpikeInfo[new_column].values
times = SpikeInfo['time'].values
St = seg.spiketrains[0]
seg.spiketrains[0] = neo.core.SpikeTrain(times, units='sec', t_start=St.t_start, t_stop=St.t_stop)
seg.spiketrains[0].array_annotate(unit_labels=list(spike_labels))

# make spiketrains
St = seg.spiketrains[0]
sts = [St]

for unit in units:
    times = St.times[sp.array(spike_labels) == unit]
    st = neo.core.SpikeTrain(times, t_start=St.t_start, t_stop=St.t_stop)
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

#save all
units = get_units(SpikeInfo, unit_column)
logger.info("Number of spikes in trace: %d" % SpikeInfo[new_column].size)
logger.info("Number of clusters: %d" % len(units))

# warning firing rates not saved, too high memory use.
save_all(results_folder, SpikeInfo, Blk, logger, FinalSpikes=True, f_extension='post')

do_plot = Config.getboolean('postprocessing', 'plot_fitted_spikes')

if do_plot:
    logger.info("creating plots")
    outpath = plots_folder / ('overview' + fig_format)
    plot_segment(seg, units, save=outpath, colors=colors)

    max_window = Config.getfloat('output', 'max_window_fitted_spikes_overview')
    plot_fitted_spikes_complete(seg, Models, SpikeInfo, new_column, max_window,
                                plots_folder, fig_format, wsize=n_samples,
                                extension='_templates', spike_label_interval=spike_label_interval,
                                colors=colors)
    logger.info("plotting done")

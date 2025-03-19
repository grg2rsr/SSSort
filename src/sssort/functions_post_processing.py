# system

# sci
import numpy as np

# ephys


# plotting


# own
import sssort.functions as sf
from sssort import sssio


def calc_update_final_frates(SpikeInfo, unit_column, kernel_fast):
    """calculate all firing rates for all units, based on unit_column. This is for after units
    have been identified as 'A' or 'B' (or unknown). Updates SpikeInfo with new columns frate_A, frate_B"""

    from_units = sf.get_units(SpikeInfo, unit_column)

    # estimating firing rate profile for "from unit" and getting the rate at "to unit" timepoints
    for j, from_unit in enumerate(from_units):
        try:
            SInfo = SpikeInfo.groupby([unit_column]).get_group((from_unit))

            # spike times
            from_times = SInfo['time'].values
            to_times = SpikeInfo['time'].values
            # estimate its own rate at its own spike times
            rate = sf.est_rate(from_times, to_times, kernel_fast)
            # set
            SpikeInfo['frate_' + from_unit] = rate
        except:
            # can not set it's own rate, when there are no spikes in this segment for this unit
            pass


def save_all(results_folder, SpikeInfo, Blk, logger, FinalSpikes=False, f_extension=''):
    # store SpikeInfo
    outpath = results_folder / ('SpikeInfo_%s.csv' % f_extension)
    logger.info('saving SpikeInfo to %s' % outpath)
    SpikeInfo.to_csv(outpath, index=False)

    if FinalSpikes:
        # store separate spike time lists for A and B cells
        for unit in ['A', 'B']:
            st = SpikeInfo.groupby('unit_final').get_group(unit)['time']
            outpath = results_folder / ('Spikes' + unit + '.csv')
            np.savetxt(outpath, st)

    # store Block
    outpath = results_folder / 'result.dill'
    logger.info('saving Blk as .dill to %s' % outpath)
    sssio.blk2dill(Blk, outpath)

    logger.info('data is stored')


"""

 ########  ########   ######  ########  ########   #######   ######  ########  ######   ######
 ##     ## ##     ## ##    ## ##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ##
 ##     ## ##     ## ##       ##     ## ##     ## ##     ## ##       ##       ##       ##
 ########  ##     ##  ######  ########  ########  ##     ## ##       ######    ######   ######
 ##        ##     ##       ## ##        ##   ##   ##     ## ##       ##             ##       ##
 ##        ##     ## ##    ## ##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ##
 ##        #########  ######  ##        ##     ##  #######   ######  ########  ######   ######

"""


# def get_neighbors_amplitude(st, Templates, SpikeInfo, unit_column, unit, idx=0, t=0.3):
#     times_all = SpikeInfo['time']

#     idx_t = times_all.values[idx]

#     ini = idx_t - t
#     end = idx_t + t

#     times = times_all.index[np.where((times_all.values > ini) & (times_all.values < end) & (times_all.values != idx_t))]
#     neighbors = times[np.where(SpikeInfo.loc[times, unit_column].values==unit)]

#     T_b = Templates[:,neighbors].T
#     T_b = np.array([max(t[t.size//2:])-min(t[t.size//2:]) for t in T_b])

#     return sp.average(T_b)

# def get_duration(waveform):
#     ampl = (max(waveform)-min(waveform))
#     thres = max(waveform)-(ampl)/3
#     try:
#         duration_vals = np.where(np.isclose(waveform, thres,atol=0.06))[0]
#         dur = duration_vals[-1]-duration_vals[0]
#     except:
#         dur = -np.inf

#     return dur

# def get_neighbors_duration(st, Templates, SpikeInfo, unit_column, unit, idx=0, t=0.3):
#     times_all = SpikeInfo['time']

#     idx_t = times_all.values[idx]

#     ini = idx_t - t
#     end = idx_t + t

#     times = times_all.index[np.where((times_all.values > ini) & (times_all.values < end) & (times_all.values != idx_t))]
#     neighbors = times[np.where(SpikeInfo.loc[times,unit_column].values==unit)]

#     T_b = Templates[:,neighbors].T

#     durations = []

#     for waveform in T_b:
#         dur = get_duration(waveform)
#         durations.append(dur)

#     return sp.average(durations)

# def remove_spikes(SpikeInfo, unit_column, criteria):
#     if criteria == 'min':
#         units = get_units(SpikeInfo, unit_column)
#         spike_labels = SpikeInfo[unit_column]

#         n_spikes_units = []
#         for unit in units:
#             ix = sp.where(spike_labels == unit)[0]
#             n_spikes_units.append(ix.shape[0])

#         rm_unit = units[sp.argmin(n_spikes_units)]
#     else:
#         rm_unit = criteria

#     SpikeInfo[unit_column] = SpikeInfo[unit_column].replace(rm_unit, '-1')


# def distance_to_average(Templates, averages):
#     D_pw = sp.zeros((len(averages), Templates.shape[1]))

#     for i,average in enumerate(averages):
#         D_pw[i,:] = metrics.pairwise.euclidean_distances(Templates.T,average.reshape(1, -1)).reshape(-1)
#     return D_pw.T


def align_to(spike, mode='peak'):
    if spike.shape[0] != 0:
        if type(mode) is not str:
            mn = mode
        elif mode == 'min':
            mn = np.min(spike)
        elif mode == 'peak':
            mn = np.max(spike)
        elif mode == 'end':
            mn = spike[-1]
        elif mode == 'ini':
            mn = spike[0]
        elif mode == 'mean':
            mn = np.mean(spike)
        else:
            print('fail')
            return spike

        if mn != 0:
            spike = spike - mn

    return spike


# generate a template from a model at a given firing rate
def make_single_template(Model, frate):
    d = Model.predict(frate)
    return d


"""
function bounds() - indices for adding a template at a defined position into a frame of length ln
inputs:
ln - number of samples in the data window
n_samples - list of length 2 with number of samples to consider left and right of typical template peak
pos - index of the current spike under consideration
outputs:
start - index in the data window where to start pasting template data
stop - index in the data window where to stop
t_start - index in the template where to start taking data from
t_end - index in the templae where to stop
"""


def bounds(ln, n_samples, pos):
    start = max(int(pos - n_samples[0]), 0)  # start index of data in data window
    stop = min(int(pos + n_samples[1]), ln)  # stop index of data in data window
    t_start = max(
        int(n_samples[0] - pos), 0
    )  # start index of data taken from template within the template
    t_stop = t_start + stop - start  # stop index of data taken
    return (start, stop, t_start, t_stop)


"""
function dist() - calculate the distance between a data trace and a template at a shift
Inputs:
d - a data window from the experimental data (centred around a candidate spike)
t - a template of a candidate spike
n_samples - list of length 2 with number of samples to consider left and right of typical template peak
pos - position of the template to be tested, relative to original candidate spike
unit - name of the neuron unit considered (for axis label if plotting)
ax - axis to plot into, no plotting if None
"""


def dist(d, t, n_samples, pos, unit=None, ax=None, avg_amplitude=1):
    # Make a template at position pos expressed as index in data window d
    t2 = np.zeros(len(d))
    start, stop, t_start, t_stop = bounds(len(d), n_samples, pos)
    t2[start:stop] = t[
        t_start:t_stop
    ]  # template shifted and cropped to comparison region
    # data outside where the template sits is zeroed, so that those
    # regions are not considered during the comparison
    d2 = np.zeros(len(d))
    d2[start:stop] = d[start:stop]  # data cropped to comparison region
    dst = np.linalg.norm(d2 - t2)
    dst = dst / (stop - start)
    if ax is not None:
        ax.plot(d, '.', markersize=1, label='org. trace')
        ax.plot(d2, linewidth=0.7, label='comp. region')
        ax.plot(t2, linewidth=0.7, label='template')
        ax.set_ylim(-1.2, 1.2)
        lbl = unit + ': d=' if unit is not None else ''
        ax.set_title(lbl + ('%.2f' % (dst * 100 / avg_amplitude) + '%'))
        ax.legend()
    return dst


# calculate the distance between a data trace and a compound template
def compound_dist(d, t1, t2, n_samples, pos1, pos2, ax=None, avg_amplitude=1):
    # assemble a compound template with positions pos1 and pos2
    t = np.zeros(len(d))
    start1, stop1, t_start1, t_stop1 = bounds(len(d), n_samples, pos1)
    t[start1:stop1] += t1[t_start1:t_stop1]
    start2, stop2, t_start2, t_stop2 = bounds(len(d), n_samples, pos2)
    t[start2:stop2] += t2[t_start2:t_stop2]
    # blank out data left and right of compound template
    # NOTE: we are not blanking between templates if there is a gap
    # This is deliberate; such cases get thus penalized - they should
    # be treated as individual spikes
    d2 = np.zeros(len(d))
    start_l = min(start1, start2)
    stop_r = max(stop1, stop2)
    d2[start_l:stop_r] = d[start_l:stop_r]
    dst = np.linalg.norm(d2 - t)
    dst = dst / (stop_r - start_l)
    if ax is not None:
        ax.plot(d, '.', markersize=1)
        ax.plot(d2, linewidth=0.7)
        ax.plot(t, linewidth=0.7)
        ax.set_ylim(-1.2, 1.2)
        lbl = 'A+B: d=' if pos1 <= pos2 else 'B+A: d='
        ax.set_title(lbl + ('%.2f' % (dst * 100 / avg_amplitude) + '%'))
    return dst


# # Populate block anotates spike trains in the segment and add 2 spike trains with each unit.
# def populate_block(Blk, SpikeInfo, unit_column, units):
#     for i, seg in enumerate(Blk.segments):
#         spike_labels = SpikeInfo.groupby(('segment')).get_group((i))[unit_column].values
#         SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
#         SpikeTrain.annotations['unit_labels'] = list(spike_labels)

#         # make spiketrains
#         spike_labels = SpikeTrain.annotations['unit_labels']
#         sts = [SpikeTrain]
#         for unit in units:
#             times = SpikeTrain.times[np.array(spike_labels) == unit]
#             st = neo.core.SpikeTrain(times, t_start = SpikeTrain.t_start, t_stop=SpikeTrain.t_stop)
#             st.annotate(unit=unit)
#             sts.append(st)
#         seg.spiketrains = sts

#         asigs = [seg.analogsignals[0]]
#         seg.analogsignals = asigs

#     return Blk


def resize_waveforms(template_A, template_B, Waveforms, n_samples):
    # get boundaries
    tmid_a = np.argmax(template_A)
    tmid_b = np.argmax(template_B)
    left = np.amin([tmid_a, tmid_b, n_samples[0]])
    right = np.amin([len(template_A) - tmid_a, len(template_B) - tmid_b, n_samples[1]])

    # adjuts waveforms
    template_A = template_A[tmid_a - left : tmid_a + right]
    template_B = template_B[tmid_b - left : tmid_b + right]
    Waveforms = Waveforms[n_samples[0] - left : n_samples[0] + right, :]

    return template_A, template_B, Waveforms


def get_aligned_wmean_by_unit(Waveforms, SpikeInfo, units, unit_column, mode):
    mean_waveforms = {}

    for unit in units:
        unit_ids = SpikeInfo.groupby(unit_column).get_group(unit)['id']
        waveforms = Waveforms[:, unit_ids]

        # Align waveforms by mode
        waveforms = np.array([np.array(align_to(t, mode)) for t in waveforms.T])

        # Get mean for each unit and amplitude
        mean_waveforms[unit] = np.average(waveforms, axis=0)

    return mean_waveforms

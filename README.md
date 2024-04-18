WORK IN PROGRESS
# SSSort
A dedicated spike sorting algorithm for single sensillum recordings
 
## Authors
+ Georg Raiser, PhD, Champalimaud Research, grg2rsr@gmail.com
+ 
+ 

 
## Summary
_SSSort_ is a spike sorting algorithm for single sensillum recordings (SSR). _SSSort_ is the successor of [_SeqPeelSort_](https://github.com/grg2rsr/SeqPeelSort). Both algorithms have been designed to sort spikes from SSR into units, adressing the challenge that spike shape in SSR can change quite a lot, depending on the unit's firing rate. However, the core of both algorithms is quite different. While _SeqPeelSort_ tries to estimate the distribution of possible spike shapes and attempts to template match many different spike shapes drawn from those estimated distributions, _SSSort_ forms an explicit model on how the firing rate impacts the spike shape. Although developed for data from insect single sensillum recordings, _SSSort_ can probably be used for any type of recordings in which spike shape changes as a function of firing rate, however currently only single electrode recordings are implemented.
 
## Installation

```
git clone https://github.com/grg2rsr/SSSort.git
cd SSSort
conda env create -n sssort -f ./envs/ubuntu_22_04.yml
python setup.py
```
currently does it.

## Usage
To launch SSSort, generally use:
```
sssort mode path_to_config.ini
```
`path_to_config.ini` is the path to the config file
`mode` can be any of: `convert`, `detect`, `sort`, `inspect`, `merge`, `postprocess`, each described below:

### `convert`
```
sssort convert path_to_data [extra_args]
```
will try to convert your data into the internal data representation (a `neo.core.AnalogSignal`) and store it as a `.dill` in the same folder as your recording.

`extra args` will depend on your data format:

#### `.csv` files:
`sssort convert data.csv t_index v_index`
+ `t_index` = the 0-based index for the column of the samples time stamps
+ `v_index` = the 0-based index for the column of your voltage signals.

#### Spike2 `.smr` files
`sssort convert data.smr channel_index`
+ `channel_index` = the 0-based index with the channel of your recording 

#### Autospike `.asc` files

#### Raw / binary `.bin` files
`sssort convert data.bin sampling_rate data_type`
+ `sampling_rate` = the sampling frequency of your recording, in Hz
+ `data_type` = allowed are `[u]int16/32/64`, `float32/64`

### `detect`
```
sssort detect path_to_config.ini
```
runs only spike detection and opens a windows to interactively set the spike detection parameters (peak amplitude and prominence). It is recommended to do this first before running `sssort sort`, as erronous spike detection will greatly impact the success of the following steps.

In the window, you can left-click on the individual points in the scatter plots on the left-hand side to visualize the corresponding spike on the right-hand side.

A right-click will update the thresholds for both amplitude and prominence that will be used for spike detection.

Pressing enter stores the current values in the config. If your last click was in one of the upper two plots, the recording peak detection will run on the positive peaks, if you clicked in the lower row the peak detection will run on the negative peaks.

[]image describing the process

### `sort`
To run the main sorting pipline, use:
```
sssort sort path_to_config.ini
```
If in the `config.ini` the parameter `manual_merge = True`, all the algorithm will halt at every merge and ask for user confirmation, otherwise it runs automatically until any of the termination criteria are met.

### `inspect`
After sorting, you can interactively inspect the result using:
```
sssort inspect path_to_config.ini -e [trial_index]
```
This opens an interactive viewer (matplotlib) to inspect the sorting quality in an interactive manner

+ `trial_index` (optional) = is the index of the trial you would like to inspect. Warning: will plot the entire recording if no trial index is given

### `merge`

If the algorithm either did not merge down to the number of desired clusters, or you want to do the final merges yourself, you can use
```
sssort merge path_to_config.ini [n_clusters] [iteration]
```
+ `n_clusters`: number of clusters to merge down to. If not provided, the number in the `config.ini` will be used.


### `postprocess`


 
## Algorithm details
_SSSort_ detects all spiky deviations within a recording (the _spikes_) and performs an initial clustering (k means). As now each spike belongs to a cluster grouping them into units, the local firing rates can be estimated. This gives a single value of the local firing rate at each timepoint of a spike. This in turn allows us to construct a model on how the shape of the spikes from a given unit depends on the unit's firing rate. Spikes are then scored and assigned to units in an iterative manner, where after each iteration new models are formed from the labels of the last iteration, firing rates are estimated, and again spikes are scored.
 
More details coming soon.

## 2do for now
upload a sample recording on dandi
provide example usage
provide example output
make a package on pypi?

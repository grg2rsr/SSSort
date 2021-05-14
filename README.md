# SSSort
A spike sorting algorithm for single sensillum recordings
 
## Author
Georg Raiser, PhD, Champalimaud Research, grg2rsr@gmail.com
 
## Summary
_SSSort_ is a spike sorting algorithm for single sensillum recordings (SSR). _SSSort_ is the successor of [_SeqPeelSort_](https://github.com/grg2rsr/SeqPeelSort). Both algorithms have been designed to sort spikes from SSR into units, adressing the challenge that spike shape in SSR can change quite a lot, depending on the unit's firing rate. However, the core of both algorithms is quite different. While _SeqPeelSort_ tries to estimate the distribution of possible spike shapes and attempts to template match many different spike shapes drawn from those estimated distributions, _SSSort_ forms an explicit model on how the firing rate impacts the spike shape. Although developed for data from insect single sensillum recordings, _SSSort_ can probably be used for any type of recordings in which spike shape changes as a function of firing rate, however currently only single electrode recordings are implemented.
 
## Installation and Usage
A more in depth guide is coming soon, but following the instructions from [SeqPeelSort](https://github.com/grg2rsr/SeqPeelSort) will do.
 
## Algorithm details
_SSSort_ detects all spiky deviations within a recording (the _spikes_) and performs an initial clustering (k means). As now each spike belongs to a cluster grouping them into units, the local firing rates can be estimated. This gives a single value of the local firing rate at each timepoint of a spike. This in turn allows us to construct a model on how the shape of the spikes from a given unit depends on the unit's firing rate. Spikes are then scored and assigned to units in an iterative manner, where after each iteration new models are formed from the labels of the last iteration, firing rates are estimated, and again spikes are scored.
 
More details coming soon.


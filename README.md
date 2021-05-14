# SSSort
A spike sorting algorithm for single sensillum recordings

## Author
Georg Raiser, PhD, Champalimaud Research, grg2rsr@gmail.com

## Summary
_SSSort_ is a spike sorting algorithm for single sensillum recordings (SSR). SSSort is the sucessor of [SeqPeelSort](https://github.com/grg2rsr/SeqPeelSort) and contains parts of it's original code, however, the core of the algorithm is now a completely different one and have been completely rewritten. _SSSort_ forms an explicit model on how spike shape depends on the local firing rate of the unit, and uses predicted spike shapes from each unit for scoring individual spikes. Although developed for data from insect single sensillum recordings, _SSSort_ can probably be used for any type of recordings in which spike shape changes as a function of firing rate, however currently only single electrode recordings are implemented.

## Installation and Usage
A more in depth guide is coming soon, but following the instructions from [SeqPeelSort](https://github.com/grg2rsr/SeqPeelSort) will do.

## Algorithm details
_SSSort_ detects all spiky deviations within a recording (the _spikes_) and performs an initial clustering (kmeans). As now each spike belongs to a cluster grouping them into units, the local firing rates can be estimated. This gives a single value of the local firing rate at each timepoint of a spike. This in turn allows to construct a model on how the shape of the spikes from a given unit depend on the units firing rate. Spikes are then scored and assigned to units in an iterative manner, where after each iteration new models are formed from the labels of the last iteration, firing rates are estimated, and again spikes are scored.

More details coming soon.
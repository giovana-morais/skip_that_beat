"""
Data sequence handling

In order to be able to train our network, we need to provide the tracks in a way the network can deal with it.

Since our model needs to be able to process sequences (i.e. songs) of variable length, we use a batch size of 1.
Thus, no padding of the sequences is needed and we can simply iterate over them.

As features, we use a spectrogram representation, as targets we use the `beats`, `downbeats`, and `tempo` annotations of the songs.
Tempo information is always computed from the beats.

Beats and downbeats are one hot-encoded, i.e. frames representing a (down-)beat have a value of 1, all non-beat frames a value of 0.

Tempo is encoded as a vector with the bin representing the target tempo in bpm (beats per minute) having a value of 1, all other 0.

If there is no downbeat information, we mask the targets (all values set to -1).
This way the error can be ignored and is not backpropagated when updating the weights.

To improve training accuracy and speed, we "widen" the targets, i.e. give the neighbouring frames / tempo bins a value in between 0 and 1.
"""
import sys

import keras
import madmom
import numpy as np

from keras.utils import Sequence
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter1d
from scipy.signal import argrelmax

from constants import *
from preprocessor import *

# pad features
def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = np.repeat(data[:1], pad_frames, axis=0)
    pad_stop = np.repeat(data[-1:], pad_frames, axis=0)
    return np.concatenate((pad_start, data, pad_stop))

# infer (global) tempo from beats
def infer_tempo(beats, hist_smooth=15, fps=FPS, no_tempo=MASK_VALUE):
    ibis = np.diff(beats) * fps
    bins = np.bincount(np.round(ibis).astype(int))
    # if no beats are present, there is no tempo
    if not bins.any():
        return NO_TEMPO
    intervals = np.arange(len(bins))
    # smooth histogram bins
    if hist_smooth > 0:
        bins = madmom.audio.signal.smooth(bins, hist_smooth)
    # create interpolation function
    interpolation_fn = interp1d(intervals, bins, "quadratic")
    # generate new intervals with 1000x the resolution
    intervals = np.arange(intervals[0], intervals[-1], 0.001)
    tempi = 60.0 * fps / intervals
    # apply quadratic interpolation
    bins = interpolation_fn(intervals)
    peaks = argrelmax(bins, mode="wrap")[0]
    if len(peaks) == 0:
        # no peaks, no tempo
        return no_tempo
    else:
        # report only the strongest tempo
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        return tempi[sorted_peaks][0]




# wrap training/test data as a Keras sequence so we can use it with fit_generator()
class DataSequence(Sequence):
    def __init__(self, tracks, pre_processor, num_tempo_bins=300, pad_frames=None):
        # store features and targets in dictionaries with name of the song as key
        self.x = {}
        self.beats = {}
        self.downbeats = {}
        self.tempo = {}
        self.pad_frames = pad_frames
        self.ids = []
        # iterate over all tracks
        for i, key in enumerate(tracks):
            sys.stderr.write(f"\rprocessing track {i + 1}/{len(tracks)}: {key + ' ' * 20}")
            sys.stderr.flush()
            t = tracks[key]
            try:
                # use track only if it contains beats
                beats = t.beats.times
                # wrap librosa wav data & sample rate as Signal
                s = madmom.audio.Signal(*t.audio)
                # compute features first to be able to quantize beats
                x = pre_processor(s)
                self.x[key] = x
                # quantize beats
                beats = madmom.utils.quantize_events(beats, fps=pre_processor.fps, length=len(x))
                self.beats[key] = beats
            except AttributeError:
                # no beats found, skip this file
                print(f"\r{key} has no beat information, skipping\n")
                continue
            # downbeats
            try:
                downbeats = t.beats.positions.astype(int) == 1
                downbeats = t.beats.times[downbeats]
                downbeats = madmom.utils.quantize_events(downbeats, fps=pre_processor.fps, length=len(x))
            except AttributeError:
                print(f"\r{key} has no downbeat information, masking\n")
                downbeats = np.ones(len(x), dtype="float32") * MASK_VALUE
            self.downbeats[key] = downbeats
            # tempo
            tempo = None
            try:
                # Note: to be able to augment a dataset, we need to scale the beat times
                tempo = infer_tempo(t.beats.times * pre_processor.fps / 100, fps=pre_processor.fps)
                tempo = keras.utils.to_categorical(int(np.round(tempo)), num_classes=num_tempo_bins)
            except IndexError:
                # tempo out of bounds (too high)
                print(f"\r{key} has no valid tempo ({tempo}), masking\n")
                tempo = np.ones(num_tempo_bins, dtype="float32") * MASK_VALUE
            self.tempo[key] = tempo
            # keep track of IDs
            self.ids.append(key)
        assert len(self.x) == len(self.beats) == len(self.downbeats) == len(self.tempo) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # convert int idx to key
        if isinstance(idx, int):
            idx = self.ids[idx]
        # Note: we always use a batch size of 1 since the tracks have variable length
        #       keras expects the batch to be the first dimension, the prepend an axis;
        #       append an axis to beats and downbeats as well
        # define targets
        y = {}
        y["beats"] = self.beats[idx][np.newaxis, ..., np.newaxis]
        y["downbeats"] = self.downbeats[idx][np.newaxis, ..., np.newaxis]
        y["tempo"] = self.tempo[idx][np.newaxis, ...]
        # add context to frames
        x = self.x[idx]
        if self.pad_frames:
            x = cnn_pad(x, self.pad_frames)
        return x[np.newaxis, ..., np.newaxis], y

    def widen_beat_targets(self, size=3, value=0.5):
        for y in self.beats.values():
            # skip masked beat targets
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def widen_downbeat_targets(self, size=3, value=0.5):
        for y in self.downbeats.values():
            # skip masked downbeat targets
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def widen_tempo_targets(self, size=3, value=0.5):
        for y in self.tempo.values():
            # skip masked tempo targets
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def append(self, other):
        assert not any(key in self.ids for key in other.ids), "IDs must be unique"
        self.x.update(other.x)
        self.beats.update(other.beats)
        self.downbeats.update(other.downbeats)
        self.tempo.update(other.tempo)
        self.ids.extend(other.ids)


def create_data_sequence(all_tracks, split_tracks, data_augmentation=False):
    pad_frames = 2
    pre_processor = PreProcessor()

    split = DataSequence(
        tracks={k: v for k, v in all_tracks.items() if k in split_tracks},
        pre_processor=pre_processor,
        pad_frames=pad_frames
    )
    split.widen_beat_targets()
    split.widen_downbeat_targets()
    split.widen_tempo_targets()
    split.widen_tempo_targets()

    if data_augmentation:
        for fps in [95, 97.5, 102.5, 105]:
            ds = DataSequence(
                tracks={f"{k}_{fps}": v for k, v in tracks.items() if k in train_files},
                pre_processor=PreProcessor(fps=fps),
                pad_frames=pad_frames,
            )
            ds.widen_beat_targets()
            ds.widen_downbeat_targets()
            ds.widen_tempo_targets(3, 0.5)
            ds.widen_tempo_targets(3, 0.5)
            split.append(ds)

    return split

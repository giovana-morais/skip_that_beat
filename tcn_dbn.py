"""
Loads model and run postprocessing with DBN.
"""
import os
import pickle
import sys
import warnings

import keras
import librosa
import librosa.display
import madmom
import matplotlib.pyplot as plt
import mirdata
import numpy as np
import tensorflow as tf
from scipy.ndimage import maximum_filter1d
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

from keras.utils import Sequence
from madmom.processors import ParallelProcessor, SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor

sys.path.append("..")
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
PATH = "/media/gigibs/DD02EEEC68459F17/datasets/"
MASK_VALUE = -1

experiment = sys.argv[1]
outdir = f"./models/{experiment}"

datasets = [
    # "gtzan", "gtzan_augmented/24", "gtzan_augmented/34",
    # "beatles", "beatles_augmented/24", "beatles_augmented/34",
    # "rwcc", "rwcc_augmented/24", "rwcc_augmented/34",
    # "rwcj", "rwcj_augmented/24", "rwcj_augmented/34"
    "brid"
]

tracks = {}
for d in datasets:
    d = utils.custom_dataset_loader(
            path = PATH,
            folder = "",
            dataset_name = d
        )
    tracks = tracks | d.load_tracks()
    print(len(tracks))

model = keras.models.load_model(f"{outdir}/model_final.keras", compile=False)
full_test_files = utils.get_split_tracks(f"data/splits/brid.txt")
# full_test_files = utils.get_split_tracks(f"data/splits/{experiment}_test.txt")
test_files = [
    os.path.splitext(os.path.basename(i))[0] for i in full_test_files
]

FPS = 100
FFT_SIZE = 2048
NUM_BANDS = 12

beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
    min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=100, threshold=0.05
)

# track downbeats with a DBN
# as input, use a combined beat & downbeat activation function
downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
    beats_per_bar=[2, 3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=100
)

# track bars, i.e. first track the beats and then infer the downbeat positions
bar_tracker = madmom.features.downbeats.DBNBarTrackingProcessor(
    beats_per_bar=(2, 3, 4), meter_change_prob=1e-3, observation_weight=4
)

def detect_tempo(bins, hist_smooth=11, min_bpm=10):
    min_bpm = int(np.floor(min_bpm))
    tempi = np.arange(min_bpm, len(bins))
    bins = bins[min_bpm:]
    # smooth histogram bins
    if hist_smooth > 0:
        bins = madmom.audio.signal.smooth(bins, hist_smooth)
    # create interpolation function
    interpolation_fn = interp1d(tempi, bins, "quadratic")
    # generate new intervals with 1000x the resolution
    tempi = np.arange(tempi[0], tempi[-1], 0.001)
    # apply quadratic interpolation
    bins = interpolation_fn(tempi)
    peaks = argrelmax(bins, mode="wrap")[0]
    if len(peaks) == 0:
        # no peaks, no tempo
        tempi = np.array([], ndmin=2)
    elif len(peaks) == 1:
        # report only the strongest tempo
        ret = np.array([tempi[peaks[0]], 1.0])
        tempi = np.array([tempi[peaks[0]], 1.0])
    else:
        # sort the peaks in descending order of bin heights
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        # normalize their strengths
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        # return the tempi and their normalized strengths
        ret = np.array(list(zip(tempi[sorted_peaks], strengths)))
        tempi = np.array(list(zip(tempi[sorted_peaks], strengths)))
    return tempi[:2]


# function to predict the model"s output(s), post-process everything and save if needed
def predict(model, dataset, detdir=None, activations={}, detections={}):

    for i, t in enumerate(dataset):
        # file name
        f = dataset.ids[i]
        # print progress
        sys.stderr.write("\rprocessing file %d of %d: %12s" % (i + 1, len(dataset), f))
        sys.stderr.flush()
        # predict activations
        x = t[0]
        beats, downbeats, tempo = model.predict(x)
        beats_act = beats.squeeze()
        downbeats_act = downbeats.squeeze()
        tempo_act = tempo.squeeze()
        # beats
        beats = beat_tracker(beats_act)
        # downbeats
        combined_act = np.vstack((np.maximum(beats_act - downbeats_act, 0), downbeats_act)).T
        downbeats = downbeat_tracker(combined_act)
        # bars (i.e. track beats and then downbeats)
        beat_idx = (beats * FPS).astype(np.int64)
        bar_act = maximum_filter1d(downbeats_act, size=3)
        bar_act = bar_act[beat_idx]
        bar_act = np.vstack((beats, bar_act)).T
        try:
            bars = bar_tracker(bar_act)
        except IndexError:
            bars = np.empty((0, 2))
        # tempo
        tempo = detect_tempo(tempo_act)

        # collect activations and detections
        activations[f] = {"beats": beats_act, "downbeats": downbeats_act, "combined": combined_act, "tempo": tempo_act}
        detections[f] = {"beats": beats, "downbeats": downbeats, "bars": bars, "tempo": tempo}

        # save activations & detections
        if detdir is not None:
            os.makedirs(detdir, exist_ok=True)
            np.save("%s/%s.beats.npy" % (detdir, f), beats_act)
            np.save("%s/%s.downbeats.npy" % (detdir, f), downbeats_act)
            np.save("%s/%s.tempo.npy" % (detdir, f), tempo_act)
            madmom.io.write_beats(beats, "%s/%s.beats.txt" % (detdir, f))
            madmom.io.write_beats(downbeats, "%s/%s.downbeats.txt" % (detdir, f))
            madmom.io.write_beats(bars, "%s/%s.bars.txt" % (detdir, f))
            madmom.io.write_tempo(tempo, "%s/%s.bpm.txt" % (detdir, f))

    return activations, detections

# define pre-processor
class PreProcessor(SequentialProcessor):
    def __init__(self, frame_size=FFT_SIZE, num_bands=NUM_BANDS, log=np.log, add=1e-6, fps=FPS):
        # resample to a fixed sample rate in order to get always the same number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # split audio signal in overlapping frames
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        # compute STFT
        stft = ShortTimeFourierTransformProcessor()
        # filter the magnitudes
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        # scale them logarithmically
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)
        # instantiate a SequentialProcessor
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))
        # safe fps as attribute (needed for quantization of events)
        self.fps = fps


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


# pad features
def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = np.repeat(data[:1], pad_frames, axis=0)
    pad_stop = np.repeat(data[-1:], pad_frames, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


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

pad_frames = 2
pre_processor = PreProcessor()

test = DataSequence(
    tracks={k: v for k, v in tracks.items() if k in test_files},
    pre_processor=pre_processor,
    pad_frames=pad_frames
)
test.widen_beat_targets()
test.widen_downbeat_targets()
test.widen_tempo_targets()
test.widen_tempo_targets()


# Predict the model"s activations (i.e. the raw outputs) and the post-processed detections.
# create a directory to put the activations and detections into
detdir = f"{outdir}/detections/"

# we can use validation set as test set, since we did not use it for anything besides monitoring progress
activations, detections = predict(model, test, detdir)


# # Evaluation
#
def evaluate_beats(detections, annotations):
    evals = []
    for key, det in detections.items():
        ann = annotations[key]
        e = madmom.evaluation.beats.BeatEvaluation(det, ann)
        evals.append(e)
    return madmom.evaluation.beats.BeatMeanEvaluation(evals)


def evaluate_downbeats(detections, annotations):
    evals = []
    for key, det in detections.items():
        ann = annotations[key]
        e = madmom.evaluation.beats.BeatEvaluation(det, ann, downbeats=True)
        evals.append(e)
    return madmom.evaluation.beats.BeatMeanEvaluation(evals)


def evaluate_tempo(detections, annotations):
    evals = []
    for key, det in detections.items():
        ann = annotations[key]
        e = madmom.evaluation.tempo.TempoEvaluation(det, ann)
        evals.append(e)
    return madmom.evaluation.tempo.TempoMeanEvaluation(evals)


# ## Evaluate beats & tempo
beat_detections = {k: v["beats"] for k, v in detections.items()}
downbeat_detections = {k: v["downbeats"] for k, v in detections.items()}
bar_detections = {k: v["bars"] for k, v in detections.items()}
#tempo_detections = {k: v["tempo"][0, 0] for k, v in detections.items()}

beat_annotations = {k: v.beats.times for k, v in tracks.items() if v.beats is not None}
downbeat_annotations = {k: v.beats.times[v.beats.positions == 1] for k, v in tracks.items() if v.beats is not None}
#tempo_annotations = {k: v.tempo for k, v in tracks.items() if v.tempo is not None}

# evaluate beats
print("Beat evaluation\n---------------")
print(" Beat tracker:    ", evaluate_beats(beat_detections, beat_annotations))
print(" Downbeat tracker:", evaluate_beats(downbeat_detections, beat_annotations))

# evaluate downbeats
print("\nDownbeat evaluation\n-------------------")
print(" Bar tracker:     ", evaluate_downbeats(bar_detections, downbeat_annotations))
print(" Downbeat tracker:", evaluate_downbeats(downbeat_detections, downbeat_annotations))

#!/usr/bin/env python
# coding: utf-8

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

sys.path.append("..")
import utils

experiment = "augmented_full"

# ignore certain warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

PATH = "/media/gigibs/DD02EEEC68459F17/datasets/"

datasets = [
    "gtzan", "gtzan_augmented/24", "gtzan_augmented/34",
    "beatles", "beatles_augmented/24", "beatles_augmented/34",
    "rwcc", "rwcc_augmented/24", "rwcc_augmented/34",
    "rwcj", "rwcj_augmented/24", "rwcj_augmented/34"
]

# tracks = utils.multi_dataset_loader(PATH, datasets)
tracks = {}
for d in datasets:
    d = utils.custom_dataset_loader(
            path = PATH,
            folder = "",
            dataset_name = d
        )
    tracks = tracks | d.load_tracks()
    print(len(tracks))

full_train_files = utils.get_split_tracks(f"data/splits/{experiment}_train.txt")
full_validation_files = utils.get_split_tracks(f"data/splits/{experiment}_val.txt")
full_test_files = utils.get_split_tracks(f"data/splits/{experiment}_test.txt")

train_files = [
    os.path.splitext(os.path.basename(i))[0] for i in full_train_files
]
validation_files = [
    os.path.splitext(os.path.basename(i))[0] for i in full_validation_files
]
test_files = [
    os.path.splitext(os.path.basename(i))[0] for i in full_test_files
]

print(f"train: {len(train_files)}")
print(f"validation: {len(validation_files)}")
print(f"test: {len(test_files)}")

# # Audio pre-processing
#
# Our approach operates on a spectrogram representation of the audio signal.
#
# We define a processor which transforms the raw audio into a spectrogram with 100 frames per second and 81 frequency bins.

from madmom.processors import ParallelProcessor, SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor

FPS = 100
FFT_SIZE = 2048
NUM_BANDS = 12

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


# # NN infrastructure
#
# We create a multi-task model to jointly predict tempo, beats and doenbeats, which mostly follows our ISMIR 2020 paper "Deconstruct, analyse, reconstruct: how to improve tempo, beat, and downbeat estimation.".

import tensorflow.keras.backend as K

from keras.models import Sequential, Model
from keras.layers import (
    Activation,
    Dense,
    Input,
    Conv1D,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Dropout,
    SpatialDropout1D,
    GaussianNoise,
    GlobalAveragePooling1D,
)

# from keras.legacy import interfaces
from keras.utils import Sequence
# from keras.optimizers import Optimizer


# ## TCN network layers
#
# The heart of the network is a TCN (temporal convolutional network) with 11 TCN layers with increasing dilation rates.
#
# The structure is as follows:
# ![Multi-task TCN structure](https://docs.google.com/uc?export=download&id=1Mt-lig8CFmMRrSjbIF-DUhaBivIHZUtk)

def residual_block(x, i, activation, num_filters, kernel_size, padding, dropout_rate=0, name=""):
    # name of the layer
    name = name + "_dilation_%d" % i
    # 1x1 conv. of input (so it can be added as residual)
    res_x = Conv1D(num_filters, 1, padding="same", name=name + "_1x1_conv_residual")(x)
    # two dilated convolutions, with dilation rates of i and 2i
    conv_1 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + "_dilated_conv_1",
    )(x)
    conv_2 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i * 2,
        padding=padding,
        name=name + "_dilated_conv_2",
    )(x)
    # concatenate the output of the two dilations
    concat = keras.layers.concatenate([conv_1, conv_2], name=name + "_concat")
    # apply activation function
    x = Activation(activation, name=name + "_activation")(concat)
    # apply spatial dropout
    x = SpatialDropout1D(dropout_rate, name=name + "_spatial_dropout_%f" % dropout_rate)(x)
    # 1x1 conv. to obtain a representation with the same size as the residual
    x = Conv1D(num_filters, 1, padding="same", name=name + "_1x1_conv")(x)
    # add the residual to the processed data and also return it as skip connection
    return keras.layers.add([res_x, x], name=name + "_merge_residual"), x


class TCN:
    def __init__(
        self,
        num_filters=20,
        kernel_size=5,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        activation="elu",
        padding="same",
        dropout_rate=0.15,
        name="tcn",
    ):
        self.name = name
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = padding

        if padding != "causal" and padding != "same":
            raise ValueError("Only \"causal\" or \"same\" padding are compatible for this layer.")

    def __call__(self, inputs):
        x = inputs
        # gather skip connections, each having a different context
        skip_connections = []
        # build the TCN models
        for i, num_filters in zip(self.dilations, self.num_filters):
            # feed the output of the previous layer into the next layer
            # increase dilation rate for each consecutive layer
            x, skip_out = residual_block(
                x, i, self.activation, num_filters, self.kernel_size, self.padding, self.dropout_rate, name=self.name
            )
            # collect skip connection
            skip_connections.append(skip_out)
        # activate the output of the TCN stack
        x = Activation(self.activation, name=self.name + "_activation")(x)
        # merge the skip connections by simply adding them
        skip = keras.layers.add(skip_connections, name=self.name + "_merge_skip_connections")
        return x, skip


# ## Multi-task model
#
# The network to be trained consists of two main parts.
#  1. a stack of convolutional layers, and
#  2. the TCN itself.
#
# The former is used to learn meaningful local features, whereas the latter learns the temporal dependencies of these (local) features.


def create_model(input_shape, num_filters=20, num_dilations=11, kernel_size=5, activation="elu", dropout_rate=0.15):
    # input layer
    input_layer = Input(shape=input_shape)

    # stack of 3 conv layers, each conv, activation, max. pooling & dropout
    conv_1 = Conv2D(num_filters, (3, 3), padding="valid", name="conv_1_conv")(input_layer)
    conv_1 = Activation(activation, name="conv_1_activation")(conv_1)
    conv_1 = MaxPooling2D((1, 3), name="conv_1_max_pooling")(conv_1)
    conv_1 = Dropout(dropout_rate, name="conv_1_dropout")(conv_1)

    conv_2 = Conv2D(num_filters, (1, 10), padding="valid", name="conv_2_conv")(conv_1)
    conv_2 = Activation(activation, name="conv_2_activation")(conv_2)
    conv_2 = MaxPooling2D((1, 3), name="conv_2_max_pooling")(conv_2)
    conv_2 = Dropout(dropout_rate, name="conv_2_dropout")(conv_2)

    conv_3 = Conv2D(num_filters, (3, 3), padding="valid", name="conv_3_conv")(conv_2)
    conv_3 = Activation(activation, name="conv_3_activation")(conv_3)
    conv_3 = MaxPooling2D((1, 3), name="conv_3_max_pooling")(conv_3)
    conv_3 = Dropout(dropout_rate, name="conv_3_dropout")(conv_3)

    # reshape layer to reduce dimensions
    x = Reshape((-1, num_filters), name="tcn_input_reshape")(conv_3)

    # TCN layers
    dilations = [2 ** i for i in range(num_dilations)]
    tcn, skip = TCN(
        num_filters=[num_filters] * len(dilations),
        kernel_size=kernel_size,
        dilations=dilations,
        activation=activation,
        padding="same",
        dropout_rate=dropout_rate,
    )(x)

    # output layers; beats & downbeats use TCN output, tempo the skip connections
    beats = Dropout(dropout_rate, name="beats_dropout")(tcn)
    beats = Dense(1, name="beats_dense")(beats)
    beats = Activation("sigmoid", name="beats")(beats)

    downbeats = Dropout(dropout_rate, name="downbeats_dropout")(tcn)
    downbeats = Dense(1, name="downbeats_dense")(downbeats)
    downbeats = Activation("sigmoid", name="downbeats")(downbeats)

    tempo = Dropout(dropout_rate, name="tempo_dropout")(skip)
    tempo = GlobalAveragePooling1D(name="tempo_global_average_pooling")(tempo)
    tempo = GaussianNoise(dropout_rate, name="tempo_noise")(tempo)
    tempo = Dense(300, name="tempo_dense")(tempo)
    tempo = Activation("softmax", name="tempo")(tempo)

    # instantiate a Model and return it
    return Model(input_layer, outputs=[beats, downbeats, tempo])


# ## Data sequence handling
#
# In order to be able to train our network, we need to provide the tracks in a way the network can deal with it.
#
# Since our model needs to be able to process sequences (i.e. songs) of variable length, we use a batch size of 1.
# Thus, no padding of the sequences is needed and we can simply iterate over them.
#
# As features, we use a spectrogram representation, as targets we use the `beats`, `downbeats`, and `tempo` annotations of the songs.
# Tempo information is always computed from the beats.
#
# Beats and downbeats are one hot-encoded, i.e. frames representing a (down-)beat have a value of 1, all non-beat frames a value of 0.
#
# Tempo is encoded as a vector with the bin representing the target tempo in bpm (beats per minute) having a value of 1, all other 0.
#
# If there is no downbeat information, we mask the targets (all values set to -1).
# This way the error can be ignored and is not backpropagated when updating the weights.
#
# To improve training accuracy and speed, we "widen" the targets, i.e. give the neighbouring frames / tempo bins a value in between 0 and 1.


MASK_VALUE = -1

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


# pad features
def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = np.repeat(data[:1], pad_frames, axis=0)
    pad_stop = np.repeat(data[-1:], pad_frames, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


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


# ## Loss & metrics
#
# We train our model with cross entropy, but need to mask the loss function and metrics if the targets are set to be ignored (i.e. no tempo / downbeat information at hand).


# https://github.com/keras-team/keras/issues/3893
def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def masked_accuracy(y_true, y_pred):
    total = K.sum(K.not_equal(y_true, MASK_VALUE))
    correct = K.sum(K.equal(y_true, K.round(y_pred)))
    return correct / total


# # Train network

# ## Training & testing sequences
#
# We wrap our previously split dataset as `DataSequences`.
#
# We widen the beat and downbeat targets to have a value of 0.5 at the frames next to the annotated beat locations.
#
# We assign tempo values ±1 bpm apart a value of 0.5, and those ±2bpm a value of0.25.

pad_frames = 2
pre_processor = PreProcessor()

train = DataSequence(
    tracks={k: v for k, v in tracks.items() if k in train_files},
    pre_processor=pre_processor,
    pad_frames=pad_frames
)
train.widen_beat_targets()
train.widen_downbeat_targets()
train.widen_tempo_targets()
train.widen_tempo_targets()

# # data augmentation
# for fps in [95, 97.5, 102.5, 105]:
#     ds = DataSequence(
#         tracks={f"{k}_{fps}": v for k, v in tracks.items() if k in train_files},
#         pre_processor=PreProcessor(fps=fps),
#         pad_frames=pad_frames,
#     )
#     ds.widen_beat_targets()
#     ds.widen_downbeat_targets()
#     ds.widen_tempo_targets(3, 0.5)
#     ds.widen_tempo_targets(3, 0.5)
#     train.append(ds)

validation = DataSequence(
    tracks={k: v for k, v in tracks.items() if k in validation_files},
    pre_processor=pre_processor,
    pad_frames=pad_frames
)
validation.widen_beat_targets()
validation.widen_downbeat_targets()
validation.widen_tempo_targets()
validation.widen_tempo_targets()

# ## Put everything together
#
# Finally create the network, add an optimizer, loss functions and metrics to it, and compile the graph.

# use a training sample to infer input shape
input_shape = (None,) + train[0][0].shape[-2:]
model = create_model(input_shape)

learnrate = 0.005
clipnorm = 0.5

# optimizer = Lookahead(RAdam(lr=learnrate, clipnorm=clipnorm))
model.compile(
    optimizer="Adam",
    loss=[
        build_masked_loss(K.binary_crossentropy),
        build_masked_loss(K.binary_crossentropy),
        build_masked_loss(K.binary_crossentropy),
    ],
    metrics=["binary_accuracy"]
)
model.summary(200)


# ## Train network
#
# We train the network for 100 epochs, each epoch randomly iterating over all sequences (i.e. songs) of the dataset.
#
# Whenever a better performance is observed (i.e. lower loss), we store the model.
#
# If the performance does not increase for a certain number of epochs, we reduce the learnrate by a factor of 5.
# If no further improvement can be observed, we stop the training (early stopping).
#
# We log everything to a Tensorboard log.

epochs = 100
verbose = 0

# create output dir
outdir = f"./models/{experiment}"
os.makedirs(outdir, exist_ok=True)
print(f"output dir: {outdir}")

# model checkpointing
mc = keras.callbacks.ModelCheckpoint(f"{outdir}/model_best.h5", monitor="loss", save_best_only=True, verbose=verbose)

# learn rate scheduler
lr = keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.2, patience=10, verbose=1, mode="auto", min_delta=1e-3, cooldown=0, min_lr=1e-7
)

# early stopping
es = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=verbose)

# tensorboard logging
tb = keras.callbacks.TensorBoard(log_dir=f"{outdir}/logs", write_graph=True, write_images=True)

# actually train network
history = model.fit_generator(
    train,
    steps_per_epoch=len(train),
    epochs=epochs,
    shuffle=True,
    validation_data=validation,
    validation_steps=len(validation),
    callbacks=[mc, es, tb, lr],
)
model.save(f"{outdir}/model_final.keras")

with open(f"{outdir}/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

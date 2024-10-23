#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
import sys
import warnings

import keras
import librosa
import librosa.display
import madmom
import mirdata
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d, median_filter

from constants import *
from data_sequence import *
from model import *
from preprocessor import *

sys.path.append("..")
import utils

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


def predict_dbn(beat_activation, downbeat_activation):
    beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
        min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=100, threshold=0.05
    )
    downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[2, 3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=100
    )

    beats = beat_tracker(beat_activation)
    downbeats = downbeat_tracker(downbeat_activation)
    return beats, downbeats


def predict_pp(beat_activation, downbeat_activation):
    beats = peak_picking_MSFA(beat_activation)
    # convert back to seconds
    beats = beats/100.

    downbeats = peak_picking_MSFA(downbeat_activation)
    # convert back to seconds
    downbeats = downbeats/100.
    return beats, downbeats


def predict(model, dataset, detdir_dbn=None, detdir_pp=None, activations={},
        detections_dbn={}, detections_pp={}):
    for i, t in enumerate(dataset):
        # file name
        f = dataset.ids[i]
        print(f)
        # print progress
        sys.stderr.write("\rprocessing file %d of %d: %12s" % (i + 1, len(dataset), f))
        sys.stderr.flush()
        # predict activations
        x = t[0]
        beats, downbeats, tempo = model.predict(x)
        beats_act = beats.squeeze()
        downbeats_act = downbeats.squeeze()
        tempo_act = tempo.squeeze()

        # DBN beats
        combined_act = np.vstack((np.maximum(beats_act - downbeats_act, 0), downbeats_act)).T
        beats_dbn, downbeats_dbn = predict_dbn(beats_act, combined_act)
        # PP detections
        beats_pp, downbeats_pp = predict_pp(beats_act, downbeats_act)

        # collect activations and detections
        activations[f] = {"beats": beats_act, "downbeats": downbeats_act, "combined": combined_act}
        detections_dbn[f] = {"beats": beats_dbn, "downbeats": downbeats_dbn}
        detections_pp[f] = {"beats": beats_pp, "downbeats": downbeats_pp}

        # save activations & detections
        if detdir_dbn is not None:
            os.makedirs(detdir_dbn, exist_ok=True)
            np.save("%s/%s.beats.npy" % (detdir_dbn, f), beats_act)
            np.save("%s/%s.downbeats.npy" % (detdir_dbn, f), downbeats_act)
            madmom.io.write_beats(beats_dbn, "%s/%s.beats.txt" % (detdir_dbn, f))
            madmom.io.write_beats(downbeats_dbn, "%s/%s.downbeats.txt" % (detdir_dbn, f))

        if detdir_pp is not None:
            os.makedirs(detdir_pp, exist_ok=True)
            np.save("%s/%s.beats.npy" % (detdir_pp, f), beats_act)
            np.save("%s/%s.downbeats.npy" % (detdir_pp, f), downbeats_act)
            madmom.io.write_beats(beats_pp, "%s/%s.beats.txt" % (detdir_pp, f))
            madmom.io.write_beats(downbeats_pp, "%s/%s.downbeats.txt" % (detdir_pp, f))

    return activations, detections_dbn, detections_pp

def create_parser():
    """
    creates ArgumentParser
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_home",
        type=str,
        required=True,
        help="path for your datasets"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="which experiment you want to run. ['baseline', 'augmented_sampled', 'augmented_full']"
    )
    parser.add_argument(
        "--data_augmentation",
        type=bool,
        required=False,
        default=False,
        help="if set to True, performs on-the-fly augmentation to increase tempo range"
    )

    return parser

def get_tracks(experiment):
    full_train_files = utils.get_split_tracks(f"../splits/{experiment}_train.txt")
    full_validation_files = utils.get_split_tracks(f"../splits/{experiment}_val.txt")
    full_test_files = utils.get_split_tracks(f"../splits/{experiment}_test.txt")
    full_brid_files = utils.get_split_tracks(f"../splits/brid.txt")

    train_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_train_files
    ]
    validation_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_validation_files
    ]
    test_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_test_files
    ]
    brid_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_brid_files
    ]

    print(f"loaded train tracks: {len(train_tracks)}")
    print(f"loaded validation tracks: {len(validation_tracks)}")
    print(f"loaded test tracks: {len(test_tracks)}")
    print(f"loaded brid tracks: {len(brid_tracks)}")

    return train_tracks, validation_tracks, test_tracks, brid_tracks

def train_model(model, train, validation, epochs, experiment):
    learnrate = 0.005
    clipnorm = 0.5

    model.compile(
        optimizer="Adam",
        loss=[
            build_masked_loss(K.binary_crossentropy),
            build_masked_loss(K.binary_crossentropy),
            build_masked_loss(K.binary_crossentropy),
        ],
        metrics=["binary_accuracy"]
    )
    verbose = 0

    # model checkpointing
    mc = keras.callbacks.ModelCheckpoint(f"{output_path}/model_best.h5", monitor="loss", save_best_only=True, verbose=verbose)

    # learn rate scheduler
    lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=10, verbose=1, mode="auto", min_delta=1e-3, cooldown=0, min_lr=1e-7
    )

    # early stopping
    es = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=verbose)

    # tensorboard logging
    tb = keras.callbacks.TensorBoard(log_dir=f"{output_path}/logs", write_graph=True, write_images=True)

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

    return model, history

def peak_picking_MSFA(x, median_len=16, offset_rel=0.05, sigma=4.0):
    """
    Source: FMP notebooks
    ---
    Peak picking strategy following MSFA using an adaptive threshold (https://github.com/urinieto/msaf)

    Notebook: C6/C6S1_PeakPicking.ipynb

    Args:
        x (np.ndarray): Input function
        median_len (int): Length of media filter used for adaptive thresholding (Default value = 16)
        offset_rel (float): Additional offset used for adaptive thresholding (Default value = 0.05)
        sigma (float): Variance for Gaussian kernel used for smoothing the novelty function (Default value = 4.0)

    Returns:
        peaks (np.ndarray): Peak positions
    """
    offset = x.mean() * offset_rel
    x = gaussian_filter1d(x, sigma=sigma)
    threshold_local = median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks


def evaluate_detections(tracks, detections):
    beat_detections = {k: v["beats"] for k, v in detections.items()}
    downbeat_detections = {k: v["downbeats"] for k, v in detections.items()}
    # bar_detections = {k: v["bars"] for k, v in detections.items()}

    beat_annotations = {k: v.beats.times for k, v in tracks.items() if v.beats is not None}
    downbeat_annotations = {k: v.beats.times[v.beats.positions == 1] for k, v in tracks.items() if v.beats is not None}

    # evaluate beats
    print("Beat evaluation\n---------------")
    print(" Beat tracker:    ", evaluate_beats(beat_detections, beat_annotations))
    print(" Downbeat tracker:", evaluate_beats(downbeat_detections, beat_annotations))

    # evaluate downbeats
    print("\nDownbeat evaluation\n-------------------")
    # print(" Bar tracker:     ", evaluate_downbeats(bar_detections, downbeat_annotations))
    print(" Downbeat tracker:", evaluate_downbeats(downbeat_detections, downbeat_annotations))

    return


if __name__ == "__main__":
    args = create_parser().parse_args()

    # ignore certain warnings
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    args.data_home = "/media/gigibs/DD02EEEC68459F17/datasets/"
    datasets = [
        "gtzan", "gtzan_augmented/24", "gtzan_augmented/34",
        "beatles", "beatles_augmented/24", "beatles_augmented/34",
        "rwcc", "rwcc_augmented/24", "rwcc_augmented/34",
        "rwcj", "rwcj_augmented/24", "rwcj_augmented/34",
        "brid"
    ]

    tracks = {}
    for d in datasets:
        d = utils.custom_dataset_loader(
                path = args.data_home,
                folder = "",
                dataset_name = d
            )
        tracks = tracks | d.load_tracks()

    train_keys, validation_keys, test_keys, brid_keys = get_tracks(args.experiment)

    train = create_data_sequence(tracks, train_keys, args.data_augmentation)
    validation = create_data_sequence(tracks, validation_keys)
    test = create_data_sequence(tracks, test_keys)
    brid = create_data_sequence(tracks, brid_keys)

    epochs = 1

    # use a training sample to infer input shape
    input_shape = (None,) + train[0][0].shape[-2:]
    model = create_model(input_shape)

    # create output dir
    output_path = f"./models/{args.experiment}"
    os.makedirs(output_path, exist_ok=True)
    print(f"output dir: {output_path}")

    model, history = train_model(model, train, validation, epochs, output_path)
    model.save(f"{output_path}/model_final.keras")

    with open(f"{output_path}/history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    test_tracks = {k: tracks[k] for k in test_keys}
    brid_tracks = {k: tracks[k] for k in brid_keys}

    detections_dbn_path = f"{output_path}/detections_dbn/"
    detections_pp_path = f"{output_path}/detections_pp/"

    print("==========> Test set")
    test_activations, test_dbn_detections, test_pp_detections = predict(model,
            test, detections_dbn_path, detections_pp_path)
    print("\nTCN-DBN")
    evaluate_detections(test_tracks, test_dbn_detections)
    print("\nTCN-PP")
    evaluate_detections(test_tracks, test_pp_detections)

    print("\n==========> BRID")
    brid_activations, brid_dbn_detections, brid_pp_detections = predict(model,
            brid, detections_dbn_path, detections_pp_path)
    print("\nTCN-DBN")
    evaluate_detections(brid_tracks, brid_dbn_detections)
    print("\nTCN-PP")
    evaluate_detections(brid_tracks, brid_pp_detections)

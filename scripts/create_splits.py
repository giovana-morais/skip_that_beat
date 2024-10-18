"""
create splits for training the model.

train/validation should belong to the same set of data
test should be data from different dataset
"""
import sys
sys.path.append("..")

import argparse
import os
import random
from collections import Counter

import mirdata
import numpy as np

import utils

random.seed(42)

def save_splits(output_path, prefix, train, val, test=None):
    """
    helper function to write splits to .txt file
    """

    os.makedirs(output_path, exist_ok=True)

    print(f"writing {prefix}_train.txt")
    with open(os.path.join(output_path, f"{prefix}_train.txt"), "w") as f:
        for i in train:
            f.write(i + "\n")

    print(f"writing {prefix}_val.txt")
    with open(os.path.join(output_path, f"{prefix}_val.txt"), "w") as f:
        for i in val:
            f.write(i + "\n")

    if test is not None:
        print(f"writing {prefix}_test.txt")
        with open(os.path.join(output_path, f"{prefix}_test.txt"), "w") as f:
            for i in test:
                f.write(i + "\n")

    return


def load_meter(dataset, include, print_stats=False):
    """
    load tracks and their time signatures

    args
    ---
    datasets : Dataset
    include : list
       list with meters to include

    return
    ---
    track_meter : dict
        track and the most common meter
    """
    meters = {}

    for t in dataset.track_ids:
        tid = dataset.track(t)
        tid_fullpath = dataset.track(t).audio_path

        try:
            tid_meter = dataset.track(t).meter
            if tid_meter in include:
                meters[tid_fullpath] = tid_meter
        except (AttributeError, ValueError, FileNotFoundError):
            print(f"track {t} has no meter information. skipping")
            continue
        except IndexError:
            print(f"track {t} has meter information. skipping")
            continue

    if print_stats:
        print(Counter(meters.values()))

    return meters


def baseline_split(train_tracks, test_tracks):
    """
    creates baseline split with 4/4 tracks only

    args
    ---
    train_tracks : list
        list with train tracks
    test_tracks : dict
        list with test tracks ids

    return
    ---
    train : list
        list with train tracks
    val : list
        list with validation tracks
    test : list
        list with test tracks
    """
    train_split_size = int(0.8*len(train_tracks))
    val_split_size = len(train_tracks)-train_split_size

    train = random.sample(train_tracks, k=train_split_size)
    val = set(train_tracks) - set(train)
    test = test_tracks

    print(len(train), len(val), len(test))

    return list(train), list(val), test


def augmented_full_split(train, val, test, aug24, aug34):
    """
    creates splits with redundancy. we have all the original tracks + all
    augmentations for this track.

    args
    ---
    train : list
        baseline train tracks
    val : list
        baseline val tracks
    test : list
        baseline test tracks

    return
    ---
    aug_train : list
        augmented train tracks
    aug_val : list
        augmented val tracks
    aug_test : list
        augmented test tracks
    """
    aug_train = []
    aug_val = []
    aug_test = []

    for i in train:
        tid = os.path.basename(i).replace(".wav", "")
        aug_train.append(i)
        aug_train.append(aug24[f"{tid}_24"].audio_path)
        aug_train.append(aug34[f"{tid}_34"].audio_path)

    for i in val:
        tid = os.path.basename(i).replace(".wav", "")
        aug_val.append(i)
        aug_val.append(aug24[f"{tid}_24"].audio_path)
        aug_val.append(aug34[f"{tid}_34"].audio_path)

    # test remains the same
    for i in test:
        aug_test.append(i)

    return aug_train, aug_val, aug_test

def augmented_sampled_split(train, val, test, aug24, aug34):
    """
    creates splits with the same size of baseline, but instead of all 4/4
    tracks, we sample between original and augmented tracks. therefore, this
    dataset has equal representation between each time signature

    args
    ---
    train : list
        baseline train tracks
    val : list
        baseline val tracks
    test : list
        baseline test tracks

    return
    ---
    aug_train : list
        augmented train tracks
    aug_val : list
        augmented val tracks
    aug_test : list
        augmented val tracks
    """

    aug_train = []
    aug_val = []
    aug_test = []

    for i in train:
        tid = os.path.basename(i).replace(".wav", "")
        choice = random.choice(["24", "34", "original"])
        if choice == "original":
            aug_train.append(i)
        elif choice == "24":
            aug_train.append(aug24[f"{tid}_24"].audio_path)
        elif choice == "34":
            aug_train.append(aug34[f"{tid}_34"].audio_path)

    for i in val:
        tid = os.path.basename(i).replace(".wav", "")
        choice = random.choice(["24", "34", "original"])
        if choice == "original":
            aug_val.append(i)
        elif choice == "24":
            aug_val.append(aug24[f"{tid}_24"].audio_path)
        elif choice == "34":
            aug_val.append(aug34[f"{tid}_34"].audio_path)

    # test remains the same
    for i in test:
        aug_test.append(i)

    return aug_train, aug_val, aug_test

if __name__ == "__main__":
    DATA_PATH = "/media/gigibs/DD02EEEC68459F17/datasets"
    SPLIT_PATH = "/home/gigibs/Documents/meter_estimation/meter_augmentation/data/splits"

    gtzan = utils.custom_dataset_loader(DATA_PATH, "gtzan", "")
    rwcj = utils.custom_dataset_loader(DATA_PATH, "rwcj", "")
    rwcc = utils.custom_dataset_loader(DATA_PATH, "rwcc", "")
    beatles = utils.custom_dataset_loader(DATA_PATH, "beatles", "")

    # loading meter info
    print("GTZAN meter distribution")
    gtzan_meter = load_meter(gtzan, ["4/4", "3/4", "2/4"], print_stats=True)
    print("RWC Jazz meter distribution")
    rwcj_meter = load_meter(rwcj, ["4/4", "3/4", "2/4"], print_stats=True)
    print("RWC Classical meter distribution")
    rwcc_meter = load_meter(rwcc, ["4/4", "3/4", "2/4"], print_stats=True)
    print("Beatles meter distribution")
    beatles_meter = load_meter(beatles, ["4/4", "3/4", "2/4"], print_stats=True)

    # concatenate everything
    original_data = gtzan_meter | rwcj_meter | rwcc_meter | beatles_meter

    # shuffling files
    l = list(original_data.items())
    random.shuffle(l)
    original_data = dict(l)

    print("full dataset meter distribution")
    print(Counter(original_data.values()))

    train_size = int(0.8*len(original_data))

    train_tracks = []
    test_tracks = []

    train_counter = 0
    for track, meter in original_data.items():
        if meter == "4/4" and len(train_tracks) < train_size:
            train_tracks.append(track)
        else:
            test_tracks.append(track)


    baseline_train, baseline_val, baseline_test = baseline_split(train_tracks, test_tracks)
    save_splits(SPLIT_PATH, "baseline", baseline_train, baseline_val, baseline_test)

    gtzan_24 = utils.custom_dataset_loader(DATA_PATH, folder="gtzan_augmented",
            dataset_name="24")
    rwcj_24 = utils.custom_dataset_loader(DATA_PATH,
            folder="rwcj_augmented", dataset_name="24")
    rwcc_24 = utils.custom_dataset_loader(DATA_PATH,
            folder="rwcc_augmented", dataset_name="24")
    beatles_24 = utils.custom_dataset_loader(DATA_PATH,
            folder="beatles_augmented", dataset_name="24")

    aug24_data = gtzan_24.load_tracks() | rwcj_24.load_tracks() | rwcc_24.load_tracks() | beatles_24.load_tracks()

    gtzan_34 = utils.custom_dataset_loader(DATA_PATH, folder="gtzan_augmented", dataset_name="34")
    rwcj_34 = utils.custom_dataset_loader(DATA_PATH,
            folder="rwcj_augmented", dataset_name="34")
    rwcc_34 = utils.custom_dataset_loader(DATA_PATH,
            folder="rwcc_augmented", dataset_name="34")
    beatles_34 = utils.custom_dataset_loader(DATA_PATH,
            folder="beatles_augmented", dataset_name="34")

    aug34_data = gtzan_34.load_tracks() | rwcj_34.load_tracks() | rwcc_34.load_tracks() | beatles_34.load_tracks()

    # augmented full tracks
    aug_full_train, aug_full_val, aug_full_test = augmented_full_split(baseline_train, baseline_val, baseline_test, aug24_data, aug34_data)
    save_splits(SPLIT_PATH, "augmented_full", aug_full_train, aug_full_val, aug_full_test)

    # augmented sampled tracks
    aug_sampled_train, aug_sampled_val, aug_sampled_test = augmented_sampled_split(baseline_train, baseline_val, baseline_test, aug24_data, aug34_data)
    save_splits(SPLIT_PATH, "augmented_sampled", aug_sampled_train, aug_sampled_val, aug_sampled_test)

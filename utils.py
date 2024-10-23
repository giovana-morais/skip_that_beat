import os
from collections import Counter

import numpy as np
from mirdata import initialize

from dataset import Dataset

def get_split_tracks(split_file):
    """
    given a txt file with track names separated by \n,
    return a list with all the files
    """
    print(f"split file {split_file}")
    with open(split_file, "r") as f:
        tracks = f.read().splitlines()
    return tracks


def custom_dataset_loader(path, dataset_name, folder="datasets"):
    """
    loads a custom dataset
    """
    print(f"Loading {dataset_name} through custom loader")
    datasetdir = os.path.join(path, folder, dataset_name)
    dataset = Dataset(
        dataset_name = dataset_name,
        data_home=os.path.join(datasetdir, "audio"),
        annotations_home=os.path.join(datasetdir, "annotations")
    )
    return dataset


def multi_dataset_loader(data_home, dataset_names):
    """
    load and concatenate multiple datasets into a single one

    arguments
    ---
        datasets : list[str]
            list with datasets names (mirdata or custom)

    return
    ---
        tracks : dict{str: mirdata.Track}
           dictionary with mirdata.Track information

    """

    tracks = {}
    augs = ["_24", "_34", "_64"]


    if "beatles" in dataset_names:
        dataset = initialize("beatles", version="default",
                data_home=os.path.join(data_home, "beatles"))
        tracks = tracks | dataset.load_tracks()

    if "beatles_24" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "beatles_augmented",
                dataset_name = "24"
                )
        tracks = tracks | dataset.load_tracks()

    if "beatles_34" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "beatles_augmented",
                dataset_name = "34"
                )
        tracks = tracks | dataset.load_tracks()

    if "rwc_jazz" in dataset_names:
        dataset = initialize("rwc_jazz", version="default",
                data_home=os.path.join(data_home, "rwc_jazz"))
        tracks = tracks | dataset.load_tracks()

    if "rwc_jazz_24" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "rwc_jazz_24_augmented",
                dataset_name = "24"
                )
        tracks = tracks | dataset.load_tracks()

    if "rwc_jazz_34" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "rwc_jazz_34_augmented",
                dataset_name = "34"
                )
        tracks = tracks | dataset.load_tracks()

    if "rwc_classical" in dataset_names:
        dataset = initialize("rwc_classical", version="default",
                data_home=os.path.join(data_home, "rwc_classical"))
        tracks = tracks | dataset.load_tracks()

    if "rwc_classical_24" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "rwc_classical_34_augmented",
                dataset_name = "24"
                )
        tracks = tracks | dataset.load_tracks()

    if "rwc_classical_34" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "rwc_classical_34_augmented",
                dataset_name = "34"
                )
        tracks = tracks | dataset.load_tracks()


    if "rwc_pop" in dataset_names:
        dataset = initialize("rwc_pop", version="default",
                data_home=os.path.join(data_home, "rwc_pop"))
        tracks = tracks | dataset.load_tracks()

    if "gtzan" in dataset_names:
        dataset = initialize("gtzan_genre", version="default",
                data_home=os.path.join(data_home, "gtzan_genre"))

        tracks = tracks | dataset.load_tracks()

    if "gtzan_24" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "gtzan_genre_augmented",
                dataset_name = "24"
                )
        tracks = tracks | dataset.load_tracks()

    if "gtzan_34" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "gtzan_genre_augmented",
                dataset_name = "34"
                )
        tracks = tracks | dataset.load_tracks()

    if "gtzan_64" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "gtzan_genre_augmented",
                dataset_name = "64"
                )
        tracks = tracks | dataset.load_tracks()

    if "gtzan_74" in dataset_names:
        dataset = custom_dataset_loader(
                path = data_home,
                folder = "gtzan_genre_augmented",
                dataset_name = "74"
                )
        tracks = tracks | dataset.load_tracks()

    return tracks


def dataset_meter(dataset):
    """
    return a dictionary with the dataset tracks and their respective meter (time
    signature) based on beat annotations.

    this method will skip tracks with no beat information.
    this method also skip tracks with no beat **position** information, i.e. the
    number of the beat inside the bar.

    Parameters
    ---
    dataset: mirdata.Dataset
        an instance of a mirdata dataset or a custom dataset that implements the
        Dataset class

    Return
    ---
    dataset_meter: dict
        dictionary of type {track_id: meter}

    """
    dataset_meter = {}

    for t in dataset.track_ids:
        tid = dataset.track(t)

        try:
            beat_positions = tid.beats.positions
            c = Counter(beat_positions[np.where(np.diff(beat_positions) < 0)])
            dataset_meter[t] = int(c.most_common()[0][0])
        except (AttributeError, ValueError):
            print(f"track {t} has no beat information.skipping")
            continue

    return dataset_meter

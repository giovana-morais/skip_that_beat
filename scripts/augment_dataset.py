"""
example usage
---
augment gtzan, beatles and rwc_jazz tracks to 6/4:

    python augment_dataset.py \
            --data_home /home/Documents/datasets/ \
            --datasets gtzan_genre beatles rwc_jazz \
            --target_aug 64

augment rwc_classical to all available time signature:
    python augment_dataset.py \
            --data_home /home/Documents/datasets/ \
            --datasets rwc_classical
"""
import argparse
import os
import sys
from collections import Counter
sys.path.append("..")

import mirdata
import soundfile as sf
import tqdm

import meter_augmentation as me # noqa: E402
import utils

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
        try:
            tid_meter = dataset.track(t).meter
            if tid_meter in include:
                meters[t] = tid_meter
        except (AttributeError, ValueError, FileNotFoundError):
            print(f"track {t} has no meter information. skipping")
            continue
        except IndexError:
            print(f"track {t} has meter information. skipping")
            continue

    if print_stats:
        print(Counter(meters.values()))

    return meters


def augment(dataset, track_meter, target_augmentation, aug_dict):
    """
    augment tracks and write new audio file into specified folder defined inside
    the aug_dict parameter

    arguments
    ---
    dataset : mirdata.Dataset
        dataset that we're going to augment
    track_meter : dict
        dictionary with meter information of the track
    target_augmentation : str
        target augmentation value. 34 stands for 3/4. supported augmentations
        are ["24", "34", "64", "74"]
    aug_dict : dict
        dictionary with augmentation-related information
    """
    augmentation_fn = aug_dict[target_augmentation]["function"]
    audio_path = aug_dict[target_augmentation]["audio_path"]
    annotations_path = aug_dict[target_augmentation]["annotations_path"]
    beats_path = aug_dict[target_augmentation]["beats_path"]

    for track_id, meter in tqdm.tqdm(track_meter.items()):
        _, sr = dataset.track(track_id).audio
        y2, corrected_intervals, corrected_positions = augmentation_fn(dataset, track_id)

        with open(os.path.join(beats_path, f"{track_id}_{target_augmentation}.beats"), "w") as f:
            for i in zip(corrected_intervals[:,0], corrected_positions):
                f.write(f"{i[0]}\t{i[1]}\n")

        with open(os.path.join(meter_path, f"{track_id}_{target_augmentation}.meter"), "w") as f:
            f.write(f"{target_augmentation[0]}/{target_augmentation[1]}")

        sf.write(os.path.join(audio_path, f"{track_id}_{target_augmentation}.wav"), y2, sr)

    return

def create_parser():
    """
    creates ArgumentParser
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_home",
        type=str,
        required=True,
        help="path for datasets folder"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="list of datasets to augment"
    )
    parser.add_argument(
        "--target_aug",
        type=str,
        nargs="+",
        required=False,
        help="target augmentations. if no values are provided, augment to all possible target values"
    )
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.target_aug is None:
        args.target_aug = ["24", "34", "64", "74"]

    # data_home = "/media/gigibs/DD02EEEC68459F17/datasets"
    # target_augs = ["24", "34"]

    # for dataset_name in ["beatles", "gtzan", "rwcc", "rwcj"]:
    for dataset_name in args.datasets:
        print(f"Augmenting {dataset_name}")
        dataset = utils.custom_dataset_loader(data_home, dataset_name, "")

        track_meter = load_meter(dataset, include=["4/4"])

        aug_dict = {}
        for target_aug in target_augs:
            output_path = os.path.join(data_home, f"{dataset_name}_augmented")
            aug_path = os.path.join(output_path, target_aug)
            audio_path = os.path.join(aug_path, "audio")
            annotations_path = os.path.join(aug_path, "annotations")
            beats_path = os.path.join(annotations_path, "beats")
            meter_path = os.path.join(annotations_path, "meter")
            # tempo_path = os.path.join(annotations_path, "tempo")

            aug_dict[target_aug] = {}
            aug_dict[target_aug]["aug_path"] = aug_path
            aug_dict[target_aug]["audio_path"] = audio_path
            aug_dict[target_aug]["annotations_path"] = annotations_path
            aug_dict[target_aug]["beats_path"] = beats_path
            aug_dict[target_aug]["meter_path"] = beats_path
            aug_dict[target_aug]["function"] = getattr(me, f"to_{target_aug}")

            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            if not os.path.isdir(aug_path):
                os.mkdir(aug_path)
                os.mkdir(audio_path)
                os.mkdir(annotations_path)
                os.mkdir(beats_path)
                os.mkdir(meter_path)
                # os.mkdir(tempo_path)

                print(f"target augmentation = {target_aug[0]}/{target_aug[1]}")
                print(f"\toutput path {output_path}")
                print(f"\taudio path {audio_path}")
                print(f"\tannotations path {annotations_path}")
                augment(dataset, track_meter, target_aug, aug_dict)
            else:
                print(f"{aug_path} already exists.")

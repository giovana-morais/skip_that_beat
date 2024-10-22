"""
changes Beatles structure to follow audio annotations used by BayesBeat (and our
custom loader)

folder structure:
    * audio
    * annotations
        * meter (inferred from beat annotation)
        * beat (given)
"""

import argparse
import os
import shutil
from collections import Counter

import mirdata
import numpy as np
from tqdm import tqdm

def infer_meter(track):
    try:
        beat_positions = track.beats.positions
        c = Counter(beat_positions[np.where(np.diff(beat_positions) < 0)])
        denominator = int(c.most_common()[0][0])
        # assuming only simple meters for now
        meter = f"{denominator}/4"
    except (AttributeError, ValueError):
        print(f"track {track.track_id} has no beat information.skipping")
        meter = None

    return meter

def create_parser():
    """
    creates ArgumentParser
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_home",
        type=str,
        required=True,
        help="path for your dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="where to save the parsed data"
    )
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    audio_folder = os.path.join(args.output_path, "audio")
    annotations_folder = os.path.join(args.output_path, "annotations")
    beats_folder = os.path.join(annotations_folder, "beats")
    meter_folder = os.path.join(annotations_folder, "meter")

    beatles = mirdata.initialize("beatles", data_home=args.data_home)

    tracks = beatles.track_ids

    # create folders
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(beats_folder, exist_ok=True)
    os.makedirs(meter_folder, exist_ok=True)

    for tid in tqdm(tracks):
        # copy audio and rename to tid
        audio_src = beatles.track(tid).audio_path
        audio_dest = os.path.join(audio_folder, tid + ".wav")
        shutil.copy(audio_src, audio_dest)

        # copy beats
        # note: we can't copy the .beats file because there is a very annoying
        # "NewPoint" value that i do not want to deal with, so i'll just use
        # mirdata annotations
        beats_dest = os.path.join(beats_folder, tid + ".beats")

        try:
            beat_times = beatles.track(tid).beats.times
            beat_positions = beatles.track(tid).beats.positions
        except Exception:
            print(f"we don't have beat annotations for {tid}. skipping")
            continue

        with open(beats_dest, "w") as f:
            for bt, bp in zip(beat_times, beat_positions):
                f.write(f"{bt}\t{bp}\n")

        # infer meter and save it
        meter = infer_meter(beatles.track(tid))
        if meter is None:
            continue

        with open(os.path.join(meter_folder, tid + ".meter"), "w") as f:
            f.write(meter)

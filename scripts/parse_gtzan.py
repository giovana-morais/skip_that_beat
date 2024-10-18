"""
change GTZAN structure to follow audio annotations used by BayesBeat (and our
custom loader)

folder structure:
    * audio
    * annotations
        * meter (given by gtzan rhythm or inferred from beat annotation)
        * beat (given)

GTZAN-rhythm source: http://anasynth.ircam.fr/home/media/GTZAN-rhythm
"""

import os
import shutil
from collections import Counter

import mirdata
import numpy as np
import pandas as pd
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

if __name__ == "__main__":
    # define all paths
    datasets_home = "/media/gigibs/DD02EEEC68459F17/datasets"
    data_home = os.path.join(datasets_home, "gtzan_genre")
    rhythm_annotations_path = os.path.join(datasets_home, "gtzan_rhythm")
    output_folder = os.path.join(datasets_home, "gtzan")
    audio_folder = os.path.join(output_folder, "audio")
    annotations_folder = os.path.join(output_folder, "annotations")
    beats_folder = os.path.join(annotations_folder, "beats")
    meter_folder = os.path.join(annotations_folder, "meter")

    # initialize dataset
    gtzan = mirdata.initialize("gtzan_genre", data_home=data_home)
    tracks = gtzan.track_ids

    # create folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(beats_folder, exist_ok=True)
    os.makedirs(meter_folder, exist_ok=True)

    df = pd.read_csv(os.path.join(rhythm_annotations_path, "stats.csv"))
    df["filename"] = df["filename"].apply(lambda x: x.replace(".wav", ""))
    df.index = df["filename"]

    for tid in tqdm(tracks):
        # copy audio
        audio_src = gtzan.track(tid).audio_path
        audio_dest = os.path.join(audio_folder, tid + ".wav")
        shutil.copy(audio_src, audio_dest)

        # copy beats
        beats_src = gtzan.track(tid).beats_path
        beats_dest = os.path.join(beats_folder, tid + ".beats")
        if beats_src is None:
            print(f"{tid} does not have beat annotations")
            continue
        shutil.copy(beats_src, beats_dest)

        # if we have meter from GTZAN rhythm, copy it
        # otherwise infer
        if isinstance(df.loc[tid].meter, str):
            meter = df.loc[tid].meter
        else:
            meter = infer_meter(gtzan.track(tid))

        if meter is None:
            continue

        with open(os.path.join(meter_folder, tid + ".meter"), "w") as f:
            f.write(meter)

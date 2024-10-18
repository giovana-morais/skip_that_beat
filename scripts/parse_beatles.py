"""
changes Beatles structure to follow audio annotations used by BayesBeat (and our
custom loader)

folder structure:
    * audio
    * annotations
        * meter (inferred from beat annotation)
        * beat (given)
"""

import os
import shutil
from collections import Counter

import mirdata
import numpy as np
import tqdm

def infer_meter(track):
    try:
        beat_positions = track.beats.positions
        c = Counter(beat_positions[np.where(np.diff(beat_positions) < 0)])
        denominator = int(c.most_common()[0][0])
        # assuming only simple meters for now
        meter = f"{denominator}/4"
    except (AttributeError, ValueError):
        print(f"track {t} has no beat information.skipping")
        meter = None

    return meter

if __name__ == "__main__":
    datasets_home = "/media/gigibs/DD02EEEC68459F17/datasets"
    data_home = os.path.join(datasets_home, "beatles_orig")
    output_folder = os.path.join(datasets_home, "beatles")
    audio_folder = os.path.join(output_folder, "audio")
    annotations_folder = os.path.join(output_folder, "annotations")
    beats_folder = os.path.join(annotations_folder, "beats")
    meter_folder = os.path.join(annotations_folder, "meter")

    beatles = mirdata.initialize("beatles", data_home=data_home)

    tracks = beatles.track_ids

    # create folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(beats_folder, exist_ok=True)
    os.makedirs(meter_folder, exist_ok=True)

    for tid in tqdm.tqdm(tracks):
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
        except:
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

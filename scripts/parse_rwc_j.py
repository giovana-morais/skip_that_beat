"""
changes RWC jazz structure to follow audio annotations used by BayesBeat (and our
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
    data_home = os.path.join(datasets_home, "rwc_jazz")
    output_folder = os.path.join(datasets_home, "rwcj")
    audio_folder = os.path.join(output_folder, "audio")
    annotations_folder = os.path.join(output_folder, "annotations")
    beats_folder = os.path.join(annotations_folder, "beats")
    meter_folder = os.path.join(annotations_folder, "meter")

    rwcj = mirdata.initialize("rwc_jazz", data_home=data_home)

    tracks = rwcj.track_ids

    # create folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(beats_folder, exist_ok=True)
    os.makedirs(meter_folder, exist_ok=True)

    for tid in tracks:
        # copy audio and rename to tid
        audio_src = rwcj.track(tid).audio_path
        audio_dest = os.path.join(audio_folder, tid + ".wav")
        shutil.copy(audio_src, audio_dest)

        # copy beats
        # note: we can't copy the .beats file because they use a strange format
        # that has the note duration embedded. to avoid recalculating, we just
        # use mirdata's calculation and save it directly to our own .beats file
        beats_dest = os.path.join(beats_folder, tid + ".beats")
        beat_times = rwcj.track(tid).beats.times
        beat_positions = rwcj.track(tid).beats.positions

        with open(beats_dest, "w") as f:
            for bt, bp in zip(beat_times, beat_positions):
                f.write(f"{bt}\t{bp}\n")

        # infer meter and save it
        meter = infer_meter(rwcj.track(tid))
        if meter is None:
            continue

        with open(os.path.join(meter_folder, tid + ".meter"), "w") as f:
            f.write(meter)

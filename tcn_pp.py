"""
Runs models and does postprocessing with Peak-Picking
"""
import os
import sys

import madmom
import numpy as np
from librosa import frames_to_time
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.append("..")
import utils

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


if __name__ == "__main__":
    experiment = sys.argv[1]
    # load test tracks
    # test_tracks = utils.get_split_tracks("data/splits/baseline_test.txt")
    test_tracks = utils.get_split_tracks("data/splits/brid.txt")
    test_tracks_ids = [os.path.basename(t.replace(".wav", "")) for t in test_tracks]
    # datasets = ["gtzan", "beatles", "rwcc", "rwcj"]
    datasets = ["brid"]
    data_home = "/media/gigibs/DD02EEEC68459F17/datasets/"

    data = {}
    for d in datasets:
        dataset = utils.custom_dataset_loader(data_home, d, "")
        data = data | dataset.load_tracks()

    test_data = {k: v for k,v in data.items() if k in test_tracks_ids}

    # load ground truth annotations
    gt_beat_annotations = {
        k: v.beats.times for k,v in test_data.items() if v.beats is not None
    }

    gt_downbeat_annotations = {
        k: v.beats.times[v.beats.positions == 1] for k,v in test_data.items() if v.beats is not None
    }

    # load activations outputted by TCN
    activations_path = f"models/{experiment}/detections"
    output_path = f"models/{experiment}/detections_pp"
    os.makedirs(output_path, exist_ok=True)

    beat_activations = {
        k: np.load(os.path.join(activations_path, f"{k}.beats.npy")) for k
            in test_tracks_ids
    }

    downbeat_activations = {
        k: np.load(os.path.join(activations_path, f"{k}.downbeats.npy")) for k
            in test_tracks_ids
    }

    # peak pick the shit out of them
    for tid, beat_act in beat_activations.items():
        print(f"beat -- {tid}")
        detection = peak_picking_MSFA(beat_act)
        # convert back to seconds
        detection = detection/100.
        # write
        madmom.io.write_beats(detection, "%s/%s.beats.txt" % (output_path, tid))

    for tid, downbeat_act in downbeat_activations.items():
        print(f"downbeat -- {tid}")
        detection = peak_picking_MSFA(downbeat_act)
        # convert back to seconds
        detection = detection/100.
        madmom.io.write_beats(detection, "%s/%s.downbeats.txt" % (output_path, tid))

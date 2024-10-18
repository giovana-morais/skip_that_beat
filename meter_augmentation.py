"""
Meter augmentation functions
"""
import librosa
import numpy as np

def remix(y, sr, intervals):
    """
    remix track

    arguments
    ---
        y : np.array
            audio array
        sr : float
            sampling rate
        intervals : np.array
            array with beat intervals we want to keep
    """
    # we need to add the start interval otherwise librosa.remix
    # will not consider the first miliseconds
    start_interval = np.asarray([[0, intervals[0][0]]])
    remix_intervals = np.concatenate((start_interval, intervals))

    y2 = librosa.effects.remix(y, librosa.time_to_samples(remix_intervals, sr=sr))

    return y2


def get_beat_intervals(beats):
    """
    given a list of beats, create inter beat intervals
    """
    beat_intervals = []
    for idx, val in enumerate(beats.times):
        if idx == len(beats.times)-1:
            break
        beat_intervals.append([val, beats.times[idx+1]])

    return np.asarray(beat_intervals)


def correct_annotations(beats, good_intervals):
    """
    correct annotations for time displacements
    """
    good_durations = np.diff(good_intervals)
    corrected_intervals = []
    start = beats.times[0]
    for idx in range(len(good_intervals)):
        corrected_intervals.append([start, start+good_durations[idx][0]])
        start += good_durations[idx][0]

    corrected_intervals = np.asarray(corrected_intervals)

    return corrected_intervals


def correct_positions(positions, meter):
    """
    correct the beat values for an augmented track

    example
    ---
    augmented_positions = np.array([3,4,5,1,2,3,4,5])
    corrected = np.array([3,4,5,1,2,3,4,5])
    """
    corrected = np.arange(positions[0]-1, len(positions)+positions[0]-1) % meter + 1
    return corrected


def to_24(dataset, track_id, **kwargs):
    """
    augment 4/4 track to 2/4 by removing two beat bars
    """
    y, sr = dataset.track(track_id).audio
    beats = dataset.track(track_id).beats

    beat_intervals = get_beat_intervals(beats)
    beat_positions = beats.positions.astype(int)

    good_intervals = beat_intervals[beat_positions[:-1] < 3]
    # drop 3 and 4
    good_positions = beat_positions[beat_positions < 3]

    corrected_intervals = correct_annotations(beats, good_intervals)
    corrected_positions = correct_positions(good_positions, 2)

    y2 = remix(y, sr, good_intervals)

    return y2, corrected_intervals, corrected_positions


def to_34(dataset, track_id):
    """
    augment 4/4 track to 3/4 by removing one beat interval from each bar
    """

    beat_to_skip = 4

    if beat_to_skip is None:
        import random
        beat_to_skip = random.randint(2,4)
        print(f"skipping {beat_to_skip}")

    y, sr = dataset.track(track_id).audio
    beats = dataset.track(track_id).beats

    beat_intervals = get_beat_intervals(beats)
    beat_positions = beats.positions.astype(int)

    good_intervals = beat_intervals[beat_positions[:-1] != beat_to_skip]
    good_positions = beat_positions[beat_positions != beat_to_skip]

    corrected_intervals = correct_annotations(beats, good_intervals)
    corrected_positions = correct_positions(good_positions, 3)

    y2 = remix(y, sr, good_intervals)

    return y2, corrected_intervals, corrected_positions


def to_54(dataset, track_id, **kwargs):
    """
    augment 4/4 track to 5/4 by repeating one beat interval per bar
    """
    y, sr = dataset.track(track_id).audio
    beats = dataset.track(track_id).beats

    beat_intervals = get_beat_intervals(beats)
    beat_positions = beats.positions.astype(int)

    good_intervals = []
    good_positions = []

    for ival, val in zip(beat_intervals, beat_positions):
        good_intervals.append(ival)
        good_positions.append(val)

        if val == 3:
            good_intervals.append(ival)
            good_positions.append(5)

    corrected_intervals = correct_annotations(beats, good_intervals)
    corrected_positions = correct_positions(good_positions, 5)

    y2 = remix(y, sr, good_intervals)

    return y2, corrected_intervals, corrected_positions


def to_64(dataset, track_id, **kwargs):
    """
    augment 4/4 track to 6/4 by removing two beat intervals for every other bar
    """
    y, sr = dataset.track(track_id).audio
    beats = dataset.track(track_id).beats

    beat_intervals = get_beat_intervals(beats)
    beat_positions = beats.positions.astype(int)

    good_intervals = []
    good_positions = []
    keep = True

    for ival, val in zip(beat_intervals, beat_positions):
        # Always keep 3 and 4
        if val > 2:
            good_intervals.append(ival)
            good_positions.append(val)
        else:
            if keep:
                good_intervals.append(ival)
                good_positions.append(val)
                if val == 2:
                    keep = False
            else:
                if val == 2:
                    keep = True

    corrected_intervals = correct_annotations(beats, good_intervals)
    corrected_positions = correct_positions(good_positions, 6)

    y2 = remix(y, sr, good_intervals)

    return y2, corrected_intervals, corrected_positions


def to_74(dataset, track_id, **kwargs):
    """
    augment 4/4 track to 7/4 by removing one beat interval for every other bar
    """
    y, sr = dataset.track(track_id).audio
    beats = dataset.track(track_id).beats

    beat_intervals = get_beat_intervals(beats)
    beat_positions = beats.positions.astype(int)

    good_intervals = []
    good_positions = []
    keep = True

    for ival, val in zip(beat_intervals, beat_positions):
        if val == 1:
            if keep:
                good_intervals.append(ival)
                good_positions.append(val)
                keep = False
            else:
                keep = True
        else:
            good_intervals.append(ival)
            good_positions.append(val)

    corrected_intervals = correct_annotations(beats, good_intervals)
    corrected_positions = correct_positions(good_positions, 7)

    y2 = remix(y, sr, good_intervals)

    return y2, corrected_intervals, corrected_positions

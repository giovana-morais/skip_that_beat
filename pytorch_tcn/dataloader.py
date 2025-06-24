#!/usr/bin/env python
# coding: utf-8
import madmom
import numpy as np
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor,
)
from scipy.ndimage import maximum_filter1d
from torch.utils.data import Dataset, DataLoader

FPS = 100
NUM_BANDS = 12
FFT_SIZE = 2048
MASK_VALUE = -1


class BeatData(Dataset):
    def __init__(self, dataset, split_tracks, fps=100, widen=False):
        self.fps = fps
        self.keys = split_tracks
        self.tracks = self._get_tracks(dataset)
        self.pre_processor = PreProcessor(fps=self.fps)
        self.pad_frames = 2
        self.widen = widen

        return

    def _get_tracks(self, dataset):
        tracks = {}
        for k in self.keys:
            tracks[k] = dataset[k]

        return tracks

    def __getitem__(self, idx):
        data = {}
        tid = self.keys[idx]
        track = self.tracks[tid]

        audio, sr = track.audio

        # if it's stereo, we downmix it to mono
        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)

        s = madmom.audio.Signal(audio, sr)
        x = self.pre_processor(s)

        # pad features
        pad_start = np.repeat(x[:1], self.pad_frames, axis=0)
        pad_stop = np.repeat(x[-1:], self.pad_frames, axis=0)

        x_padded = np.concatenate((pad_start, x, pad_stop))

        try:
            beats_times = track.beats.times
            beats = madmom.utils.quantize_events(
                beats_times, fps=self.fps, length=len(x)
            )
            beats = beats.astype("float32")
        except Exception as e:
            print(f"{tid} has no beat information. masking\n")
            beats = np.ones(len(x), dtype="float32") * MASK_VALUE
            beats_times = np.zeros(len(x))

        if self.widen:
            # we skip masked values
            if not np.allclose(beats, -1):
                np.maximum(beats, maximum_filter1d(beats, size=3) * 0.5, out=beats)

        try:
            downbeats_positions = track.beats.positions.astype(int) == 1
            downbeats_times = track.beats.times[downbeats_positions]
            downbeats = madmom.utils.quantize_events(
                downbeats_times, fps=self.fps, length=len(x)
            )
            downbeats = downbeats.astype("float32")
        except Exception as e:
            print(f"{tid} has no downbeat information. masking\n")
            print(e)
            downbeats = np.ones(len(x), dtype="float32") * MASK_VALUE
            downbeats_times = np.zeros(len(x))

        data["tid"] = tid
        # FIXME: adding this because torch is bothered by our batchsize=1
        data["x"] = np.expand_dims(x_padded, axis=0)
        data["beats"] = beats
        data["beats_ann"] = beats_times
        data["downbeats"] = downbeats
        data["downbeats_ann"] = downbeats_times
        # data["tempo"] = tempo

        return data

    def __len__(self):
        return len(self.keys)


class PreProcessor(SequentialProcessor):
    def __init__(
        self, frame_size=FFT_SIZE, num_bands=NUM_BANDS, log=np.log, add=1e-6, fps=FPS
    ):
        # resample to a fixed sample rate in order to get always the same number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # split audio signal in overlapping frames
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        # compute STFT
        stft = ShortTimeFourierTransformProcessor()
        # filter the magnitudes
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        # scale them logarithmically
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)
        # instantiate a SequentialProcessor
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))
        # safe fps as attribute (needed for quantization of events)
        self.fps = fps


if __name__ == "__main__":
    # testing our dataloader
    import mirdata

    gtzan_mini = mirdata.initialize("gtzan_genre", version="mini")
    # gtzan_mini.download()
    dataset_tracks = gtzan_mini.load_tracks()
    dataset_keys = list(dataset_tracks.keys())
    train_keys = dataset_keys[:5]

    train_data = BeatData(dataset_tracks, train_keys)
    train_dataloader = DataLoader(train_data)

    for i in train_dataloader:
        x, beats, downbeats = i["x"], i["beats"], i["downbeats"]
        print("x.shape", x.shape)
        print("beats.shape", beats.shape)
        print("downbeats.shape", downbeats.shape)
        print("beats", beats)
        print("downbeats", downbeats)

        test = beats.detach().cpu()
        print(len(test[test > 0]))
        print(dataset_tracks[train_keys[0]].beats.times)
        break

"""
PyTorch Lightning model. This module defines the train, validation and test
steps.
"""

import lightning as L
import madmom
import mir_eval
import numpy as np
import torch
import torch.nn.functional as F

import losses


class PLTCN(L.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.loss = self._get_loss_fn(params["LOSS"])
        self.learning_rate = params["LEARNING_RATE"]
        self.test_beat_fmeasure = []
        self.test_downbeat_fmeasure = []

    def _get_loss_fn(self, loss):
        return losses.masked_binary_cross_entropy

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # get annotations
        x = batch["x"]
        beats_ann = batch["beats"]
        downbeats_ann = batch["downbeats"]

        # get predictions
        output = self(x)
        beats_det = output["beats"].squeeze(-1)
        downbeats_det = output["downbeats"].squeeze(-1)

        # compute losses and log them
        beat_loss = self.loss(beats_det, beats_ann)
        downbeat_loss = self.loss(downbeats_det, downbeats_ann)
        loss = beat_loss + downbeat_loss

        # log them
        self.log("train_beat_loss", beat_loss, prog_bar=True, on_epoch=True)
        self.log("train_downbeat_loss", downbeat_loss, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # get annotations
        x = batch["x"]
        beats_ann = batch["beats"]
        downbeats_ann = batch["downbeats"]

        # get predictions
        output = self(x)
        beats_det = output["beats"].squeeze(-1)
        downbeats_det = output["downbeats"].squeeze(-1)

        # compute losses
        beat_loss = self.loss(beats_det, beats_ann)
        downbeat_loss = self.loss(downbeats_det, downbeats_ann)
        loss = beat_loss + downbeat_loss

        # log them
        self.log("val_beat_loss", beat_loss, prog_bar=True, on_epoch=True)
        self.log("val_downbeat_loss", downbeat_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        # get annotations
        x = batch["x"]
        beats_target = batch["beats_ann"].detach().cpu().numpy().squeeze()
        downbeats_target = batch["downbeats_ann"].detach().cpu().numpy().squeeze()

        # get activations
        output = self(x)
        beats_act = output["beats"].squeeze().detach().cpu().numpy()
        downbeats_act = output["downbeats"].squeeze().detach().cpu().numpy()

        # define beat and downbeat DBN
        beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55.0, max_bpm=215.0, fps=100, transition_lambda=100, online=False
        )
        downbeat_dbn = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=[2, 3, 4],
            min_bpm=55.0,
            max_bpm=215.0,
            fps=100,
            transition_lambda=100,
        )

        beats_prediction = beat_dbn(beats_act)
        # following TF implementation, downbeat DBN receives the combined
        # beat/downbeat activations
        combined_act = np.vstack(
            (np.maximum(beats_act - downbeats_act, 0), downbeats_act)
        ).T
        downbeats_prediction = downbeat_dbn(combined_act)
        # the combined activation results in 2d predictions, [beat_time,
        # beat_position]. therefore we need to filter only the downbeats
        # timestamps for the fmeasure calculation.
        downbeats_timestamps = downbeats_prediction[downbeats_prediction[:, 1] == 1][
            :, 0
        ]

        # calculate f-measure
        beat_fmeasure = mir_eval.beat.f_measure(beats_target, beats_prediction)
        downbeat_fmeasure = mir_eval.beat.f_measure(
            downbeats_target, downbeats_timestamps
        )

        # log it
        self.test_beat_fmeasure.append(beat_fmeasure)
        self.test_downbeat_fmeasure.append(downbeat_fmeasure)

        self.log("beat_fmeasure", beat_fmeasure, on_step=True)
        self.log("downbeat_fmeasure", downbeat_fmeasure, on_step=True)

        return [beat_fmeasure, downbeat_fmeasure]

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.005)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=10,
                threshold=1e-3,
                cooldown=0,
                min_lr=1e-7,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

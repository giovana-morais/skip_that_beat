import os
import sys
from datetime import datetime

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from config import PARAMS_TRAIN
from dataloader import BeatData
from model import MultiTracker
from pl_model import PLTCN

sys.path.append("..")
import utils

def get_tracks(experiment):
    full_train_files = utils.get_split_tracks(f"../splits/{experiment}_train.txt")
    full_validation_files = utils.get_split_tracks(f"../splits/{experiment}_val.txt")
    full_test_files = utils.get_split_tracks(f"../splits/{experiment}_test.txt")
    full_brid_files = utils.get_split_tracks(f"../splits/brid.txt")

    train_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_train_files
    ]
    validation_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_validation_files
    ]
    test_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_test_files
    ]
    brid_tracks = [
        os.path.splitext(os.path.basename(i))[0] for i in full_brid_files
    ]

    print(f"loaded train tracks: {len(train_tracks)}")
    print(f"loaded validation tracks: {len(validation_tracks)}")
    print(f"loaded test tracks: {len(test_tracks)}")
    print(f"loaded brid tracks: {len(brid_tracks)}")

    return train_tracks, validation_tracks, test_tracks, brid_tracks


if __name__ == "__main__":
    # load params
    PARAMS = PARAMS_TRAIN

    # TODO: RECEIVE THIS AS PARAMETER
    experiment = "baseline"
    data_home = "/media/gigibs/DD02EEEC68459F17/datasets/"
    datasets = [
        "gtzan", "gtzan_augmented/24", "gtzan_augmented/34",
        "beatles", "beatles_augmented/24", "beatles_augmented/34",
        "rwcc", "rwcc_augmented/24", "rwcc_augmented/34",
        "rwcj", "rwcj_augmented/24", "rwcj_augmented/34",
        "brid"
    ]

    dataset_tracks = {}
    for d in datasets:
        d = utils.custom_dataset_loader(
                path = data_home,
                folder = "",
                dataset_name = d
            )
        dataset_tracks = dataset_tracks | d.load_tracks()

    train_keys, validation_keys, test_keys, brid_keys = get_tracks(experiment)

    # create dataloaders
    train_data = BeatData(dataset_tracks, train_keys, widen=True)
    val_data = BeatData(dataset_tracks, validation_keys, widen=True)
    test_data = BeatData(dataset_tracks, test_keys, widen=True)
    brid_data = BeatData(dataset_tracks, brid_keys, widen=True)

    train_dataloader = DataLoader(
        train_data, batch_size=1, num_workers=PARAMS["NUM_WORKERS"]
    )
    val_dataloader = DataLoader(
        val_data, batch_size=1, num_workers=PARAMS["NUM_WORKERS"]
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1, num_workers=PARAMS["NUM_WORKERS"]
    )
    brid_dataloader = DataLoader(
        brid_data, batch_size=1, num_workers=PARAMS["NUM_WORKERS"]
    )

    # instatiate models
    tcn = MultiTracker(
        n_filters=PARAMS["N_FILTERS"],
        n_dilations=PARAMS["N_DILATIONS"],
        kernel_size=PARAMS["KERNEL_SIZE"],
        dropout_rate=PARAMS["DROPOUT"],
    )
    model = PLTCN(tcn, PARAMS)

    # define where to save the checkpoint
    # and create checkpoint file with the current timestamp
    CKPTS_DIR = "trained_models"
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    ckpt_name = f"tcn_{timestamp}"

    # log into wandb
    run = wandb.init(
        project="skip_the_beat", name=f"TCN_train_{timestamp}", config=PARAMS
    )
    logger = WandbLogger()
    logger.watch(model, "all")

    trainer = L.Trainer(
        max_epochs=PARAMS["N_EPOCHS"],
        logger=logger,
        gradient_clip_val=PARAMS["GRADIENT_CLIP"],
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(
                dirpath=CKPTS_DIR,
                filename=ckpt_name,
                monitor="val_loss",
                auto_insert_metric_name=True,
                save_top_k=1,  # save top two best models for this criteron
            )
        ],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(
        model=model,
        dataloaders=test_dataloader,
        ckpt_path=f"{CKPTS_DIR}/{ckpt_name}.ckpt",
        verbose=True,
    )

    wandb.finish()

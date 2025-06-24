from datetime import datetime

import lightning as L
import mirdata
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import PARAMS_TRAIN
from dataloader import BeatData
from model import MultiTracker
from pl_model import PLTCN


if __name__ == "__main__":
    # load params
    PARAMS = PARAMS_TRAIN

    # load dataset
    gtzan_mini = mirdata.initialize("gtzan_genre", version="mini")
    gtzan_mini.download(["index"])

    dataset_tracks = gtzan_mini.load_tracks()
    dataset_keys = list(dataset_tracks.keys())

    # split data into train/val/test
    train_keys, test_keys = train_test_split(
        dataset_keys, test_size=0.2, random_state=42
    )
    train_keys, val_keys = train_test_split(train_keys, test_size=0.25, random_state=42)

    # create dataloaders
    train_data = BeatData(dataset_tracks, train_keys, widen=True)
    val_data = BeatData(dataset_tracks, val_keys, widen=True)
    test_data = BeatData(dataset_tracks, test_keys, widen=True)

    train_dataloader = DataLoader(
        train_data, batch_size=1, num_workers=PARAMS["NUM_WORKERS"]
    )
    val_dataloader = DataLoader(
        val_data, batch_size=1, num_workers=PARAMS["NUM_WORKERS"]
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1, num_workers=PARAMS["NUM_WORKERS"]
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
        project="LAMIR_beat_tutorial", name=f"TCN_train_{timestamp}", config=PARAMS
    )
    logger = WandbLogger()
    logger.watch(model, "all")

    trainer = L.Trainer(
        max_epochs=PARAMS["N_EPOCHS"],
        logger=logger,
        gradient_clip_val=PARAMS["GRADIENT_CLIP"],
        callbacks=[
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

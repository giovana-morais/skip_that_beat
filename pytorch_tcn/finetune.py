import argparse
import re
from datetime import datetime

import lightning as L
import mirdata
import wandb
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import PARAMS_FINETUNE
from dataloader import BeatData
from model import MultiTracker
from pl_model import PLTCN


def _load_candombe(download=False):
    candombe = mirdata.initialize("candombe", version="default")

    if download:
        candombe.download(force_overwrite=True)

    dataset_tracks = candombe.load_tracks()
    dataset_keys = list(dataset_tracks.keys())

    return dataset_tracks, dataset_keys


def _load_brid(download=False):
    brid = mirdata.initialize("brid", version="default")

    if download:
        brid.download(force_overwrite=True)

    dataset_tracks = brid.load_tracks()
    dataset_keys = list(dataset_tracks.keys())
    # we do this to avoid loading tracks without annotations
    # it is not necessary in the case of candombe
    pattern = r"^\[\d{4}\] M\d+-\d+-[A-Z]+$"
    dataset_keys = [key for key in dataset_keys if bool(re.match(pattern, key))]

    return dataset_tracks, dataset_keys


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset name. supported options are ['brid', 'candombe']",
    )
    parser.add_argument(
        "--download", action="store_true", required=False, help="download the dataset"
    )
    return parser


if __name__ == "__main__":
    # load params
    PARAMS = PARAMS_FINETUNE
    args = create_parser().parse_args()

    if args.dataset == "candombe":
        dataset_tracks, dataset_keys = _load_candombe(args.download)
    elif args.dataset == "brid":
        dataset_tracks, dataset_keys = _load_brid(args.download)
    else:
        raise ValueError(
            "Dataset not supported. Supported options are 'brid' and 'candombe'"
        )

    # split keys into train/val/test
    train_keys, test_keys = train_test_split(
        dataset_keys, test_size=0.2, random_state=42
    )
    train_keys, val_keys = train_test_split(train_keys, test_size=0.25, random_state=42)

    for num_files in range(1, PARAMS["MAX_NUM_FILES"] + 1):
        # create dataloaders with different amount of files and evaluate them
        print(f"TRAIN WITH {num_files} FILES")
        train_data = BeatData(dataset_tracks, train_keys[:num_files], widen=True)
        val_data = BeatData(dataset_tracks, val_keys[:num_files], widen=True)
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

        # instantiate model
        tcn = MultiTracker(
            n_filters=PARAMS["N_FILTERS"],
            n_dilations=PARAMS["N_DILATIONS"],
            kernel_size=PARAMS["KERNEL_SIZE"],
            dropout_rate=PARAMS["DROPOUT"],
        )

        # load model checkpoint
        model = PLTCN.load_from_checkpoint(
            f"trained_models/{PARAMS['CHECKPOINT_FILE']}.ckpt", model=tcn, params=PARAMS
        )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run = wandb.init(
            project="LAMIR_beat_tutorial",
            name=f"TCN_brid_finetuning_{num_files}_{timestamp}",
            config=PARAMS,
        )
        logger = WandbLogger()

        trainer = L.Trainer(logger=logger, max_epochs=PARAMS["N_EPOCHS"])

        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader, verbose=True)

        wandb.finish()

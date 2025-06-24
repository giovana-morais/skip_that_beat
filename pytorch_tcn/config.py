PARAMS_TRAIN = {
    "LEARNING_RATE": 0.005,
    "N_FILTERS": 20,
    "KERNEL_SIZE": 5,
    "DROPOUT": 0.15,
    "N_DILATIONS": 11,
    "N_EPOCHS": 2,
    "GRADIENT_CLIP": 0.5,
    "LOSS": "BCE",
    "NUM_WORKERS": 7,
}

PARAMS_FINETUNE = {
    "MAX_NUM_FILES": 5,
    "LEARNING_RATE": 0.005,
    "N_FILTERS": 20,
    "KERNEL_SIZE": 5,
    "DROPOUT": 0.15,
    "N_DILATIONS": 11,
    "N_EPOCHS": 5,
    "LOSS": "BCE",
    "NUM_WORKERS": 7,
    # Change here with your checkpoint filename
    "CHECKPOINT_FILE": "tcn_202412052128",
}

def load_data(data_home, datasets):
    tracks = {}
    for d in datasets:
        d = utils.custom_dataset_loader(
                path = args.data_home,
                folder = "",
                dataset_name = d
            )
        tracks = tracks | d.load_tracks()
        print(len(tracks))

    full_train_files = utils.get_split_tracks(f"../splits/{args.experiment}_train.txt")
    full_validation_files = utils.get_split_tracks(f"../splits/{args.experiment}_val.txt")
    full_test_files = utils.get_split_tracks(f"../splits/{args.experiment}_test.txt")
    full_brid_files = utils.get_split_tracks(f"../splits/brid.txt")

    train_files = [
        os.path.splitext(os.path.basename(i))[0] for i in full_train_files
    ][10]
    validation_files = [
        os.path.splitext(os.path.basename(i))[0] for i in full_validation_files
    ][10]
    test_files = [
        os.path.splitext(os.path.basename(i))[0] for i in full_test_files
    ][10]
    brid_files = [
        os.path.splitext(os.path.basename(i))[0] for i in full_brid_files
    ][10]

    print(f"train: {len(train_files)}")
    print(f"validation: {len(validation_files)}")
    print(f"test: {len(test_files)}")
    print(f"brid: {len(brid_files)}")

    # # Train network

    # ## Training & testing sequences
    #
    # We wrap our previously split dataset as `DataSequences`.
    #
    # We widen the beat and downbeat targets to have a value of 0.5 at the frames next to the annotated beat locations.
    #
    # We assign tempo values ±1 bpm apart a value of 0.5, and those ±2bpm a value of0.25.

    pad_frames = 2
    pre_processor = PreProcessor()

    train = DataSequence(
        tracks={k: v for k, v in tracks.items() if k in train_files},
        pre_processor=pre_processor,
        pad_frames=pad_frames
    )
    train.widen_beat_targets()
    train.widen_downbeat_targets()
    train.widen_tempo_targets()
    train.widen_tempo_targets()

    if args.data_augmentation:
        for fps in [95, 97.5, 102.5, 105]:
            ds = DataSequence(
                tracks={f"{k}_{fps}": v for k, v in tracks.items() if k in train_files},
                pre_processor=PreProcessor(fps=fps),
                pad_frames=pad_frames,
            )
            ds.widen_beat_targets()
            ds.widen_downbeat_targets()
            ds.widen_tempo_targets(3, 0.5)
            ds.widen_tempo_targets(3, 0.5)
            train.append(ds)

    validation = DataSequence(
        tracks={k: v for k, v in tracks.items() if k in validation_files},
        pre_processor=pre_processor,
        pad_frames=pad_frames
    )
    validation.widen_beat_targets()
    validation.widen_downbeat_targets()
    validation.widen_tempo_targets()
    validation.widen_tempo_targets()

    return train, validation, test

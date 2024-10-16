# Skip That Beat: Augmenting Meter Tracking Models for Underrepresented Time Signatures 

## Datasets
Supported datasets: 
* Beatles
* BRID
* GTZAN
* GTZAN Rhythm
* RWC Classical
* RWC Jazz

### Dataset parsing
We use a custom `Dataset` class based on `mirdata`'s, so we are able to load original and augmented data. This class requires the data to be structured in `audio` and `annotation` folders. We provide scripts to parse the datasets in their original structure to the structure compatible with our loader. 

### GTZAN
For example, to run the GTZAN parsing script you should do
```bash
python parse_gtzan.py --dataset_path=gtzan_path --gtzan_rhythm_path=gtzan_rhythm_path --output_path=new_gtzan_path 
```

This will create a new folder in `new_gtzan_path` with the correct structure. 

## Augmentation
![augmentation](https://github.com/user-attachments/assets/40be038a-faba-47b0-88f1-d1530571b998)

Once the datasets are in the correct format, we can also augment them. To do so, run the `augment_dataset.py` script.

To augment all supported datasets to all meters:
```bash
python augment_dataset.py \
	--data_home /path/to/datasets
```

To augment specific datasets to specific meters:

```bash
python augment_dataset.py \
    --data_home /path/to/datasets \
    --datasets gtzan beatles \
    --target_aug 24
```

## Create splits
![dataset_distribution](https://github.com/user-attachments/assets/800064b9-6d68-475a-971f-abc318e37e52)

To create the splits shown in the figure above, just run 

```bash
python create_splits.py \
	--data_home /path/to/datasets \
	--splits_home /path/to/output/splits
```

This will save your splits in the `splits_home` folder.

## Run BayesBeat experiments
1. Make sure you have MATLAB installed
2. Clone the [bayesbeat GitHub repo](https://github.com/flokadillo/bayesbeat/tree/master)
3. Inside `bayesbeat_experiments/bayes_beat_training.m` file, replace the `base_path` variable (L46) to the location where you saved the repo in step 2
4. Run it!

The BayesBeat inference is the same process, but you change the `bayesbeat` path on L41 of the `inference.m` file and run it.

## Run TCN experiments
To train the TCN, run the `tcn.py` script providing the path to your dataset folder and the experiment you wish to reproduce (baseline, augmented_sampled, augmented_full)

```bash
python tcn.py \
	--data_home /path/to/datasets \
	--splits_home /path/to/splits \
	--experiment exp \
```

To run inference with both the DBN and PP methods, run 

```bash
python tcn_predict.py \
	--data_home /path/to/datasets \
	--test_set brid \
	--experiment exp
```

## Reproduce paper figures
To reproduce paper figures, please run the script `paper_figures.py`

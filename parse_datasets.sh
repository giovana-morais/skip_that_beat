#!/bin/bash

# CHANGE YOUR PATHS HERE
# path to where all your datasets are stored
DATASETS=/media/gigibs/DD02EEEC68459F17/datasets

GTZAN_PATH=$DATASETS/gtzan_genre
GTZAN_RHYTHM_PATH=$DATASETS/gtzan_rhythm
GTZAN_OUTPUT_PATH=$DATASETS/TMP/gtzan

BEATLES_PATH=$DATASETS/beatles_orig
BEATLES_OUTPUT_PATH=$DATASETS/TMP/beatles_new

RWCC_PATH=$DATASETS/rwc_classical
RWCC_OUTPUT_PATH=$DATASETS/TMP/rwcc

RWCJ_PATH=$DATASETS/rwc_jazz
RWCJ_OUTPUT_PATH=$DATASETS/TMP/rwcj
# ======================

echo "=====> Parsing GTZAN"
python parse_gtzan.py \
	--gtzan_path=$GTZAN_PATH \
	--gtzan_rhythm_path=$GTZAN_RHYTHM_PATH \
	--output_path=$GTZAN_OUTPUT_PATH
echo "Parsed dataset in $GTZAN_OUTPUT_PATH"

echo "=====> Parsing Beatles"
python parse_beatles.py \
	--data_home=$BEATLES_PATH \
	--output_path=$BEATLES_OUTPUT_PATH
echo "Parsed dataset in $BEATLES_OUTPUT_PATH"

echo "=====> Parsing RWC Classical"
python parse_rwc_c.py \
	--data_home=$RWCC_PATH \
	--output_path=$RWCC_OUTPUT_PATH
echo "Parsed dataset in $RWCC_OUTPUT_PATH"

echo "=====> Parsing RWC Jazz"
python parse_rwc_j.py \
	--data_home=$RWCJ_PATH \
	--output_path=$RWCJ_OUTPUT_PATH
echo "Parsed dataset in $RWCJ_OUTPUT_PATH"

#!/bin/sh
python train.py --experiment=baseline --model=default
python train.py --experiment=baseline --model=opnorm

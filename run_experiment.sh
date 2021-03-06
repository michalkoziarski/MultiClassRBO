#!/bin/bash

datasets=(
    'automobile'
    'balance'
    'car'
    'cleveland'
    'contraceptive'
    'dermatology'
    'ecoli'
    'flare'
    'glass'
    'hayes-roth'
    'led7digit'
    'lymphography'
    'newthyroid'
    'pageblocks'
    'thyroid'
    'vehicle'
    'wine'
    'winequality-red'
    'yeast'
    'zoo'
)

for dataset in "${datasets[@]}"; do
    for partition in {1..5}; do
        for fold in {1..2}; do
            sbatch slurm.sh trial.py -dataset $dataset -partition $partition -fold $fold -mode OVA -method sampling -results_path results_OVA_sampling
            sbatch slurm.sh trial.py -dataset $dataset -partition $partition -fold $fold -mode OVA -method complete -results_path results_OVA_complete
            sbatch slurm.sh trial.py -dataset $dataset -partition $partition -fold $fold -mode OVO -results_path results_OVO
        done
    done
done

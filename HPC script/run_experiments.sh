#!/bin/bash
#SBATCH --time=96:00:00              # Wall time (HH:MM:SS)
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=16G                     # Memory per node

module load StdEnv/2020
module load python/3.9.6
module load scipy-stack/2023a              
module load openmpi/4.0.3         
module load mpi4py/3.1.3 

#path to the directory where the Python script is located
SCRIPT_DIR=".../" 

datasets=("ACSEmployment")
models=("xgb" "rf")
ages=("g0" "g1")
seeds=("0 42 100 400 1000 5000 1234567")

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for age in "${ages[@]}"
        do
            srun python "${SCRIPT_DIR}/run.py" --dataset "$dataset" --model_name "$model" --run_stab --age "$age" --reductionist_type EO --seeds ${seeds[@]}
        done
    done
done
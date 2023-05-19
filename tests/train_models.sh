export LD_LIBRARY_PATH=${HOME}/miniconda3/envs/forecast/lib:${LD_LIBRARY_PATH}

python tests/train_models.py --input_folder data/Abilene-OD_pair.csv --model=tft --multivariate
# conda activate forecast
CUDA_VISIBLE_DEVICES=6 python tests/deepar_forecast.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_deepar_ckpts &> abilene_deepar_ckpts.log &
CUDA_VISIBLE_DEVICES=7 python tests/deepar_forecast.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_deepar_ckpts &> geant_deepar_ckpts.log &


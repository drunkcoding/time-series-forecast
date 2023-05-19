# conda activate forecast
python tests/vanilla_forecast.py --folder directed-abilene-zhang-5min-over-6months-ALL --layer gru --checkpoint tests/abilene_gru_ckpts &> logs/abilene_gru_ckpts.log
python tests/vanilla_forecast.py --folder directed-geant-uhlig-15min-over-4months-ALL --layer gru --checkpoint tests/geant_gru_ckpts &> logs/geant_gru_ckpts.log

python tests/vanilla_forecast.py --folder directed-abilene-zhang-5min-over-6months-ALL --layer lstm --checkpoint tests/abilene_lstm_ckpts &> logs/abilene_lstm_ckpts.log
python tests/vanilla_forecast.py --folder directed-geant-uhlig-15min-over-4months-ALL --layer lstm --checkpoint tests/geant_lstm_ckpts &> logs/geant_lstm_ckpts.log

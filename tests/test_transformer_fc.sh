# conda activate forecast
python tests/transformer_forecast.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_transformer_ckpts &> logs/abilene_transformer_ckpts.log
python tests/transformer_forecast.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_transformer_ckpts &> logs/geant_transformer_ckpts.log


# # transformer
# taskset --cpu-list 10-50 python tests/pyfc.py --layer transformer --folder directed-abilene-zhang-5min-over-6months-ALL &> logs/abilene_transformer_ckpts.log
# taskset --cpu-list 10-50 python tests/pyfc.py --layer transformer --folder directed-geant-uhlig-15min-over-4months-ALL &> logs/geant_transformer_ckpts.log

# vanilla
taskset --cpu-list 10-50 python tests/pyfc.py --layer gru --folder directed-abilene-zhang-5min-over-6months-ALL &> logs/abilene_gru_ckpts.log
taskset --cpu-list 10-50 python tests/pyfc.py --layer gru --folder directed-geant-uhlig-15min-over-4months-ALL &> logs/geant_gru_ckpts.log
taskset --cpu-list 10-50 python tests/pyfc.py --layer lstm --folder directed-abilene-zhang-5min-over-6months-ALL &> logs/abilene_lstm_ckpts.log
taskset --cpu-list 10-50 python tests/pyfc.py --layer lstm --folder directed-geant-uhlig-15min-over-4months-ALL &> logs/geant_lstm_ckpts.log

# deepar
taskset --cpu-list 10-50 python tests/pyfc.py --layer deepar --folder directed-abilene-zhang-5min-over-6months-ALL &> logs/abilene_deepar_ckpts.log
taskset --cpu-list 10-50 python tests/pyfc.py --layer deepar --folder directed-geant-uhlig-15min-over-4months-ALL &> logs/geant_deepar_ckpts.log


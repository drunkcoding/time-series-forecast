# Fill Methods S
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_gru_zfill_M_ckpts --fill zero --layer gru
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_gru_bfill_M_ckpts --fill bfill --layer gru
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_gru_ffill_M_ckpts --fill ffill --layer gru

# Fill Methods S
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_gru_zfill_M_ckpts --fill zero --layer gru
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_gru_bfill_M_ckpts --fill bfill --layer gru
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_gru_ffill_M_ckpts --fill ffill --layer gru


# Fill Methods S
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_lstm_zfill_M_ckpts --fill zero --layer lstm
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_lstm_bfill_M_ckpts --fill bfill --layer lstm
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_lstm_ffill_M_ckpts --fill ffill --layer lstm

# Fill Methods S
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_lstm_zfill_M_ckpts --fill zero --layer lstm
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_lstm_bfill_M_ckpts --fill bfill --layer lstm
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_lstm_ffill_M_ckpts --fill ffill --layer lstm
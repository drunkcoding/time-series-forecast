# Fill Methods S
python tests/test_prophet.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_prophet_zfill_S_ckpts --fill zero
python tests/test_prophet.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_prophet_bfill_S_ckpts --fill bfill
python tests/test_prophet.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_prophet_ffill_S_ckpts --fill ffill

# Fill Methods S
python tests/test_prophet.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_prophet_zfill_S_ckpts --fill zero
python tests/test_prophet.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_prophet_bfill_S_ckpts --fill bfill
python tests/test_prophet.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_prophet_ffill_S_ckpts --fill ffill
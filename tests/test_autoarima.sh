
# Fill Methods MS
python tests/test_autoarima.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoarima_zfill_MS_ckpts --features MS --fill zero
python tests/test_autoarima.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoarima_bfill_MS_ckpts --features MS --fill bfill
python tests/test_autoarima.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoarima_ffill_MS_ckpts --features MS --fill ffill

# Fill Methods S
python tests/test_autoarima.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoarima_zfill_S_ckpts --features S --fill zero
python tests/test_autoarima.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoarima_bfill_S_ckpts --features S --fill bfill
python tests/test_autoarima.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoarima_ffill_S_ckpts --features S --fill ffill


# Fill Methods MS
python tests/test_autoarima.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoarima_zfill_MS_ckpts --features MS --fill zero
python tests/test_autoarima.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoarima_bfill_MS_ckpts --features MS --fill bfill
python tests/test_autoarima.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoarima_ffill_MS_ckpts --features MS --fill ffill

# Fill Methods S
python tests/test_autoarima.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoarima_zfill_S_ckpts --features S --fill zero
python tests/test_autoarima.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoarima_bfill_S_ckpts --features S --fill bfill
python tests/test_autoarima.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoarima_ffill_S_ckpts --features S --fill ffill
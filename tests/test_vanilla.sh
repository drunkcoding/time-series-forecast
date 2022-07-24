# Fill Methods S
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_vanilla_zfill_S_ckpts --fill zero --layer gru
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_vanilla_bfill_S_ckpts --fill bfill --layer gru
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_vanilla_ffill_S_ckpts --fill ffill --layer gru

# Fill Methods S
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_vanilla_zfill_S_ckpts --fill zero --layer gru
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_vanilla_bfill_S_ckpts --fill bfill --layer gru
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_vanilla_ffill_S_ckpts --fill ffill --layer gru



# Fill Methods S
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_vanilla_zfill_S_ckpts --fill zero --layer lstm
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_vanilla_bfill_S_ckpts --fill bfill --layer lstm
python tests/test_vanilla.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_vanilla_ffill_S_ckpts --fill ffill --layer lstm

# Fill Methods S
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_vanilla_zfill_S_ckpts --fill zero --layer lstm
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_vanilla_bfill_S_ckpts --fill bfill --layer lstm
python tests/test_vanilla.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_vanilla_ffill_S_ckpts --fill ffill --layer lstm
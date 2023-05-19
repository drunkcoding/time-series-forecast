CHANNEL=132
# Fill Methods M
python im tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_zfill_M_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features M --fill zero
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_bfill_M_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features M --fill bfill
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_ffill_M_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features M --fill ffill

# Fill Methods MS
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_zfill_MS_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features MS --fill zero
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_bfill_MS_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features MS --fill bfill
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_ffill_MS_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features MS --fill ffill

# Fill Methods S
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_zfill_S_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features S --fill zero
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_bfill_S_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features S --fill bfill
python tests/test_autoformer.py --folder directed-abilene-zhang-5min-over-6months-ALL --checkpoint tests/abilene_autoformer_ffill_S_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features S --fill ffill

CHANNEL=462
# Fill Methods MS
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_zfill_M_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features M --fill zero
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_bfill_M_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features M --fill bfill
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_ffill_M_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features M --fill ffill

# Fill Methods MS
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_zfill_MS_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features MS --fill zero
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_bfill_MS_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features MS --fill bfill
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_ffill_MS_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features MS --fill ffill

# Fill Methods S
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_zfill_S_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features S --fill zero
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_bfill_S_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features S --fill bfill
python tests/test_autoformer.py --folder directed-geant-uhlig-15min-over-4months-ALL --checkpoint tests/geant_autoformer_ffill_S_ckpts --enc_in ${CHANNEL} --dec_in ${CHANNEL} --c_out ${CHANNEL} --pred_len 1 --features S --fill ffill
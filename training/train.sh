# CUDA_VISIBLE_DEVICES=4 python training/train_arima.py --data data/Abilene-OD_pair.csv --clip &> logs/arima_abilene_clip[true].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_arima.py --data data/GEANT-OD_pair.csv --clip &> logs/arima_geant_clip[true].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_arima.py --data data/CERNET-OD_pair.csv --clip &> logs/arima_cernet_clip[true].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_arima.py --data data/Abilene-OD_pair.csv &> logs/arima_abilene_clip[false].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_arima.py --data data/GEANT-OD_pair.csv &> logs/arima_geant_clip[false].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_arima.py --data data/CERNET-OD_pair.csv &> logs/arima_cernet_clip[false].log &

# CUDA_VISIBLE_DEVICES=4 python training/train_prophet.py --data data/Abilene-OD_pair.csv --clip &> logs/prophet_abilene_clip[true].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_prophet.py --data data/GEANT-OD_pair.csv --clip &> logs/prophet_geant_clip[true].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_prophet.py --data data/CERNET-OD_pair.csv --clip &> logs/prophet_cernet_clip[true].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_prophet.py --data data/Abilene-OD_pair.csv &> logs/prophet_abilene_clip[false].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_prophet.py --data data/GEANT-OD_pair.csv &> logs/prophet_geant_clip[false].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_prophet.py --data data/CERNET-OD_pair.csv &> logs/prophet_cernet_clip[false].log &

# CUDA_VISIBLE_DEVICES=5 python training/train_cross.py --data data/Abilene-OD_pair.csv --clip --norm roll &> logs/cross_abilene_clip[true]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_cross.py --data data/GEANT-OD_pair.csv --clip --norm roll &> logs/cross_geant_clip[true]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=7 python training/train_cross.py --data data/CERNET-OD_pair.csv --clip --norm roll &> logs/cross_cernet_clip[true]_norm[roll].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_cross.py --data data/Abilene-OD_pair.csv --norm roll &> logs/cross_abilene_clip[false]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_cross.py --data data/GEANT-OD_pair.csv --norm roll &> logs/cross_geant_clip[false]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_cross.py --data data/CERNET-OD_pair.csv --norm roll &> logs/cross_cernet_clip[false]_norm[roll].log &

# CUDA_VISIBLE_DEVICES=5 python training/train_cross.py --data data/Abilene-OD_pair.csv --clip --norm glob &> logs/cross_abilene_clip[true]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_cross.py --data data/GEANT-OD_pair.csv --clip --norm glob &> logs/cross_geant_clip[true]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=7 python training/train_cross.py --data data/CERNET-OD_pair.csv --clip --norm glob &> logs/cross_cernet_clip[true]_norm[glob].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_cross.py --data data/Abilene-OD_pair.csv --norm glob &> logs/cross_abilene_clip[false]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_cross.py --data data/GEANT-OD_pair.csv --norm glob &> logs/cross_geant_clip[false]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_cross.py --data data/CERNET-OD_pair.csv --norm glob &> logs/cross_cernet_clip[false]_norm[glob].log &

# CUDA_VISIBLE_DEVICES=5 python training/train_cross.py --data data/Abilene-OD_pair.csv --clip --norm indv &> logs/cross_abilene_clip[true]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_cross.py --data data/GEANT-OD_pair.csv --clip --norm indv &> logs/cross_geant_clip[true]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=7 python training/train_cross.py --data data/CERNET-OD_pair.csv --clip --norm indv &> logs/cross_cernet_clip[true]_norm[indv].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_cross.py --data data/Abilene-OD_pair.csv --norm indv &> logs/cross_abilene_clip[false]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_cross.py --data data/GEANT-OD_pair.csv --norm indv &> logs/cross_geant_clip[false]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_cross.py --data data/CERNET-OD_pair.csv --norm indv &> logs/cross_cernet_clip[false]_norm[indv].log &

# CUDA_VISIBLE_DEVICES=4 python training/train_convlstm.py --data data/Abilene-OD_pair.csv --clip --norm roll &> logs/convlstm_abilene_clip[true]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_convlstm.py --data data/GEANT-OD_pair.csv --clip --norm roll &> logs/convlstm_geant_clip[true]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=7 python training/train_convlstm.py --data data/CERNET-OD_pair.csv --clip --norm roll &> logs/convlstm_cernet_clip[true]_norm[roll].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_convlstm.py --data data/Abilene-OD_pair.csv --norm roll &> logs/convlstm_abilene_clip[false]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_convlstm.py --data data/GEANT-OD_pair.csv --norm roll &> logs/convlstm_geant_clip[false]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_convlstm.py --data data/CERNET-OD_pair.csv --norm roll &> logs/convlstm_cernet_clip[false]_norm[roll].log &

# CUDA_VISIBLE_DEVICES=4 python training/train_convlstm.py --data data/Abilene-OD_pair.csv --clip --norm glob &> logs/convlstm_abilene_clip[true]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_convlstm.py --data data/GEANT-OD_pair.csv --clip --norm glob &> logs/convlstm_geant_clip[true]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=7 python training/train_convlstm.py --data data/CERNET-OD_pair.csv --clip --norm glob &> logs/convlstm_cernet_clip[true]_norm[glob].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_convlstm.py --data data/Abilene-OD_pair.csv --norm glob &> logs/convlstm_abilene_clip[false]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_convlstm.py --data data/GEANT-OD_pair.csv --norm glob &> logs/convlstm_geant_clip[false]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_convlstm.py --data data/CERNET-OD_pair.csv --norm glob &> logs/convlstm_cernet_clip[false]_norm[glob].log &

# CUDA_VISIBLE_DEVICES=4 python training/train_convlstm.py --data data/Abilene-OD_pair.csv --clip --norm indv &> logs/convlstm_abilene_clip[true]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_convlstm.py --data data/GEANT-OD_pair.csv --clip --norm indv &> logs/convlstm_geant_clip[true]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=7 python training/train_convlstm.py --data data/CERNET-OD_pair.csv --clip --norm indv &> logs/convlstm_cernet_clip[true]_norm[indv].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_convlstm.py --data data/Abilene-OD_pair.csv --norm indv &> logs/convlstm_abilene_clip[false]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_convlstm.py --data data/GEANT-OD_pair.csv --norm indv &> logs/convlstm_geant_clip[false]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_convlstm.py --data data/CERNET-OD_pair.csv --norm indv &> logs/convlstm_cernet_clip[false]_norm[indv].log &

# CUDA_VISIBLE_DEVICES=4 python training/train_lstm.py --data data/Abilene-OD_pair.csv --clip --norm roll &> logs/lstm_abilene_clip[true]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_lstm.py --data data/GEANT-OD_pair.csv --clip --norm roll &> logs/lstm_geant_clip[true]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_lstm.py --data data/CERNET-OD_pair.csv --clip --norm roll &> logs/lstm_cernet_clip[true]_norm[roll].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_lstm.py --data data/Abilene-OD_pair.csv --norm roll &> logs/lstm_abilene_clip[false]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_lstm.py --data data/GEANT-OD_pair.csv --norm roll &> logs/lstm_geant_clip[false]_norm[roll].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_lstm.py --data data/CERNET-OD_pair.csv --norm roll &> logs/lstm_cernet_clip[false]_norm[roll].log &

# CUDA_VISIBLE_DEVICES=4 python training/train_lstm.py --data data/Abilene-OD_pair.csv --clip --norm glob &> logs/lstm_abilene_clip[true]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_lstm.py --data data/GEANT-OD_pair.csv --clip --norm glob &> logs/lstm_geant_clip[true]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_lstm.py --data data/CERNET-OD_pair.csv --clip --norm glob &> logs/lstm_cernet_clip[true]_norm[glob].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_lstm.py --data data/Abilene-OD_pair.csv --norm glob &> logs/lstm_abilene_clip[false]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_lstm.py --data data/GEANT-OD_pair.csv --norm glob &> logs/lstm_geant_clip[false]_norm[glob].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_lstm.py --data data/CERNET-OD_pair.csv --norm glob &> logs/lstm_cernet_clip[false]_norm[glob].log &

# CUDA_VISIBLE_DEVICES=4 python training/train_lstm.py --data data/Abilene-OD_pair.csv --clip --norm indv &> logs/lstm_abilene_clip[true]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=5 python training/train_lstm.py --data data/GEANT-OD_pair.csv --clip --norm indv &> logs/lstm_geant_clip[true]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=6 python training/train_lstm.py --data data/CERNET-OD_pair.csv --clip --norm indv &> logs/lstm_cernet_clip[true]_norm[indv].log &

# CUDA_VISIBLE_DEVICES=1 python training/train_lstm.py --data data/Abilene-OD_pair.csv --norm indv &> logs/lstm_abilene_clip[false]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=2 python training/train_lstm.py --data data/GEANT-OD_pair.csv --norm indv &> logs/lstm_geant_clip[false]_norm[indv].log &
# CUDA_VISIBLE_DEVICES=3 python training/train_lstm.py --data data/CERNET-OD_pair.csv --norm indv &> logs/lstm_cernet_clip[false]_norm[indv].log &

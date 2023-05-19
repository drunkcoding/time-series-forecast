# CUDA_VISIBLE_DEVICES=0 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=tft --multivariate --normalize &> outputs/abilene_tft.log &
# CUDA_VISIBLE_DEVICES=1 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=deepar --multivariate --normalize &> outputs/abilene_deepar.log &
# CUDA_VISIBLE_DEVICES=2 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=rnn --multivariate --normalize &> outputs/abilene_rnn.log &
# CUDA_VISIBLE_DEVICES=3 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=tcn --multivariate --normalize &> outputs/abilene_tcn.log &
# CUDA_VISIBLE_DEVICES=4 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=tf --multivariate --normalize &> outputs/abilene_tf.log &

# CUDA_VISIBLE_DEVICES=5 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=tft --multivariate --normalize &> outputs/geant_tft.log &
# CUDA_VISIBLE_DEVICES=6 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=deepar --multivariate --normalize &> outputs/geant_deepar.log &
# CUDA_VISIBLE_DEVICES=7 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=rnn --multivariate --normalize &> outputs/geant_rnn.log &
# CUDA_VISIBLE_DEVICES=0 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=tcn --multivariate --normalize &> outputs/geant_tcn.log &
# CUDA_VISIBLE_DEVICES=1 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=tf --multivariate --normalize &> outputs/geant_tf.log &

# CUDA_VISIBLE_DEVICES=2 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=tft --multivariate --normalize &> outputs/cernet_tft.log &
# CUDA_VISIBLE_DEVICES=3 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=deepar --multivariate --normalize &> outputs/cernet_deepar.log &
# CUDA_VISIBLE_DEVICES=4 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=rnn --multivariate --normalize &> outputs/cernet_rnn.log &
# CUDA_VISIBLE_DEVICES=5 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=tcn --multivariate --normalize &> outputs/cernet_tcn.log &
# CUDA_VISIBLE_DEVICES=6 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=tf --multivariate --normalize &> outputs/cernet_tf.log &

# nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=arima --normalize &> outputs/abilene_arima.log &
# nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=kalman --normalize &> outputs/abilene_kalman.log &
# nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=prophet --normalize &> outputs/abilene_prophet.log &

# nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=arima --normalize &> outputs/geant_arima.log &
# nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=kalman --normalize &> outputs/geant_kalman.log &
# nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=prophet --normalize &> outputs/geant_prophet.log &

# nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=arima --normalize &> outputs/cernet_arima.log &
# nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=kalman --normalize &> outputs/cernet_kalman.log &
# nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=prophet --normalize &> outputs/cernet_prophet.log &


CUDA_VISIBLE_DEVICES=0 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=tft --multivariate --normalize --remove_outliers &> outputs/abilene_tft_F.log &
CUDA_VISIBLE_DEVICES=1 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=deepar --multivariate --normalize --remove_outliers &> outputs/abilene_deepar_F.log &
CUDA_VISIBLE_DEVICES=2 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=rnn --multivariate --normalize --remove_outliers &> outputs/abilene_rnn_F.log &
CUDA_VISIBLE_DEVICES=3 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=tcn --multivariate --normalize --remove_outliers &> outputs/abilene_tcn_F.log &
CUDA_VISIBLE_DEVICES=4 nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=tf --multivariate --normalize --remove_outliers &> outputs/abilene_tf_F.log &

CUDA_VISIBLE_DEVICES=5 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=tft --multivariate --normalize --remove_outliers &> outputs/geant_tft_F.log &
CUDA_VISIBLE_DEVICES=6 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=deepar --multivariate --normalize --remove_outliers &> outputs/geant_deepar_F.log &
CUDA_VISIBLE_DEVICES=7 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=rnn --multivariate --normalize --remove_outliers &> outputs/geant_rnn_F.log &
CUDA_VISIBLE_DEVICES=0 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=tcn --multivariate --normalize --remove_outliers &> outputs/geant_tcn_F.log &
CUDA_VISIBLE_DEVICES=1 nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=tf --multivariate --normalize --remove_outliers &> outputs/geant_tf_F.log &

CUDA_VISIBLE_DEVICES=2 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=tft --multivariate --normalize --remove_outliers &> outputs/cernet_tft_F.log &
CUDA_VISIBLE_DEVICES=3 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=deepar --multivariate --normalize --remove_outliers &> outputs/cernet_deepar_F.log &
CUDA_VISIBLE_DEVICES=4 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=rnn --multivariate --normalize --remove_outliers &> outputs/cernet_rnn_F.log &
CUDA_VISIBLE_DEVICES=5 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=tcn --multivariate --normalize --remove_outliers &> outputs/cernet_tcn_F.log &
CUDA_VISIBLE_DEVICES=6 nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=tf --multivariate --normalize --remove_outliers &> outputs/cernet_tf_F.log &

nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=arima --normalize --remove_outliers &> outputs/abilene_arima_F.log &
nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=kalman --normalize --remove_outliers &> outputs/abilene_kalman_F.log &
nohup python tests/test_models.py --input_folder data/Abilene-OD_pair.csv --model=prophet --normalize --remove_outliers &> outputs/abilene_prophet_F.log &

nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=arima --normalize --remove_outliers &> outputs/geant_arima_F.log &
nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=kalman --normalize --remove_outliers &> outputs/geant_kalman_F.log &
nohup python tests/test_models.py --input_folder data/GEANT-OD_pair.csv --model=prophet --normalize --remove_outliers &> outputs/geant_prophet_F.log &

nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=arima --normalize --remove_outliers &> outputs/cernet_arima_F.log &
nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=kalman --normalize --remove_outliers &> outputs/cernet_kalman_F.log &
nohup python tests/test_models.py --input_folder data/CERNET-OD_pair.csv --model=prophet --normalize --remove_outliers &> outputs/cernet_prophet_F.log &



CUDA_VISIBLE_DEVICES=4 python training/train_arima.py --data data/Abilene-OD_pair.csv --clip --norm roll &> logs/arima_abilene_clip[true]_norm[roll].log &
CUDA_VISIBLE_DEVICES=5 python training/train_arima.py --data data/GEANT-OD_pair.csv --clip --norm roll &> logs/arima_geant_clip[true]_norm[roll].log &
CUDA_VISIBLE_DEVICES=6 python training/train_arima.py --data data/CERNET-OD_pair.csv --clip --norm roll &> logs/arima_cernet_clip[true]_norm[roll].log &

CUDA_VISIBLE_DEVICES=1 python training/train_arima.py --data data/Abilene-OD_pair.csv --norm roll &> logs/arima_abilene_clip[false]_norm[roll].log &
CUDA_VISIBLE_DEVICES=2 python training/train_arima.py --data data/GEANT-OD_pair.csv --norm roll &> logs/arima_geant_clip[false]_norm[roll].log &
CUDA_VISIBLE_DEVICES=3 python training/train_arima.py --data data/CERNET-OD_pair.csv --norm roll &> logs/arima_cernet_clip[false]_norm[roll].log &

CUDA_VISIBLE_DEVICES=4 python training/train_arima.py --data data/Abilene-OD_pair.csv --clip --norm glob &> logs/arima_abilene_clip[true]_norm[glob].log &
CUDA_VISIBLE_DEVICES=5 python training/train_arima.py --data data/GEANT-OD_pair.csv --clip --norm glob &> logs/arima_geant_clip[true]_norm[glob].log &
CUDA_VISIBLE_DEVICES=6 python training/train_arima.py --data data/CERNET-OD_pair.csv --clip --norm glob &> logs/arima_cernet_clip[true]_norm[glob].log &

CUDA_VISIBLE_DEVICES=1 python training/train_arima.py --data data/Abilene-OD_pair.csv --norm glob &> logs/arima_abilene_clip[false]_norm[glob].log &
CUDA_VISIBLE_DEVICES=2 python training/train_arima.py --data data/GEANT-OD_pair.csv --norm glob &> logs/arima_geant_clip[false]_norm[glob].log &
CUDA_VISIBLE_DEVICES=3 python training/train_arima.py --data data/CERNET-OD_pair.csv --norm glob &> logs/arima_cernet_clip[false]_norm[glob].log &

CUDA_VISIBLE_DEVICES=4 python training/train_arima.py --data data/Abilene-OD_pair.csv --clip --norm indv &> logs/arima_abilene_clip[true]_norm[indv].log &
CUDA_VISIBLE_DEVICES=5 python training/train_arima.py --data data/GEANT-OD_pair.csv --clip --norm indv &> logs/arima_geant_clip[true]_norm[indv].log &
CUDA_VISIBLE_DEVICES=6 python training/train_arima.py --data data/CERNET-OD_pair.csv --clip --norm indv &> logs/arima_cernet_clip[true]_norm[indv].log &

CUDA_VISIBLE_DEVICES=1 python training/train_arima.py --data data/Abilene-OD_pair.csv --norm indv &> logs/arima_abilene_clip[false]_norm[indv].log &
CUDA_VISIBLE_DEVICES=2 python training/train_arima.py --data data/GEANT-OD_pair.csv --norm indv &> logs/arima_geant_clip[false]_norm[indv].log &
CUDA_VISIBLE_DEVICES=3 python training/train_arima.py --data data/CERNET-OD_pair.csv --norm indv &> logs/arima_cernet_clip[false]_norm[indv].log &

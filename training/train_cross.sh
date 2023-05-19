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

CUDA_VISIBLE_DEVICES=5 python training/train_cross.py --data data/Abilene-OD_pair.csv --clip --norm indv &> logs/cross_abilene_clip[true]_norm[indv].log &
CUDA_VISIBLE_DEVICES=6 python training/train_cross.py --data data/GEANT-OD_pair.csv --clip --norm indv &> logs/cross_geant_clip[true]_norm[indv].log &
CUDA_VISIBLE_DEVICES=7 python training/train_cross.py --data data/CERNET-OD_pair.csv --clip --norm indv &> logs/cross_cernet_clip[true]_norm[indv].log &

CUDA_VISIBLE_DEVICES=1 python training/train_cross.py --data data/Abilene-OD_pair.csv --norm indv &> logs/cross_abilene_clip[false]_norm[indv].log &
CUDA_VISIBLE_DEVICES=2 python training/train_cross.py --data data/GEANT-OD_pair.csv --norm indv &> logs/cross_geant_clip[false]_norm[indv].log &
CUDA_VISIBLE_DEVICES=3 python training/train_cross.py --data data/CERNET-OD_pair.csv --norm indv &> logs/cross_cernet_clip[false]_norm[indv].log &

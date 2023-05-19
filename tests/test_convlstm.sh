# python tests/test_convlstm.py --input_folder data/Abilene-OD_pair.csv &> outputs/abilene_convlstm_O.log
# python tests/test_convlstm.py --input_folder data/GEANT-OD_pair.csv  &> outputs/geant_convlstm_O.log
# python tests/test_convlstm.py --input_folder data/CERNET-OD_pair.csv &> outputs/cernet_convlstm_O.log

python tests/test_convlstm.py --input_folder data/Abilene-OD_pair.csv --remove_outliers &> outputs/abilene_convlstm_F.log
python tests/test_convlstm.py --input_folder data/GEANT-OD_pair.csv --remove_outliers &> outputs/geant_convlstm_F.log
python tests/test_convlstm.py --input_folder data/CERNET-OD_pair.csv --remove_outliers &> outputs/cernet_convlstm_F.log




# the_config = {                 
#    "model_name": "CustomTransformerDecoder",
#    "n_targets": 12,
#    "model_type": "PyTorch",
#     "model_params": {
#       "n_time_series":19,
#       "seq_length":26,
#       "output_seq_length": 1, 
#       "output_dim":12,
#       "n_layers_encoder": 6
#      }, 
#     "dataset_params":
#     { "class": "GeneralClassificationLoader",
#       "n_classes": 9,
#        "training_path": "/kaggle/working/flow-forecast/train.csv",
#        "validation_path": "/kaggle/working/flow-forecast/train.csv",
#        "test_path": "/kaggle/working/flow-forecast/test.csv",
#        "sequence_length":26,
#        "batch_size":4,
#        "forecast_history":26,
#        "train_end": 4500,
#        "valid_start":4501,
#        "valid_end": 7000,
#        "target_col": ["labels"],
#        "relevant_cols": ["labels"] + df.columns.tolist()[:19],
#        "scaler": "StandardScaler", 
#        "interpolate": False
#     },

#     "training_params":
#     {
#        "criterion":"CrossEntropyLoss",
#        "optimizer": "Adam",
#        "optim_params":
#        {},
#        "lr": 0.3,
#        "epochs": 4,
#        "batch_size":4
#     },
#     "GCS": False,
   
#     "wandb": {
#        "name": "flood_forecast_circleci",
#        "tags": ["dummy_run", "circleci", "multi_head", "classification"],
#        "project": "repo-flood_forecast"
#     },
#    "forward_params":{},
#    "metrics":["CrossEntropyLoss"]
# }

                                                                                                                                                                                                               
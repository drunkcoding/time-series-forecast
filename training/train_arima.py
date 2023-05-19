import numpy as np
from transformers import HfArgumentParser
from tqdm import tqdm, trange
import os

from models import ARIMAModule
from utils import TMLightning, ModelConfig
from pmdarima.arima import auto_arima, ARIMA

parser = HfArgumentParser(ModelConfig)
args = parser.parse_args_into_dataclasses()[0]

model_name = "cross"
ckpt_name = (
    f"{args.dataset}_outputs_{model_name}_clip[{args.clip}]_norm[{args.norm}].npy"
)

model = ARIMAModule()
model = TMLightning(model, args.data, clip=args.clip, norm=args.norm, online=True)

# fold_id = model.fold_idx
X_train = model.X_train.cpu().numpy()
y_train = model.y_train.cpu().numpy()
X_test = model.X_test.cpu().numpy()
y_test = model.y_test.cpu().numpy()
test_rolling_std = model.test_rolling_std
train_rolling_std = model.train_rolling_std
train_rolling_mean = model.train_rolling_mean
test_rolling_mean = model.test_rolling_mean

rolling_mean = np.concatenate([train_rolling_mean, test_rolling_mean], axis=0)
rolling_std = np.concatenate([train_rolling_std, test_rolling_std], axis=0)

print(test_rolling_std.shape)
test_outputs = []
train_outputs = []

num_features = X_train.shape[-1]
for k in range(num_features):
    model = ARIMA(order=(1, 1, 1))
    model.fit(y_train[:, k], X_train[..., k])
    train_output = model.predict_in_sample(X_train[..., k])
    test_output = model.predict(X_test.shape[0], X_test[..., k])
    print(train_output.shape, test_output.shape)
    train_outputs.append(train_output.reshape(-1, 1))
    test_outputs.append(test_output.reshape(-1, 1))
    train_mae = np.mean(np.abs((train_output - y_train[:, k]) * train_rolling_std[:, k])[y_train[:, k] != 0])
    test_mae = np.mean(np.abs((test_output - y_test[:, k]) * test_rolling_std[:, k])[y_test[:, k] != 0])
    print(f"Test MAE: {test_mae:.4f}, Train MAE: {train_mae:.4f} for feature {k}")
    
    # save model
    # model.save(f"training/models/{ckpt_name}_{k}.pkl")

train_outputs = np.concatenate(train_outputs, axis=1)
test_outputs = np.concatenate(test_outputs, axis=1)

outputs = np.concatenate([train_outputs, test_outputs], axis=0)
targets = np.concatenate([y_train, y_test], axis=0)
print(outputs.shape, targets.shape)

outputs[targets == 0] = np.nan
targets[targets == 0] = np.nan

outputs = outputs * rolling_std + rolling_mean
targets = targets * rolling_std + rolling_mean

outputs[np.isnan(outputs)] = 0
targets[np.isnan(targets)] = 0

np.save(f"training/outputs/{ckpt_name}.npy", outputs, allow_pickle=False)
np.save(f"training/outputs/{ckpt_name}.npy", targets, allow_pickle=False)

train_mae = np.mean(np.abs((train_outputs - y_train) * train_rolling_std)[y_train != 0])
test_mae = np.mean(np.abs((test_outputs - y_test) * test_rolling_std)[y_test != 0])
print(f"Test MAE: {test_mae:.4f}, Train MAE: {train_mae:.4f}")

# from cuml.tsa.arima import ARIMA
import torch
import numpy as np
import torch.nn as nn
from pmdarima.arima import auto_arima, ARIMA


class ARIMAModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y, x_test):
        if isinstance(x, torch.Tensor):
            device = x.device
            X_train = x.detach().cpu().numpy()
            y_train = y.detach().cpu().numpy()
            X_test = x_test.detach().cpu().numpy()
        else:
            X_train = x
            y_train = y
            X_test = x_test
        outputs = []
        for i, x_input in enumerate(X_train):
            print(x_input.shape, y_train[i].shape)
            model = auto_arima(
                y=y_train[i], X=x_input, trace=True, suppress_warnings=True
            )
            output = model.predict(1, X_test)
            outputs.append(model.predict(1)[None, ...])
        outputs = np.concatenate(outputs)
        print(outputs.shape)
        if isinstance(x, torch.Tensor):
            outputs = torch.from_numpy(outputs).to(device)
        return outputs

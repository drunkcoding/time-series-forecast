from prophet import Prophet
import torch
import torch.nn as nn
import pandas as pd

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

class ProphetModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x):
        self.model = Prophet()
        ds = pd.date_range(start="2020-01-01", periods=len(x), freq="D")
        df = pd.DataFrame({"ds": ds, "y": x})
        self.model.fit(df)

    def forward(self, periods):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast.yhat.values
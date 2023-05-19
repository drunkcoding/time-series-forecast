import numpy as np
from darts import TimeSeries
import pandas as pd



# calculate RMSE Error (Root Mean Squared Error), tolerate nan values
def rmse(predictions, targets):
    return np.sqrt(np.nanmean((predictions - targets) ** 2))

# calculate MAE Error (Mean Absolute Error), tolerate nan values
def mae(predictions, targets):
    return np.nanmean(np.abs(predictions - targets))

# calculate MAPE Error (Mean Absolute Percentage Error), tolerate nan values
def mape(predictions, targets):
    return np.nanmean(np.abs((predictions - targets) / targets)) * 100

# calculate MASE Error (Mean Absolute Scaled Error), tolerate nan values
def mase(predictions, targets, train):
    return np.nanmean(np.abs((predictions - targets) / train))

# calculate SMAPE Error (Symmetric Mean Absolute Percentage Error), tolerate nan values
def smape(predictions, targets):
    return np.nanmean(np.abs((predictions - targets) / ((np.abs(predictions) + np.abs(targets)) / 2))) * 100

# calculate MASE Error (Mean Absolute Scaled Error), tolerate nan values
def mase(predictions, targets, train):
    return np.nanmean(np.abs((predictions - targets) / train))

def df_cleanup(df:pd.DataFrame):
    value_columns = [col for col in df.columns if col != "timestamps"]
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    # drop rows that has all nan, except timestamps column
    # df = df.dropna(axis=0, thresh=int(num_cols / 2), subset=value_columns)
    df = df.dropna(how="all", subset=df.columns[:-1])
    # drop columns that has all nan
    df = df.dropna(axis=1, thresh=int(num_rows / 2))

    return df

def prepare_timeseries(df: pd.DataFrame, time_col: str, target_col: str, freq: str, train_size: int):
    
    series = TimeSeries.from_dataframe(
        df,
        time_col,
        target_col,
        fill_missing_dates=True,
        freq=freq,
        fillna_value=0,
    )
    df = series.pd_dataframe().reset_index()
    train_size = int(len(df.index.values) * 0.6)
    
    df_train = df[:train_size]
    df_val = df[train_size:]

    # print(df_train.shape)
    # print(df_val.shape)

    nan_idx = np.isnan(df_val[target_col].values)

    train = TimeSeries.from_dataframe(
        df_train,
        time_col,
        target_col,
        fill_missing_dates=True,
        freq=freq,
        fillna_value=0,
    )

    val = TimeSeries.from_dataframe(
        df_val,
        time_col,
        target_col,
        fill_missing_dates=True,
        freq=freq,
    )

    return train, val, nan_idx
